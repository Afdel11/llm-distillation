"""
scripts/prepare_dataset.py
============================
Télécharge un dataset d'instructions HuggingFace, le reformate au format
{"prompt": ..., "response": ...} attendu par ce projet, filtre par longueur
réelle en tokens (avec le tokenizer partagé), échantillonne N exemples, et
écrit un split train/val en JSON. Nécessite internet -> à lancer sur le GPU
distant, pas dans un sandbox de dev sans réseau.

⚠️  COLLISION DE NOM CRITIQUE — LIRE AVANT DE MODIFIER CE FICHIER ⚠️
Ce projet a un package local `datasets/` (datasets/tokenizer.py,
datasets/dataloader.py, etc.), installé en mode éditable (`pip install -e .`,
voir pyproject.toml). Résultat : dans CE venv, `import datasets` renvoie
TOUJOURS notre package local, jamais la librairie HuggingFace `datasets`
(celle qui fournit `load_dataset`) — et ce, PARTOUT dans le venv, pas
seulement dans les scripts qui manipulent sys.path. Ce n'est pas un problème
ponctuel : c'est structurel tant que le package local s'appelle `datasets`.

Ce script contourne le problème en ne dépendant JAMAIS de la librairie HF
`datasets` : on interroge directement l'API publique datasets-server (qui
donne les URLs Parquet réelles de n'importe quel dataset HF, peu importe la
convention de nommage interne du repo), et on télécharge/parse avec
`requests` + `pandas`. Zéro dépendance sur un module qui s'appelle `datasets`.

Si un jour ce projet a vraiment besoin de la librairie HF `datasets` (par
exemple pour du streaming sur un très gros corpus), la seule solution
propre sera de renommer notre package `datasets/` -> autre chose (ex:
`data_pipeline/`) partout dans le projet. Pas fait ici pour ne pas
perturber les expériences en cours.

Usage :
    python scripts/prepare_dataset.py \\
        --hf_dataset jpacifico/French-Alpaca-dataset-Instruct-55K \\
        --n_samples 1000 \\
        --max_length 256 \\
        --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct \\
        --output_dir outputs/data
"""

import argparse
import io
import json
import os
import random

import requests
import pandas as pd


def fetch_parquet_urls(hf_dataset: str, split: str) -> list:
    """
    Interroge l'API datasets-server pour obtenir les URLs Parquet réelles
    d'un dataset HF, sans jamais importer la librairie `datasets`.
    """
    resp = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": hf_dataset},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    urls = [f["url"] for f in data.get("parquet_files", []) if f.get("split") == split]
    if not urls:
        available_splits = sorted({f.get("split") for f in data.get("parquet_files", [])})
        raise RuntimeError(
            f"Aucun fichier Parquet trouvé pour split={split!r} sur {hf_dataset!r}. "
            f"Splits disponibles: {available_splits}"
        )
    return urls


def load_dataframe(hf_dataset: str, split: str) -> pd.DataFrame:
    urls = fetch_parquet_urls(hf_dataset, split)
    print(f"  {len(urls)} fichier(s) Parquet trouvé(s)")
    frames = []
    for url in urls:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(resp.content)))
    return pd.concat(frames, ignore_index=True)


def format_alpaca_example(row: dict) -> dict:
    """Convertit une ligne Alpaca ({instruction, input, output}) vers {prompt, response}."""
    instruction = str(row["instruction"]).strip()
    input_text = str(row.get("input") or "").strip()
    output = str(row["output"]).strip()

    # Le dataset marque parfois l'absence d'input par "Aucun" plutôt qu'une
    # chaîne vide -> on normalise les deux cas.
    has_input = input_text and input_text.lower() not in ("aucun", "none", "n/a", "nan")

    if has_input:
        prompt = f"### Instruction:\n{instruction}\n\n### Entrée:\n{input_text}\n\n### Réponse:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Réponse:\n"

    return {"prompt": prompt, "response": " " + output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, default="jpacifico/French-Alpaca-dataset-Instruct-55K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n_samples", type=int, default=1000,
                         help="Nombre d'exemples à garder APRÈS filtrage par longueur.")
    parser.add_argument("--max_length", type=int, default=256,
                         help="Longueur max en tokens (prompt+réponse) — doit correspondre à "
                              "data.max_length dans configs/*.yaml.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Téléchargement de {args.hf_dataset} (split={args.split}) via datasets-server...")
    df = load_dataframe(args.hf_dataset, args.split)
    print(f"  {len(df)} lignes brutes")

    print(f"Chargement du tokenizer {args.tokenizer_name} (pour le filtrage par longueur)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Reformatage + filtrage par longueur réelle en tokens...")
    examples = []
    for row in df.to_dict(orient="records"):
        try:
            ex = format_alpaca_example(row)
        except (KeyError, AttributeError):
            continue  # ligne malformée -> ignorée plutôt que de planter tout le script

        if not ex["prompt"].strip() or not ex["response"].strip():
            continue

        n_tokens = len(tokenizer(ex["prompt"] + ex["response"])["input_ids"])
        if n_tokens <= args.max_length:
            examples.append(ex)

    print(f"  {len(examples)} exemples valides après filtrage (<= {args.max_length} tokens)")

    random.shuffle(examples)
    examples = examples[:args.n_samples]
    print(f"  {len(examples)} exemples retenus (n_samples={args.n_samples})")

    n_val = max(1, int(len(examples) * args.val_fraction))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_examples, f, ensure_ascii=False, indent=2)

    print(f"\nTrain: {len(train_examples)} exemples -> {train_path}")
    print(f"Val:   {len(val_examples)} exemples -> {val_path}")
    print(f"\nDans tes configs (configs/arcd.yaml, hinton.yaml, baseline.yaml), mets :")
    print(f'  data.examples_path: "{train_path}"')


if __name__ == "__main__":
    main()
