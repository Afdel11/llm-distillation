"""
scripts/build_teacher_cache.py
================================
Calcule les logits des Teachers sur tout le dataset UNE SEULE FOIS et les
sauvegarde sur disque. À lancer avant scripts/train.py --config configs/arcd.yaml
pour éviter de refaire tourner les Teachers à chaque epoch.

Usage :
    python scripts/build_teacher_cache.py --config configs/arcd.yaml
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.tokenizer import get_tokenizer
from data_pipeline.prompts import TOY_EXAMPLES
from data_pipeline.dataloader import PromptResponseDataset
from data_pipeline.cache import build_teacher_cache, save_teacher_cache
from models.teacher import TeacherEnsemble


def _load_examples(path: str):
    import json
    with open(path, "r") as f:
        return json.load(f)


def _build_and_save(ensemble, examples, tokenizer, cfg, cache_path: str, label: str):
    dataset = PromptResponseDataset(examples, tokenizer, max_length=cfg["data"]["max_length"])
    print(f"[{label}] Dataset: {len(dataset)} exemples")
    cache = build_teacher_cache(ensemble, dataset, pad_token_id=tokenizer.pad_token_id,
                                 device=cfg["training"]["device"], batch_size=cfg["data"]["batch_size"])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_teacher_cache(cache, cache_path)
    print(f"[{label}] Cache sauvegardé: {cache_path} ({len(cache)} exemples)")


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    tokenizer = get_tokenizer(cfg["data"]["tokenizer_name"])

    examples_path = cfg["data"].get("examples_path")
    train_examples = _load_examples(examples_path) if examples_path else TOY_EXAMPLES

    ensemble = TeacherEnsemble(cfg["models"]["teacher_names"], device=device)
    print(f"Teachers chargés: {ensemble.teacher_names} — {ensemble.parameter_counts()}")

    train_cache_path = cfg["output"].get("teacher_cache_path", "outputs/teacher_cache.pt")
    _build_and_save(ensemble, train_examples, tokenizer, cfg, train_cache_path, "train")

    # Cache de validation, distinct du cache d'entraînement (indices propres à
    # chaque dataset -> ne jamais mélanger les deux fichiers de cache).
    val_examples_path = cfg["data"].get("val_examples_path")
    if val_examples_path and os.path.exists(val_examples_path):
        val_examples = _load_examples(val_examples_path)
        val_cache_path = cfg["output"].get("teacher_cache_val_path", "outputs/teacher_cache_val.pt")
        _build_and_save(ensemble, val_examples, tokenizer, cfg, val_cache_path, "val")
    else:
        print("Pas de data.val_examples_path configuré (ou fichier absent) — "
              "pas de cache de validation généré. L'évaluation ne sera pas possible pour ARCD "
              "tant que ce cache n'existe pas (voir configs/arcd.yaml).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arcd.yaml")
    args = parser.parse_args()
    main(args.config)
