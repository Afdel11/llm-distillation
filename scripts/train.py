"""
scripts/train.py
=================
Point d'entrée de PRODUCTION — charge les vrais Qwen2.5 depuis Hugging Face
Hub (nécessite internet, donc à lancer sur ton GPU distant, pas dans ce
sandbox de dev). La mécanique a été validée séparément et localement dans
tests/test_pipeline.py.

Pour le régime "arcd", si outputs/teacher_cache.pt existe déjà (voir
scripts/build_teacher_cache.py), il est utilisé automatiquement — aucun
forward Teacher pendant l'entraînement. Sinon, les Teachers tournent en
direct à chaque batch (plus simple pour un premier essai, plus lent en
pratique).

Usage :
    python scripts/train.py --config configs/arcd.yaml
    python scripts/train.py --config configs/baseline.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tokenizer import get_tokenizer
from datasets.prompts import TOY_EXAMPLES
from datasets.dataloader import PromptResponseDataset, make_collate_fn, make_cached_collate_fn
from datasets.cache import load_teacher_cache
from models.teacher import TeacherEnsemble
from models.student import build_student
from trainers.baseline import train_student_alone
from trainers.hinton import train_hinton_kd
from trainers.arcd import train_arcd


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_vocab_size(teacher_names: dict) -> int:
    """
    Récupère model.config.vocab_size pour chaque Teacher nommé (léger : ne
    télécharge que config.json, pas les poids), et vérifie qu'ils coïncident
    tous. Une divergence signalerait des Teachers qui ne partagent en fait
    PAS le même tokenizer/vocabulaire malgré leurs noms — c'est exactement le
    bug qu'on a déjà neutralisé une fois en choisissant deux Qwen2.5 plutôt
    que Qwen + SmolLM2 ; cette vérification le détecterait immédiatement si
    quelqu'un change un nom de Teacher dans la config sans y penser.
    """
    from transformers import AutoConfig

    sizes = {name: AutoConfig.from_pretrained(hf_name).vocab_size
             for name, hf_name in teacher_names.items()}
    unique_sizes = set(sizes.values())
    assert len(unique_sizes) == 1, (
        f"Les Teachers n'ont pas le même vocab_size de sortie: {sizes}. "
        "Vérifie qu'ils partagent bien le même tokenizer avant de continuer."
    )
    return unique_sizes.pop()


def load_examples(path: str = None) -> list:
    if path is None:
        return TOY_EXAMPLES
    import json
    with open(path, "r") as f:
        return json.load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["training"]["seed"])

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("ATTENTION : cuda demandé mais indisponible, retour sur cpu.")
        device = "cpu"
    print(f"Device: {device}")

    # --- Tokenizer partagé ---
    tokenizer = get_tokenizer(cfg["data"]["tokenizer_name"])

    # IMPORTANT : le Student doit être dimensionné sur model.config.vocab_size
    # (la taille RÉELLE de la matrice de sortie des Teachers), pas sur
    # len(tokenizer). Qwen2.5 arrondit son vocabulaire de sortie à un multiple
    # de 128 pour l'efficacité GPU (151936), alors que len(tokenizer) ne compte
    # que les tokens réellement utilisables (151665) -> sans ça, ARCDLoss et
    # Hinton KD crashent sur un mismatch de taille entre logits Teacher/Student.
    # On vérifie au passage que tous les Teachers nommés partagent bien le
    # même vocab_size de sortie (sinon ils ne partagent pas vraiment le même
    # tokenizer, malgré leurs noms — exactement le bug qu'on a déjà chassé une
    # fois avec Qwen/SmolLM2).
    model_vocab_size = get_model_vocab_size(cfg["models"]["teacher_names"])
    print(f"Tokenizer: {cfg['data']['tokenizer_name']} — "
          f"len(tokenizer)={len(tokenizer)}, model.config.vocab_size={model_vocab_size}")

    # --- Données ---
    examples = load_examples(cfg["data"].get("examples_path"))
    dataset = PromptResponseDataset(examples, tokenizer, max_length=cfg["data"]["max_length"])
    print(f"Dataset: {len(dataset)} exemples")

    # --- Student MiniQwen (from scratch, dimensionné sur la SORTIE réelle des Teachers) ---
    student_cfg = cfg["models"]["student"]
    student = build_student(
        vocab_size=model_vocab_size,
        hidden_size=student_cfg["hidden_size"],
        num_hidden_layers=student_cfg["num_hidden_layers"],
        num_attention_heads=student_cfg["num_attention_heads"],
        num_key_value_heads=student_cfg["num_key_value_heads"],
        intermediate_size=student_cfg["intermediate_size"],
    )

    # --- Régime d'entraînement sélectionné par la config ---
    regime = cfg["training"]["regime"]  # "student_alone" | "hinton_kd" | "arcd"
    print(f"\nRégime: {regime}")

    if regime == "student_alone":
        collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
        train_loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                                   shuffle=True, collate_fn=collate_fn)
        train_student_alone(student, train_loader, epochs=cfg["training"]["epochs"],
                             lr=cfg["training"]["lr"], device=device)

    elif regime == "hinton_kd":
        collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
        train_loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                                   shuffle=True, collate_fn=collate_fn)
        # un seul Teacher pour cette baseline : le premier nommé dans la config
        first_name = list(cfg["models"]["teacher_names"].keys())[0]
        ensemble = TeacherEnsemble({first_name: cfg["models"]["teacher_names"][first_name]}, device=device)
        teacher = ensemble.models[first_name]
        train_hinton_kd(student, teacher, train_loader, epochs=cfg["training"]["epochs"],
                         lr=cfg["training"]["lr"], temperature=cfg["training"]["temperature"],
                         alpha=cfg["training"]["hinton_alpha"], device=device)

    elif regime == "arcd":
        cache_path = cfg["output"].get("teacher_cache_path", "outputs/teacher_cache.pt")
        if os.path.exists(cache_path):
            print(f"Cache de logits Teachers trouvé: {cache_path} — aucun forward Teacher pendant l'entraînement.")
            teacher_logits_cache = load_teacher_cache(cache_path)
            collate_fn = make_cached_collate_fn(pad_token_id=tokenizer.pad_token_id,
                                                 teacher_logits_cache=teacher_logits_cache)
            train_loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                                       shuffle=True, collate_fn=collate_fn)
            train_arcd(student, train_loader, epochs=cfg["training"]["epochs"],
                       lr=cfg["training"]["lr"], temperature=cfg["training"]["temperature"], device=device)
        else:
            print(f"Pas de cache trouvé ({cache_path}) — les Teachers tourneront à chaque batch. "
                  f"Lance scripts/build_teacher_cache.py avant un vrai entraînement pour éviter ça.")
            collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
            train_loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                                       shuffle=True, collate_fn=collate_fn)
            ensemble = TeacherEnsemble(cfg["models"]["teacher_names"], device=device)
            train_arcd(student, train_loader, epochs=cfg["training"]["epochs"],
                       lr=cfg["training"]["lr"], temperature=cfg["training"]["temperature"],
                       device=device, teacher_ensemble=ensemble)

    else:
        raise ValueError(f"regime inconnu: {regime!r}")

    # --- Sauvegarde ---
    out_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"student_{regime}.pt")
    torch.save(student.state_dict(), ckpt_path)
    print(f"\nStudent sauvegardé: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arcd.yaml")
    args = parser.parse_args()
    main(args.config)
