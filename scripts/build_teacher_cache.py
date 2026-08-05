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

from datasets.tokenizer import get_tokenizer
from datasets.prompts import TOY_EXAMPLES
from datasets.dataloader import PromptResponseDataset
from datasets.cache import build_teacher_cache, save_teacher_cache
from models.teacher import TeacherEnsemble


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    tokenizer = get_tokenizer(cfg["data"]["tokenizer_name"])

    examples_path = cfg["data"].get("examples_path")
    if examples_path:
        import json
        with open(examples_path, "r") as f:
            examples = json.load(f)
    else:
        examples = TOY_EXAMPLES

    dataset = PromptResponseDataset(examples, tokenizer, max_length=cfg["data"]["max_length"])
    print(f"Dataset: {len(dataset)} exemples")

    ensemble = TeacherEnsemble(cfg["models"]["teacher_names"], device=device)
    print(f"Teachers chargés: {ensemble.teacher_names} — {ensemble.parameter_counts()}")

    print("Calcul des logits Teachers sur tout le dataset (une seule fois)...")
    cache = build_teacher_cache(ensemble, dataset, pad_token_id=tokenizer.pad_token_id,
                                 device=device, batch_size=cfg["data"]["batch_size"])

    cache_path = cfg["output"].get("teacher_cache_path", "outputs/teacher_cache.pt")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_teacher_cache(cache, cache_path)
    print(f"Cache sauvegardé: {cache_path} ({len(cache)} exemples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arcd.yaml")
    args = parser.parse_args()
    main(args.config)
