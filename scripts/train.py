"""
scripts/train.py
=================
Point d'entrée de PRODUCTION — charge les vrais Qwen2.5 depuis Hugging Face
Hub (nécessite internet, donc à lancer sur ton GPU distant, pas dans ce
sandbox de dev). La mécanique a été validée séparément et localement dans
tests/test_pipeline.py.

Usage :
    python scripts/train.py --config configs/arcd.yaml
    python scripts/train.py --config configs/baseline.yaml
"""

import argparse
import csv
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tokenizer import get_tokenizer
from datasets.prompts import TOY_EXAMPLES
from datasets.dataloader import PromptResponseDataset, make_collate_fn
from models.teacher import build_teachers, DEFAULT_TEACHER_NAMES
from models.student import build_student
from trainers.baseline import train_student_alone
from trainers.hinton import train_hinton_kd
from trainers.arcd import train_arcd


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_examples(path: str = None) -> list:
    """
    Charge les paires (prompt, réponse). Par défaut, utilise le petit jeu
    d'exemples de démonstration (datasets/prompts.py).

    Pour un vrai entraînement, remplace ceci par le chargement d'un vrai
    dataset (ex: un fichier JSONL local, ou `datasets.load_dataset(...)`
    de Hugging Face si le GPU distant y a accès).
    """
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
    vocab_size = len(tokenizer)
    print(f"Tokenizer: {cfg['data']['tokenizer_name']} — vocab_size={vocab_size}")

    # --- Données ---
    examples = load_examples(cfg["data"].get("examples_path"))
    dataset = PromptResponseDataset(examples, tokenizer, max_length=cfg["data"]["max_length"])
    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                               shuffle=True, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} exemples")

    # --- Student (from scratch, dimensionné sur le vocabulaire partagé) ---
    student_cfg = cfg["models"]["student"]
    student = build_student(
        vocab_size=vocab_size,
        n_embd=student_cfg["n_embd"],
        n_layer=student_cfg["n_layer"],
        n_head=student_cfg["n_head"],
        n_positions=student_cfg["n_positions"],
    )

    # --- Régime d'entraînement sélectionné par la config ---
    regime = cfg["training"]["regime"]  # "student_alone" | "hinton_kd" | "arcd"
    print(f"\nRégime: {regime}")

    if regime == "student_alone":
        train_student_alone(student, train_loader, epochs=cfg["training"]["epochs"],
                             lr=cfg["training"]["lr"], device=device)

    elif regime == "hinton_kd":
        teachers = build_teachers([cfg["models"]["teacher_names"][0]])  # un seul Teacher
        train_hinton_kd(student, teachers[0], train_loader, epochs=cfg["training"]["epochs"],
                         lr=cfg["training"]["lr"], temperature=cfg["training"]["temperature"],
                         alpha=cfg["training"]["hinton_alpha"], device=device)

    elif regime == "arcd":
        teachers = build_teachers(cfg["models"].get("teacher_names", DEFAULT_TEACHER_NAMES))
        train_arcd(student, teachers, train_loader, epochs=cfg["training"]["epochs"],
                   lr=cfg["training"]["lr"], temperature=cfg["training"]["temperature"], device=device)

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
