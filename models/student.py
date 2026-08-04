"""
models/student.py
==================
Student entraîné from scratch (pas de pré-entraînement — c'est tout le
sujet de la distillation). Son architecture n'a PAS besoin d'être la même
que celle des Teachers Qwen2.5 : elle peut être plus petite et différente
(ici un GPT-2 miniature). La seule contrainte non négociable est que sa
couche de sortie ait exactement `vocab_size = len(tokenizer)` du tokenizer
partagé (voir datasets/tokenizer.py) — c'est le partage du tokenizer, pas
de l'architecture, qui garantit l'alignement des indices dans ARCD.
"""

import torch
import torch.nn as nn


def build_student(vocab_size: int, n_embd: int = 256, n_layer: int = 4,
                   n_head: int = 4, n_positions: int = 512) -> nn.Module:
    """
    Student GPT-2 miniature, poids aléatoires (from scratch).

    Args:
        vocab_size: DOIT correspondre exactement à celui du tokenizer partagé
                    avec les Teachers (ex: len(AutoTokenizer.from_pretrained(...))).
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_positions,
    )
    return GPT2LMHeadModel(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
