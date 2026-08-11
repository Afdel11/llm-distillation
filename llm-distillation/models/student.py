"""
models/student.py
==================
Student "MiniQwen" : même architecture que les Teachers (Qwen2 — RoPE,
RMSNorm, GQA), mais avec beaucoup moins de couches/dimensions, poids
aléatoires (entraîné from scratch — c'est tout le sujet de la distillation).

Pourquoi la même architecture que les Teachers plutôt qu'un GPT-2 miniature :
ce n'est pas une nécessité technique pour que la distillation fonctionne (la
littérature distille couramment entre familles différentes), mais ça évite
de mélanger deux inductive bias différents (encodage positionnel, attention,
normalisation) sans raison, et ça permet une phrase de positionnement plus
propre dans le mémoire : "le Student reprend l'architecture des Teachers
avec une capacité réduite".

Contrainte non négociable, elle, réelle : `vocab_size = len(tokenizer)` du
tokenizer partagé (voir datasets/tokenizer.py) — c'est le tokenizer commun,
pas l'architecture, qui garantit l'alignement des indices dans ARCD.
"""

import torch
import torch.nn as nn


def build_student(vocab_size: int, hidden_size: int = 512, num_hidden_layers: int = 4,
                   num_attention_heads: int = 8, num_key_value_heads: int = 4,
                   intermediate_size: int = 2048, max_position_embeddings: int = 2048) -> nn.Module:
    """
    MiniQwen, poids aléatoires (from scratch).

    Args:
        vocab_size: DOIT correspondre exactement à celui du tokenizer partagé
                    avec les Teachers (ex: len(AutoTokenizer.from_pretrained(...))).
        num_key_value_heads: < num_attention_heads -> Grouped Query Attention,
                    comme dans les vrais Qwen2.5 (réduit encore la taille/le coût).
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    return Qwen2ForCausalLM(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
