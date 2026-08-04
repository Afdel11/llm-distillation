"""
models/teacher.py
==================
Deux Teachers Qwen2.5-Instruct de tailles différentes, EXPLICITEMENT choisis
dans la même famille pour partager le même tokenizer / vocabulaire (voir
datasets/tokenizer.py). C'est cette contrainte, pas la similarité
d'architecture, qui garantit que la médiane pondérée d'ARCD compare des
distributions indexées de façon cohérente token par token.

  - "large" : Qwen/Qwen2.5-1.5B-Instruct
  - "small" : Qwen/Qwen2.5-0.5B-Instruct

Nécessite un accès internet vers huggingface.co (pas disponible dans ce
sandbox de dev) -> à charger sur le GPU distant. Pour le développement
local sans internet, voir `build_debug_teachers()` en bas de ce fichier,
qui simule la même interface avec des poids aléatoires et un vocabulaire
factice, pour valider la mécanique du pipeline.
"""

import torch
import torch.nn as nn

DEFAULT_TEACHER_NAMES = [
    "Qwen/Qwen2.5-1.5B-Instruct",   # large
    "Qwen/Qwen2.5-0.5B-Instruct",   # small
]


def build_teachers(model_names=None) -> list:
    """
    Charge les Teachers pré-entraînés depuis Hugging Face Hub.
    Nécessite internet (GPU distant).
    """
    from transformers import AutoModelForCausalLM

    model_names = model_names or DEFAULT_TEACHER_NAMES
    teachers = []
    for name in model_names:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(model)
    return teachers


@torch.no_grad()
def get_teacher_logits(teachers: list, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Empile les logits de tous les Teachers pour un batch tokenisé.

    Args:
        input_ids:      (batch, seq_len)
        attention_mask: (batch, seq_len)

    Returns:
        (batch, seq_len, num_teachers, vocab_size)
    """
    all_logits = []
    for teacher in teachers:
        out = teacher(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(out.logits)  # (batch, seq_len, vocab_size)
    return torch.stack(all_logits, dim=2)  # insère l'axe Teachers en position -2


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Version "debug" — AUCUN accès réseau requis, pour valider la mécanique
# du pipeline en local (poids aléatoires, vocabulaire factice).
# ---------------------------------------------------------------------------

def build_debug_teachers(vocab_size: int = 1000, num_teachers: int = 2) -> list:
    """Teachers GPT-2 miniatures, poids aléatoires, construits 100% localement."""
    from transformers import GPT2Config, GPT2LMHeadModel

    teachers = []
    for i in range(num_teachers):
        # tailles différentes entre les 2 "Teachers" factices, pour simuler
        # la diversité large/small sans télécharger quoi que ce soit
        n_embd = 64 if i == 0 else 32
        config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=2,
                             n_head=2, n_positions=64)
        model = GPT2LMHeadModel(config)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(model)
    return teachers
