"""
arcd/confidence.py
===================
Confiance via entropie de Gini : H(P) = 1 - sum(p_i^2), c = 1 - H/H_max.

Ce module n'a PAS changé par rapport à la version vision : il réduit
toujours sur le DERNIER axe (`dim=-1`), qui représente les classes en
vision et le vocabulaire en LLM. Peu importe le nombre de dimensions
qui précèdent (batch, ou batch+séquence), le calcul reste correct —
c'est précisément ce qui permet au reste du pipeline de généraliser au
cas token-par-token sans réécriture.
"""

import torch

EPS = 1e-8


def gini_confidence(logits: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    """
    c = 1 - H_Gini(P)/H_max, calculé sur des logits bruts.

    Args:
        logits:      (..., num_classes_or_vocab) le long de `dim`.
        temperature: température de la softmax (partagée Teachers/Student).
        dim:         axe des classes/vocabulaire (toujours le dernier ici).

    Returns:
        confiance dans [0, 1], `dim` supprimée.
    """
    probs = torch.softmax(logits / temperature, dim=dim)
    num_classes = logits.shape[dim]

    h = 1.0 - torch.sum(probs ** 2, dim=dim)
    h_max = 1.0 - (1.0 / num_classes)

    # num_classes == 1 -> h_max == 0 et h == 0 (une seule classe possible,
    # aucune incertitude n'est même définissable) -> 0/0 sans cet epsilon.
    # Avec l'epsilon, confidence -> 1 (confiance triviale, cohérent : il n'y
    # a rien à hésiter quand il n'existe qu'un seul choix).
    confidence = 1.0 - (h / (h_max + EPS))
    return torch.clamp(confidence, min=0.0, max=1.0)


def teacher_confidence(teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Confiance individuelle de chaque Teacher, à chaque position.

    Args:
        teacher_logits: (batch, seq_len, num_teachers, vocab_size)
                         ou (batch, num_teachers, num_classes) en vision.

    Returns:
        même forme sans le dernier axe (vocab_size supprimé).
    """
    return gini_confidence(teacher_logits, temperature=temperature, dim=-1)


def student_confidence(student_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Confiance du Student, à chaque position.

    Args:
        student_logits: (batch, seq_len, vocab_size) ou (batch, num_classes).

    Returns:
        même forme sans le dernier axe.
    """
    return gini_confidence(student_logits, temperature=temperature, dim=-1)
