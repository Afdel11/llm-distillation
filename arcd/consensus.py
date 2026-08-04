"""
arcd/consensus.py
==================
Médiane pondérée robuste + score de consensus.

Convention de forme, cas LLM :
    teacher_logits : (batch, seq_len, num_teachers, vocab_size)
    -> l'axe Teachers est l'avant-dernier (dim=-2), le vocabulaire est
       le dernier (dim=-1).

En vision, la forme était (batch, num_teachers, num_classes), donc l'axe
Teachers était dim=1 (= dim=-2 aussi, puisqu'il n'y a que 2 axes après le
batch). La convention "axe Teachers = avant-dernier" est donc rétro-
compatible : ce module n'a besoin d'aucun changement de logique, seulement
d'assumer -2 au lieu de 1 en dur.
"""

import torch

from arcd.confidence import EPS, teacher_confidence


def weighted_median(values: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Médiane pondérée le long de `dim`. Fonctionne pour n'importe quel
    nombre de dimensions précédentes (batch seul, ou batch+séquence).
    """
    weights = weights.expand_as(values)

    sorted_vals, sort_idx = torch.sort(values, dim=dim)
    sorted_weights = torch.gather(weights, dim, sort_idx)

    cum_weights = torch.cumsum(sorted_weights, dim=dim)
    total_weights = cum_weights.select(dim, -1).unsqueeze(dim)
    normalized_cum = cum_weights / (total_weights + EPS)

    reached_half = (normalized_cum >= 0.5).float()
    median_idx = reached_half.argmax(dim=dim, keepdim=True)

    median = torch.gather(sorted_vals, dim, median_idx)
    return median.squeeze(dim)


def weighted_mad(values: torch.Tensor, weights: torch.Tensor, median: torch.Tensor, dim: int) -> torch.Tensor:
    """MAD pondéré autour d'une médiane déjà calculée."""
    abs_dev = torch.abs(values - median.unsqueeze(dim))
    return weighted_median(abs_dev, weights, dim=dim)


def robust_consensus(teacher_logits: torch.Tensor, temperature: float = 1.0, teacher_dim: int = -2):
    """
    Pipeline complet : confiance des Teachers -> médiane pondérée -> MAD -> C, T.

    Args:
        teacher_logits: (..., num_teachers, vocab_size)
                         ex. LLM: (batch, seq_len, num_teachers, vocab_size)
                             vision: (batch, num_teachers, num_classes)
        teacher_dim:    axe des Teachers (-2 par défaut : avant-dernier).

    Returns:
        p_median: (..., vocab_size)   — distribution cible pour L_KD
        C:        (...,)              — score de consensus, dans [0, 1]
        T:        (...,)              — confiance moyenne des Teachers
        teacher_confidences: (..., num_teachers)
    """
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    confidences = teacher_confidence(teacher_logits, temperature=temperature)  # (..., num_teachers)

    weights = confidences.unsqueeze(-1)  # (..., num_teachers, 1) -> broadcast sur le vocab

    p_median = weighted_median(teacher_probs, weights, dim=teacher_dim)         # (..., vocab_size)
    mad = weighted_mad(teacher_probs, weights, p_median, dim=teacher_dim)       # (..., vocab_size)
    mad = mad.mean(dim=-1)                                                      # (...,)

    C = 1.0 / (1.0 + mad)
    T = confidences.mean(dim=-1)

    return p_median, C, T, confidences
