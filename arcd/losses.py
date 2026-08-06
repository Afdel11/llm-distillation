"""
arcd/losses.py
===============
lambda(x) = C * T * (1 - S), calculé PAR TOKEN (pas par phrase).

Convention labels façon Hugging Face : les positions à ignorer (tokens du
prompt qu'on ne veut pas faire prédire, ou padding) portent le label -100.
C'est la convention standard de `transformers` pour les modèles causaux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcd.confidence import EPS, student_confidence
from arcd.consensus import robust_consensus

IGNORE_INDEX = -100


def adaptive_lambda(C: torch.Tensor, T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """lambda(x) = C * T * (1 - S), déjà dans [0, 1]. Même forme que C/T/S (ex: (batch, seq_len))."""
    return C * T * (1.0 - S)


def kd_loss_per_token(student_logits: torch.Tensor, target_probs: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
    """
    KL(Student || target) par position, sans réduction.

    Args:
        student_logits: (batch, seq_len, vocab_size)
        target_probs:   (batch, seq_len, vocab_size) — déjà une distribution
                         de probabilité (ex: p_median), PAS des logits.

    Returns:
        (batch, seq_len)
    """
    log_student = torch.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(log_student, target_probs, reduction="none").sum(dim=-1)


class ARCDLoss(nn.Module):
    """
    Loss ARCD token par token pour la distillation de LLM.

    Usage:
        criterion = ARCDLoss(temperature=2.0)
        loss, metrics = criterion(student_logits, teacher_logits, labels)
        loss.backward()
    """

    def __init__(self, temperature: float = 2.0, eps: float = EPS):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            student_logits: (batch, seq_len, vocab_size)              — logits bruts du Student
            teacher_logits: (batch, seq_len, num_teachers, vocab_size) — logits bruts des Teachers
            labels:         (batch, seq_len) — indices de tokens, IGNORE_INDEX (-100) pour
                             les positions à exclure (prompt, padding).

        Returns:
            loss:    scalaire
            metrics: dict de diagnostics moyennés sur les tokens valides
        """
        mask = (labels != IGNORE_INDEX)  # (batch, seq_len)
        n_valid = mask.sum().clamp(min=1)
       
        # Alignement automatique du vocabulaire 
        student_vocab = student_logits.size(-1)
        teacher_vocab = teacher_logits.size(-1)

        if teacher_vocab != student_vocab:
            teacher_logits = teacher_logits[..., :student_vocab]
            
        assert teacher_logits.size(-1) == student_logits.size(-1)

        p_median, C, T, _ = robust_consensus(teacher_logits, temperature=self.temperature)
        S = student_confidence(student_logits, temperature=self.temperature)

        lam = adaptive_lambda(C, T, S)  # (batch, seq_len)

        l_kd = kd_loss_per_token(student_logits, p_median, temperature=self.temperature)  # (batch, seq_len)

        # cross_entropy attend (N, vocab) et (N,) ; on aplatit batch*seq
        safe_labels = labels.clone()
        safe_labels[~mask] = 0  # évite un crash sur les indices -100 ; masqué juste après de toute façon
        l_ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view_as(labels)  # (batch, seq_len)

        per_token_loss = lam * l_kd + (1.0 - lam) * l_ce
        per_token_loss = per_token_loss * mask  # annule les positions ignorées
        loss = per_token_loss.sum() / n_valid

        metrics = {
            "loss": loss.item(),
            "L_KD": (l_kd * mask).sum().item() / n_valid.item(),
            "L_CE": (l_ce * mask).sum().item() / n_valid.item(),
            "C": (C * mask).sum().item() / n_valid.item(),
            "T": (T * mask).sum().item() / n_valid.item(),
            "S": (S * mask).sum().item() / n_valid.item(),
            "lambda": (lam * mask).sum().item() / n_valid.item(),
        }
        return loss, metrics
