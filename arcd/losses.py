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

    def __init__(self, temperature: float = 2.0, eps: float = EPS, top_k: int = None, max_lambda: float = None,
                 anti_copy_weight: float = None):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.top_k = top_k  # voir robust_consensus() -- corrige la dilution du signal
                             # de désaccord sur un vocabulaire de grande dimension
        self.max_lambda = max_lambda  # plafonne lambda(x) -- utilisé pour un fine-tuning de
                                       # "récupération" en CE quasi pure, repartant des poids
                                       # déjà appris sous distillation, sans repartir from scratch
        self.anti_copy_weight = anti_copy_weight  # pénalise le réflexe de copie du token précédent
                                                   # quand ce n'est PAS la bonne réponse -- voir
                                                   # scripts/diagnose_generation.py, diagnostic qui
                                                   # a révélé ce comportement précis

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                input_ids: torch.Tensor = None):
        """
        Args:
            student_logits: (batch, seq_len, vocab_size)              — logits bruts du Student
            teacher_logits: (batch, seq_len, num_teachers, vocab_size) — logits bruts des Teachers
            labels:         (batch, seq_len) — indices de tokens, IGNORE_INDEX (-100) pour
                             les positions à exclure (prompt, padding).
            input_ids:      (batch, seq_len) — nécessaire uniquement si anti_copy_weight est actif ;
                             à la position i, c'est le dernier token de contexte quand le Student
                             prédit la position i+1 (donc "le token qu'il vient de voir").

        Returns:
            loss:    scalaire
            metrics: dict de diagnostics moyennés sur les tokens valides
        """
        mask = (labels != IGNORE_INDEX)  # (batch, seq_len)
        n_valid = mask.sum().clamp(min=1)

        assert teacher_logits.size(-1) == student_logits.size(-1), (
            f"vocab_size mismatch: teacher={teacher_logits.size(-1)} vs student={student_logits.size(-1)}. "
            "Le Student doit être dimensionné sur model.config.vocab_size des Teachers "
            "(voir scripts/train.py:get_model_vocab_size), pas sur len(tokenizer)."
        )

        # Le cache (data_pipeline/cache.py) stocke en float16 et le collate NE cast
        # PLUS en float32 (voir data_pipeline/dataloader.py) — le transfert CPU->GPU
        # se fait donc en float16 (moitié moins de données sur PCIe). On repasse en
        # float32 ici, après le transfert, où le cast est quasi gratuit (bande
        # passante mémoire GPU, pas PCIe).
        teacher_logits = teacher_logits.float()

        p_median, C, T, _ = robust_consensus(teacher_logits, temperature=self.temperature, top_k=self.top_k)
        S = student_confidence(student_logits, temperature=self.temperature)

        lam = adaptive_lambda(C, T, S)  # (batch, seq_len)
        if self.max_lambda is not None:
            lam = torch.clamp(lam, max=self.max_lambda)

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

        anti_copy_loss_value = 0.0
        if self.anti_copy_weight is not None and self.anti_copy_weight > 0 and input_ids is not None:
            # p_copy(i) = probabilité que le Student assigne, en position i, à répéter
            # le dernier token de contexte (input_ids[i]) comme prédiction suivante.
            student_probs = torch.softmax(student_logits, dim=-1)  # (batch, seq_len, vocab)
            p_copy = torch.gather(student_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)

            # Ne pénalise QUE quand copier serait FAUX (le vrai label n'est pas une
            # répétition du dernier token) ET sur les positions supervisées -- une
            # vraie répétition dans le langage naturel (ex: "très très") ne doit
            # jamais être punie.
            wrong_copy_mask = mask & (input_ids != safe_labels)
            n_wrong_copy = wrong_copy_mask.sum().clamp(min=1)
            anti_copy_loss = (p_copy * wrong_copy_mask.float()).sum() / n_wrong_copy

            loss = loss + self.anti_copy_weight * anti_copy_loss
            anti_copy_loss_value = anti_copy_loss.item()

        metrics = {
            "loss": loss.item(),
            "L_KD": (l_kd * mask).sum().item() / n_valid.item(),
            "L_CE": (l_ce * mask).sum().item() / n_valid.item(),
            "C": (C * mask).sum().item() / n_valid.item(),
            "T": (T * mask).sum().item() / n_valid.item(),
            "S": (S * mask).sum().item() / n_valid.item(),
            "lambda": (lam * mask).sum().item() / n_valid.item(),
            "anti_copy_loss": anti_copy_loss_value,
        }
        return loss, metrics


class FixedWeightConsensusLoss(nn.Module):
    """
    RÉGIME DE CONTRÔLE — isole l'effet de la pondération ADAPTATIVE de celui
    d'avoir simplement plusieurs Teachers.

    Réutilise exactement la même cible de consensus robuste que ARCDLoss
    (médiane pondérée par confiance, P*) et les 2 mêmes Teachers, mais avec
    un poids alpha CONSTANT au lieu de lambda(x) = C*T*(1-S) adaptatif.
    Une seule variable change par rapport à ARCDLoss : adaptatif vs fixe.
    Tout le reste (agrégation, Teachers, Student, données) est identique,
    ce qui permet d'attribuer un éventuel écart de performance à
    l'adaptivité elle-même, et non à la simple présence de 2 Teachers.

    Usage:
        criterion = FixedWeightConsensusLoss(temperature=2.0, alpha=0.5)
        loss, metrics = criterion(student_logits, teacher_logits, labels)
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, eps: float = EPS, top_k: int = None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps
        self.top_k = top_k

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        mask = (labels != IGNORE_INDEX)
        n_valid = mask.sum().clamp(min=1)

        assert teacher_logits.size(-1) == student_logits.size(-1), (
            f"vocab_size mismatch: teacher={teacher_logits.size(-1)} vs student={student_logits.size(-1)}. "
            "Le Student doit être dimensionné sur model.config.vocab_size des Teachers."
        )

        teacher_logits = teacher_logits.float()

        # Même cible de consensus qu'ARCD (médiane pondérée par confiance) —
        # C et T sont calculés et loggués à titre diagnostique, mais n'entrent
        # PAS dans le poids de la loss ici (contrairement à ARCDLoss).
        p_median, C, T, _ = robust_consensus(teacher_logits, temperature=self.temperature, top_k=self.top_k)

        l_kd = kd_loss_per_token(student_logits, p_median, temperature=self.temperature)

        safe_labels = labels.clone()
        safe_labels[~mask] = 0
        l_ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

        alpha = self.alpha  # constant, PAS de calcul de lambda(x)
        per_token_loss = alpha * l_kd + (1.0 - alpha) * l_ce
        per_token_loss = per_token_loss * mask
        loss = per_token_loss.sum() / n_valid

        metrics = {
            "loss": loss.item(),
            "L_KD": (l_kd * mask).sum().item() / n_valid.item(),
            "L_CE": (l_ce * mask).sum().item() / n_valid.item(),
            "C": (C * mask).sum().item() / n_valid.item(),   # diagnostic seulement
            "T": (T * mask).sum().item() / n_valid.item(),   # diagnostic seulement
            "alpha": alpha,
        }
        return loss, metrics
