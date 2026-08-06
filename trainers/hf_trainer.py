"""
trainers/hf_trainer.py
========================
Remplace les anciennes boucles d'entraînement maison (trainers/baseline.py,
hinton.py, arcd.py, supprimés) par des sous-classes de transformers.Trainer.

Ce qu'on obtient gratuitement par rapport à l'ancienne approche :
  - scheduler de learning rate (warmup, cosine, etc.)
  - accumulation de gradient
  - reprise depuis un checkpoint interrompu (resume_from_checkpoint=True)
  - précision mixte configurable en un paramètre (bf16=True)
  - logging vers TensorBoard/W&B sans code supplémentaire
  - compatibilité multi-GPU/DeepSpeed si un jour nécessaire

Ce qui NE change PAS : le cœur ARCD (arcd/confidence.py, consensus.py,
losses.py, metrics.py) est utilisé tel quel — seule la boucle qui l'appelle
change d'implémentation.

Trois régimes :
  - "student_alone" : transformers.Trainer standard, sans sous-classe.
    Le batch contient déjà "labels" -> le modèle calcule sa propre
    cross-entropy nativement, rien à personnaliser.
  - "hinton_kd"      : HintonTrainer (1 Teacher, lambda fixe)
  - "arcd"           : ARCDTrainer (N Teachers, lambda par token)
"""

import torch
import torch.nn.functional as F
from transformers import Trainer

from arcd.losses import ARCDLoss, IGNORE_INDEX


def drop_keys_collate(collate_fn, keys: tuple):
    """
    Enveloppe un collate_fn pour retirer certaines clés avant que le batch
    n'atteigne le modèle (ex: "idx", utile pour retrouver le cache de logits
    Teachers mais que le modèle ne sait pas interpréter).
    """
    def wrapped(batch):
        out = collate_fn(batch)
        return {k: v for k, v in out.items() if k not in keys}
    return wrapped


class HintonTrainer(Trainer):
    """Distillation classique, un seul Teacher, lambda (alpha) constant."""

    def __init__(self, *args, teacher, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        teacher_logits_cached = inputs.pop("teacher_logits", None)

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        if teacher_logits_cached is not None:
            teacher_logits = teacher_logits_cached.to(student_logits.dtype)
        else:
            self.teacher.to(student_logits.device)
            with torch.no_grad():
                teacher_logits = self.teacher(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                ).logits

        mask = (labels != IGNORE_INDEX)
        n_valid = mask.sum().clamp(min=1)

        log_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        l_kd = F.kl_div(log_student, teacher_probs, reduction="none").sum(dim=-1)
        l_kd = (l_kd * mask).sum() / n_valid * (self.temperature ** 2)

        safe_labels = labels.clone()
        safe_labels[~mask] = 0
        l_ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        loss = self.alpha * l_kd + (1 - self.alpha) * l_ce
        self.log({"hinton/L_KD": l_kd.item(), "hinton/L_CE": l_ce.item()})

        return (loss, student_outputs) if return_outputs else loss


class ARCDTrainer(Trainer):
    """Méthode proposée : lambda(x) = C * T * (1 - S), calculé par token."""

    def __init__(self, *args, teacher_ensemble=None, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_ensemble = teacher_ensemble  # None si on utilise uniquement le cache
        self.arcd_loss = ARCDLoss(temperature=temperature)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        teacher_logits_cached = inputs.pop("teacher_logits", None)

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        if teacher_logits_cached is not None:
            teacher_logits = teacher_logits_cached.to(student_logits.device)
        else:
            assert self.teacher_ensemble is not None, (
                "Ni cache de logits Teachers dans le batch, ni teacher_ensemble fourni. "
                "Lance scripts/build_teacher_cache.py, ou passe teacher_ensemble à ARCDTrainer."
            )
            self.teacher_ensemble.to(student_logits.device)
            teacher_logits = self.teacher_ensemble(inputs["input_ids"], inputs["attention_mask"])

        loss, metrics = self.arcd_loss(student_logits, teacher_logits, labels)
        self.log({f"arcd/{k}": v for k, v in metrics.items()})

        return (loss, student_outputs) if return_outputs else loss
