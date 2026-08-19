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

Quatre régimes :
  - "student_alone"      : transformers.Trainer standard, sans sous-classe.
    Le batch contient déjà "labels" -> le modèle calcule sa propre
    cross-entropy nativement, rien à personnaliser.
  - "hinton_kd"           : HintonTrainer (1 Teacher, lambda fixe)
  - "multi_teacher_fixed" : FixedWeightTrainer (2 Teachers, MÊME consensus
    robuste qu'ARCD, mais lambda CONSTANT) — régime de contrôle, isole
    l'effet de l'adaptivité de celui d'avoir simplement 2 Teachers.
  - "arcd"                : ARCDTrainer (N Teachers, lambda par token adaptatif)
"""

import torch
import torch.nn.functional as F
from transformers import Trainer

from arcd.losses import ARCDLoss, FixedWeightConsensusLoss, IGNORE_INDEX


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
        self._eval_metrics_buffer = []

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
        metrics = {"L_KD": l_kd.item(), "L_CE": l_ce.item()}

        if model.training:
            # En entraînement : logging brut immédiat, pour le suivi live.
            self.log({f"hinton/{k}": v for k, v in metrics.items()})
        else:
            # En évaluation : PAS de log par batch (spam inutile dans
            # log_history) -> accumulé puis moyenné dans evaluation_loop,
            # fusionné avec eval_loss dans la MÊME entrée.
            self._eval_metrics_buffer.append(metrics)

        return (loss, student_outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                         ignore_keys=None, metric_key_prefix="eval"):
        self._eval_metrics_buffer = []
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )
        if self._eval_metrics_buffer:
            keys = self._eval_metrics_buffer[0].keys()
            for k in keys:
                avg = sum(d[k] for d in self._eval_metrics_buffer) / len(self._eval_metrics_buffer)
                output.metrics[f"{metric_key_prefix}_hinton/{k}"] = avg
        return output


class ARCDTrainer(Trainer):
    """Méthode proposée : lambda(x) = C * T * (1 - S), calculé par token."""

    def __init__(self, *args, teacher_ensemble=None, temperature: float = 2.0,
                 eval_data_collator=None, top_k: int = None, max_lambda: float = None,
                 anti_copy_weight: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_ensemble = teacher_ensemble  # None si on utilise uniquement le cache
        self.arcd_loss = ARCDLoss(temperature=temperature, top_k=top_k, max_lambda=max_lambda,
                                   anti_copy_weight=anti_copy_weight)
        # transformers.Trainer ne supporte qu'un seul data_collator (train ET eval).
        # Ici, train utilise le cache d'entraînement et eval le cache de validation
        # (deux fichiers distincts, indices non interchangeables) -> on a besoin
        # d'un collator différent pour l'évaluation. On surcharge get_eval_dataloader
        # pour substituer temporairement self.data_collator plutôt que de bidouiller
        # l'API du Trainer parent.
        self._eval_data_collator = eval_data_collator
        self._eval_metrics_buffer = []

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        self.data_collator = self._eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

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

        loss, metrics = self.arcd_loss(student_logits, teacher_logits, labels, input_ids=inputs["input_ids"])

        if model.training:
            # En entraînement : logging brut immédiat, pour le suivi live.
            self.log({f"arcd/{k}": v for k, v in metrics.items()})
        else:
            # En évaluation : accumulé, moyenné et fusionné avec eval_loss
            # dans evaluation_loop -> pas de spam, une seule entrée propre
            # par epoch dans log_history / trainer_state.json.
            self._eval_metrics_buffer.append(metrics)

        return (loss, student_outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                         ignore_keys=None, metric_key_prefix="eval"):
        self._eval_metrics_buffer = []
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )
        if self._eval_metrics_buffer:
            keys = self._eval_metrics_buffer[0].keys()
            for k in keys:
                avg = sum(d[k] for d in self._eval_metrics_buffer) / len(self._eval_metrics_buffer)
                output.metrics[f"{metric_key_prefix}_arcd/{k}"] = avg
        return output


class FixedWeightTrainer(Trainer):
    """
    RÉGIME DE CONTRÔLE (multi_teacher_fixed) — mêmes 2 Teachers et même
    cible de consensus robuste qu'ARCDTrainer, mais poids de mélange
    ALPHA CONSTANT au lieu de lambda(x) adaptatif. Sert à isoler l'effet
    de la pondération adaptative de celui d'avoir simplement 2 Teachers
    plutôt qu'1 (voir arcd/losses.py:FixedWeightConsensusLoss).

    Structure volontairement identique à ARCDTrainer (même gestion du
    cache, du collate d'évaluation séparé, du logging) pour que la seule
    différence entre les deux classes soit la loss utilisée.
    """

    def __init__(self, *args, teacher_ensemble=None, temperature: float = 2.0,
                 alpha: float = 0.5, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_ensemble = teacher_ensemble
        self.fixed_loss = FixedWeightConsensusLoss(temperature=temperature, alpha=alpha)
        self._eval_data_collator = eval_data_collator
        self._eval_metrics_buffer = []

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        self.data_collator = self._eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

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
                "Lance scripts/build_teacher_cache.py, ou passe teacher_ensemble à FixedWeightTrainer."
            )
            self.teacher_ensemble.to(student_logits.device)
            teacher_logits = self.teacher_ensemble(inputs["input_ids"], inputs["attention_mask"])

        loss, metrics = self.fixed_loss(student_logits, teacher_logits, labels)

        if model.training:
            self.log({f"fixedmt/{k}": v for k, v in metrics.items()})
        else:
            self._eval_metrics_buffer.append(metrics)

        return (loss, student_outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                         ignore_keys=None, metric_key_prefix="eval"):
        self._eval_metrics_buffer = []
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )
        if self._eval_metrics_buffer:
            keys = self._eval_metrics_buffer[0].keys()
            for k in keys:
                avg = sum(d[k] for d in self._eval_metrics_buffer) / len(self._eval_metrics_buffer)
                output.metrics[f"{metric_key_prefix}_fixedmt/{k}"] = avg
        return output
