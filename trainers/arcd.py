"""
trainers/arcd.py
=================
Méthode proposée : distillation multi-Teacher avec lambda(x) = C*T*(1-S)
calculé PAR TOKEN.

Deux modes, mêmes résultats, coûts très différents :
  - teacher_ensemble fourni : les Teachers tournent à CHAQUE batch (plus
    simple, mais recalcule un forward pass déjà gelé à chaque epoch).
  - batch["teacher_logits"] déjà présent (via make_cached_collate_fn) :
    aucun forward Teacher pendant l'entraînement, juste une lecture disque.
    Recommandé dès que le dataset dépasse quelques centaines d'exemples
    (voir scripts/build_teacher_cache.py).
"""

import torch

from arcd.losses import ARCDLoss
from arcd.metrics import MetricTracker


def train_arcd(student, train_loader, epochs: int = 3, lr: float = 5e-4,
                temperature: float = 2.0, device: str = "cpu", teacher_ensemble=None):
    """
    Args:
        teacher_ensemble: TeacherEnsemble (ou DebugTeacherEnsemble) pour calculer
                          les logits en direct. Si None, chaque batch DOIT déjà
                          contenir "teacher_logits" (loader construit avec
                          make_cached_collate_fn).
    """
    student.to(device)
    if teacher_ensemble is not None:
        teacher_ensemble.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    criterion = ARCDLoss(temperature=temperature)

    for epoch in range(epochs):
        student.train()
        tracker = MetricTracker()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if teacher_ensemble is not None:
                teacher_logits = teacher_ensemble(input_ids, attention_mask)
            else:
                assert "teacher_logits" in batch, (
                    "Pas de teacher_ensemble fourni et pas de logits en cache dans le batch : "
                    "utilise make_cached_collate_fn (voir datasets/cache.py) ou passe teacher_ensemble."
                )
                teacher_logits = batch["teacher_logits"].to(device)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            loss, metrics = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
            tracker.update(metrics, batch_size=input_ids.size(0))

        tracker.log(epoch=epoch + 1)
        tracker.reset()

    return student
