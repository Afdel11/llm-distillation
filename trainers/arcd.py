"""
trainers/arcd.py
================

Méthode proposée :
Adaptive Robust Confidence Distillation (ARCD)

Lambda adaptatif :

    lambda(x) = C * T * (1 - S)

Deux modes possibles :

1.
teacher_ensemble fourni
→ les Teachers sont calculés à chaque batch

2.
teacher_logits déjà présents dans le batch
→ cache construit avec build_teacher_cache.py
→ beaucoup plus rapide
"""

import torch
from tqdm import tqdm

from arcd.losses import ARCDLoss
from arcd.metrics import MetricTracker


def train_arcd(
    student,
    train_loader,
    epochs: int = 3,
    lr: float = 5e-4,
    temperature: float = 2.0,
    device: str = "cpu",
    teacher_ensemble=None,
):

    student.to(device)

    if teacher_ensemble is not None:
        teacher_ensemble.to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
    )

    criterion = ARCDLoss(
        temperature=temperature,
    )

    for epoch in range(epochs):

        print(f"\n========== Epoch {epoch+1}/{epochs} ==========")

        student.train()

        tracker = MetricTracker()

        for batch in tqdm(
            train_loader,
            desc="ARCD",
            leave=False,
        ):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # --------------------------------------------------
            # Teacher logits
            # --------------------------------------------------

            if teacher_ensemble is not None:

                teacher_logits = teacher_ensemble(
                    input_ids,
                    attention_mask,
                )

            else:

                if "teacher_logits" not in batch:

                    raise RuntimeError(
                        "teacher_logits absents du batch.\n"
                        "Utiliser build_teacher_cache.py "
                        "ou fournir teacher_ensemble."
                    )

                teacher_logits = batch["teacher_logits"].to(device)

            # --------------------------------------------------
            # Student
            # --------------------------------------------------

            student_logits = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            # --------------------------------------------------
            # ARCD Loss
            # --------------------------------------------------

            loss, metrics = criterion(
                student_logits,
                teacher_logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            tracker.update(
                metrics,
                batch_size=input_ids.size(0),
            )

        # ======================================================
        # Fin d'époque
        # ======================================================

        tracker.log(epoch + 1)

        tracker.save_csv(
            "outputs/arcd_metrics.csv",
            epoch + 1,
        )

        tracker.reset()

    return student