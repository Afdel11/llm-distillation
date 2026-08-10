"""
trainers/baseline.py
====================

Baseline 1 : Student entraîné seul (CrossEntropy uniquement).

Cette baseline sert de borne inférieure de comparaison pour Hinton et ARCD.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arcd.losses import IGNORE_INDEX
from arcd.metrics import MetricTracker


def train_student_alone(
    student,
    train_loader,
    epochs: int = 50,
    lr: float = 5e-4,
    device: str = "cpu",
):

    student.to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
    )

    for epoch in range(epochs):

        print(f"\n========== Epoch {epoch+1}/{epochs} ==========")

        student.train()

        tracker = MetricTracker()

        for batch in tqdm(
            train_loader,
            desc="Baseline",
            leave=False,
        ):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            loss.backward()

            optimizer.step()

            tracker.update(
                {
                    "loss": loss.item(),
                },
                batch_size=input_ids.size(0),
            )

        tracker.log(epoch + 1)

        tracker.save_csv(
            "outputs/baseline_metrics.csv",
            epoch + 1,
        )

        tracker.reset()

    return student