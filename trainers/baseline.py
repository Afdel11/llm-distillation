"""
trainers/baseline.py
=====================
Baseline 1 : Student entraîné seul, cross-entropy standard par token
(aucune distillation). Sert de borne inférieure de comparaison.
"""

import torch
import torch.nn.functional as F

from arcd.losses import IGNORE_INDEX
from arcd.metrics import MetricTracker


def train_student_alone(student, train_loader, epochs: int = 3, lr: float = 5e-4, device: str = "cpu"):
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    for epoch in range(epochs):
        student.train()
        tracker = MetricTracker()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            optimizer.step()
            tracker.update({"loss": loss.item()}, batch_size=input_ids.size(0))

        print(f"  [student_alone] epoch {epoch+1}/{epochs} — loss={tracker.average()['loss']:.4f}")

    return student
