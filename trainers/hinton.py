"""
trainers/hinton.py
===================
Baseline 2 : distillation classique de Hinton, un seul Teacher, un
lambda (alpha) constant, appliquée token par token.
"""

import torch
import torch.nn.functional as F

from arcd.losses import IGNORE_INDEX
from arcd.metrics import MetricTracker


def train_hinton_kd(student, teacher, train_loader, epochs: int = 3, lr: float = 5e-4,
                     temperature: float = 2.0, alpha: float = 0.5, device: str = "cpu"):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    for epoch in range(epochs):
        student.train()
        tracker = MetricTracker()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask = (labels != IGNORE_INDEX)
            n_valid = mask.sum().clamp(min=1)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            log_student = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            l_kd = F.kl_div(log_student, teacher_probs, reduction="none").sum(dim=-1)  # (batch, seq)
            l_kd = (l_kd * mask).sum() / n_valid * (temperature ** 2)

            safe_labels = labels.clone()
            safe_labels[~mask] = 0
            l_ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            loss = alpha * l_kd + (1 - alpha) * l_ce
            loss.backward()
            optimizer.step()
            tracker.update({"loss": loss.item()}, batch_size=input_ids.size(0))

        print(f"  [hinton_kd] epoch {epoch+1}/{epochs} — loss={tracker.average()['loss']:.4f}")

    return student
