"""
trainers/arcd.py
=================
Méthode proposée : distillation multi-Teacher avec lambda(x) = C*T*(1-S)
calculé PAR TOKEN.
"""

import torch

from arcd.losses import ARCDLoss
from arcd.metrics import MetricTracker
from models.teacher import get_teacher_logits


def train_arcd(student, teachers: list, train_loader, epochs: int = 3, lr: float = 5e-4,
                temperature: float = 2.0, device: str = "cpu"):
    student.to(device)
    for t in teachers:
        t.to(device)
        t.eval()

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

            teacher_logits = get_teacher_logits(teachers, input_ids, attention_mask)
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            loss, metrics = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
            tracker.update(metrics, batch_size=input_ids.size(0))

        tracker.log(epoch=epoch + 1)
        tracker.reset()

    return student
