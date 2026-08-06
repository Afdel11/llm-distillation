"""
trainers/hinton.py
==================

Baseline 2 : Distillation classique de Hinton.

- Un seul Teacher
- alpha constant
- température constante
- CrossEntropy + KL Divergence
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arcd.losses import IGNORE_INDEX
from arcd.metrics import MetricTracker


def train_hinton_kd(
    student,
    teacher,
    train_loader,
    epochs: int = 3,
    lr: float = 5e-4,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str = "cpu",
):

    student.to(device)

    teacher.to(device)
    teacher.eval()

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
            desc="Hinton KD",
            leave=False,
        ):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            mask = (labels != IGNORE_INDEX)
            n_valid = mask.sum().clamp(min=1)

            optimizer.zero_grad()

            with torch.no_grad():

                teacher_logits = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            # -------------------------------------------------
            # Alignement Teacher / Student
            # Certains modèles Qwen possèdent plus d'embeddings
            # que de tokens réellement exposés par le tokenizer.
            # On projette donc le Teacher sur le vocabulaire du Student.
            # -------------------------------------------------

            student_vocab = student_logits.size(-1)
            teacher_vocab = teacher_logits.size(-1)

            if teacher_vocab != student_vocab:
                teacher_logits = teacher_logits[..., :student_vocab]

            # -------------------------------------------------
            # Knowledge Distillation Loss
            # -------------------------------------------------

            log_student = F.log_softmax(
                student_logits / temperature,
                dim=-1,
            )

            teacher_probs = F.softmax(
                teacher_logits / temperature,
                dim=-1,
            )

            l_kd = F.kl_div(
                log_student,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)

            l_kd = (
                (l_kd * mask).sum()
                / n_valid
                * (temperature ** 2)
            )

            # -------------------------------------------------
            # Cross Entropy
            # -------------------------------------------------

            l_ce = F.cross_entropy(
                student_logits.reshape(
                    -1,
                    student_logits.size(-1),
                ),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            # -------------------------------------------------
            # Loss finale
            # -------------------------------------------------

            loss = alpha * l_kd + (1 - alpha) * l_ce

            loss.backward()

            optimizer.step()

            tracker.update(
                {
                    "loss": loss.item(),
                    "L_KD": l_kd.item(),
                    "L_CE": l_ce.item(),
                },
                batch_size=input_ids.size(0),
            )

        tracker.log(epoch + 1)

        tracker.save_csv(
            "outputs/hinton_metrics.csv",
            epoch + 1,
        )

        tracker.reset()

    return student