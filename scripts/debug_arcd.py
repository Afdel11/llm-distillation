import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from models.teacher import (
    build_teachers,
    get_teacher_logits,
)

from arcd.confidence import teacher_confidence
from arcd.consensus import robust_teacher_consensus


PROMPT = """
Explain knowledge distillation in one sentence.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    print("=" * 70)
    print("ARCD DIAGNOSTIC")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    teachers = build_teachers()

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    print("\nRunning Teachers...\n")

    logits = get_teacher_logits(
        teachers,
        input_ids,
        attention_mask
    )

    print("Teacher logits shape")
    print(logits.shape)

    probs = F.softmax(logits, dim=-1)

    print("\nTeacher probabilities shape")
    print(probs.shape)

    confidence = teacher_confidence(probs)

    print("\nTeacher confidence")

    print(confidence)

    print()

    T = confidence.mean(dim=-1)

    print("Average Teacher confidence T")

    print(T)

    print()

    teacher_distribution, C = robust_teacher_consensus(
        probs,
        confidence
    )

    print("Consensus C")

    print(C)

    print()

    print("Teacher distribution shape")

    print(teacher_distribution.shape)

    print()

    print("=" * 70)

    print("Summary")

    print("=" * 70)

    print(f"Teacher logits      : {tuple(logits.shape)}")

    print(f"Teacher confidence  : {tuple(confidence.shape)}")

    print(f"T                   : {tuple(T.shape)}")

    print(f"C                   : {tuple(C.shape)}")

    print(f"Consensus output    : {tuple(teacher_distribution.shape)}")


if __name__ == "__main__":
    main()