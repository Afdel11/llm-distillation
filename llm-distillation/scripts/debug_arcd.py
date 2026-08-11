"""
scripts/debug_arcd.py
=====================

Diagnostic complet du pipeline ARCD.

Ce script ne réalise AUCUN entraînement.

Il vérifie uniquement :

1. Chargement des Teachers
2. Tokenisation
3. Forward des Teachers
4. Calcul des confiances
5. Consensus robuste
6. Dimensions des tenseurs
"""

import torch

from transformers import AutoTokenizer

from models.teacher import TeacherEnsemble

from arcd.consensus import robust_consensus


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = """
Explain knowledge distillation in one sentence.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    separator("ARCD DIAGNOSTIC")

    print(f"Device : {DEVICE}")

    separator("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    separator("Loading Teacher Ensemble")

    teachers = TeacherEnsemble(device=DEVICE)

    print("Teachers loaded\n")

    for name in teachers.teacher_names:
        print(" •", name)

    separator("Parameter Count")

    for name, n_params in teachers.parameter_counts().items():
        print(f"{name:<10} : {n_params:,}")

    separator("Tokenization")

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    print("Input shape")

    print(tuple(input_ids.shape))

    separator("Teacher Forward")

    with torch.no_grad():

        teacher_logits = teachers(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    print("Teacher logits shape")

    print(tuple(teacher_logits.shape))

    separator("ARCD Consensus")

    teacher_distribution, C, T, confidences = robust_consensus(
        teacher_logits
    )

    separator("Teacher Confidence")

    print(confidences)

    separator("Average Teacher Confidence (T)")

    print(T)

    separator("Consensus (C)")

    print(C)

    separator("Teacher Distribution")

    print(tuple(teacher_distribution.shape))

    separator("SUMMARY")

    print(f"Teacher logits          : {tuple(teacher_logits.shape)}")
    print(f"Teacher confidences     : {tuple(confidences.shape)}")
    print(f"Average confidence (T)  : {tuple(T.shape)}")
    print(f"Consensus (C)           : {tuple(C.shape)}")
    print(f"Teacher distribution    : {tuple(teacher_distribution.shape)}")

    separator("ARCD diagnostic finished")


if __name__ == "__main__":
    main()