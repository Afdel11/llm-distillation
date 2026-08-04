"""
tests/test_pipeline.py
========================
Test d'intégration de bout en bout, ENTIÈREMENT local (aucun accès réseau) :
Teachers factices -> ARCD -> Student factice, sur un mini dataset factice.

Objectif : valider la mécanique complète (formes de tenseurs, masquage,
backprop, les 3 régimes d'entraînement) avant de brancher les vrais modèles
Qwen2.5 sur le GPU distant (voir scripts/train.py).
"""

import torch
from torch.utils.data import DataLoader

from models.teacher import build_debug_teachers
from models.student import build_student
from datasets.dataloader import PromptResponseDataset, make_collate_fn
from trainers.baseline import train_student_alone
from trainers.hinton import train_hinton_kd
from trainers.arcd import train_arcd

VOCAB_SIZE = 128  # doit couvrir les ids générés par DebugTokenizer (mots courts)

EXAMPLES = [
    {"prompt": "Question courte simple", "response": "Réponse brève ici maintenant"},
    {"prompt": "Une question un peu plus longue avec plein de mots différents",
     "response": "Une réponse également assez longue pour tester le padding"},
    {"prompt": "Test", "response": "Ok court"},
    {"prompt": "Autre exemple de question test", "response": "Autre réponse de test complète"},
]


def _make_loader(debug_tokenizer):
    dataset = PromptResponseDataset(EXAMPLES, debug_tokenizer, max_length=32)
    collate_fn = make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id)
    return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


def _new_student():
    return build_student(VOCAB_SIZE, n_embd=32, n_layer=2, n_head=2, n_positions=64)


def test_student_alone_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    train_loader = _make_loader(debug_tokenizer)
    student = _new_student()
    train_student_alone(student, train_loader, epochs=1, device="cpu")


def test_hinton_kd_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    train_loader = _make_loader(debug_tokenizer)
    teacher = build_debug_teachers(vocab_size=VOCAB_SIZE, num_teachers=1)[0]
    student = _new_student()
    train_hinton_kd(student, teacher, train_loader, epochs=1, device="cpu")


def test_arcd_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    train_loader = _make_loader(debug_tokenizer)
    teachers = build_debug_teachers(vocab_size=VOCAB_SIZE, num_teachers=2)
    student = _new_student()
    train_arcd(student, teachers, train_loader, epochs=1, device="cpu")
