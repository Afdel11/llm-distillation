"""
tests/test_pipeline.py
========================
Test d'intégration de bout en bout, ENTIÈREMENT local (aucun accès réseau) :
Teachers factices -> ARCD -> Student factice, sur un mini dataset factice.
Couvre aussi le chemin "cache de logits" (build_teacher_cache + collate caché).
"""

import torch
from torch.utils.data import DataLoader

from models.teacher import DebugTeacherEnsemble
from models.student import build_student
from datasets.dataloader import PromptResponseDataset, make_collate_fn, make_cached_collate_fn
from datasets.cache import build_teacher_cache
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


def _dataset(debug_tokenizer):
    return PromptResponseDataset(EXAMPLES, debug_tokenizer, max_length=32)


def _new_student():
    return build_student(VOCAB_SIZE, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1,
                          intermediate_size=64, max_position_embeddings=64)


def test_student_alone_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    student = _new_student()
    train_student_alone(student, train_loader, epochs=1, device="cpu")


def test_hinton_kd_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=1)
    teacher = ensemble.models[ensemble.teacher_names[0]]
    student = _new_student()
    train_hinton_kd(student, teacher, train_loader, epochs=1, device="cpu")


def test_arcd_live_teachers_runs_without_error(debug_tokenizer):
    """Mode "direct" : les Teachers tournent à chaque batch."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    student = _new_student()
    train_arcd(student, train_loader, epochs=1, device="cpu", teacher_ensemble=ensemble)


def test_arcd_cached_teachers_runs_without_error(debug_tokenizer):
    """Mode "cache" : les logits Teachers sont pré-calculés une fois, aucun
    forward Teacher pendant l'entraînement — doit produire un pipeline
    tout aussi fonctionnel que le mode direct."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    cache = build_teacher_cache(ensemble, dataset, pad_token_id=debug_tokenizer.pad_token_id,
                                 device="cpu", batch_size=2)
    assert len(cache) == len(dataset)

    collate_fn = make_cached_collate_fn(pad_token_id=debug_tokenizer.pad_token_id,
                                         teacher_logits_cache=cache)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    student = _new_student()
    train_arcd(student, train_loader, epochs=1, device="cpu")  # pas de teacher_ensemble : lit le cache
