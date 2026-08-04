import torch

from models.student import build_student, count_parameters
from models.teacher import build_debug_teachers, count_parameters as count_teacher_params


def test_student_forward_shape():
    vocab_size = 500
    student = build_student(vocab_size, n_embd=32, n_layer=2, n_head=2, n_positions=64)
    input_ids = torch.randint(0, vocab_size, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)

    out = student(input_ids=input_ids, attention_mask=attention_mask)

    assert out.logits.shape == (2, 10, vocab_size)


def test_student_smaller_than_teachers():
    # n_embd=16, strictement inférieur au plus petit Teacher factice (n_embd=32,
    # voir build_debug_teachers) : sans cette marge, le test dépend d'un hasard
    # de configuration plutôt que de garantir réellement l'invariante voulue.
    vocab_size = 500
    student = build_student(vocab_size, n_embd=16, n_layer=2, n_head=2, n_positions=64)
    teachers = build_debug_teachers(vocab_size=vocab_size, num_teachers=2)

    n_student = count_parameters(student)
    n_teachers = [count_teacher_params(t) for t in teachers]

    assert all(n_student < n for n in n_teachers)
