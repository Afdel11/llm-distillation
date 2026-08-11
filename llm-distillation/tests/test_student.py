import torch

from models.student import build_student, count_parameters
from models.teacher import DebugTeacherEnsemble


def _tiny_student(vocab_size: int):
    # dimensions minuscules pour que le test tourne vite en CPU/sandbox
    return build_student(vocab_size, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1,
                          intermediate_size=64, max_position_embeddings=64)


def test_student_forward_shape():
    vocab_size = 500
    student = _tiny_student(vocab_size)
    input_ids = torch.randint(0, vocab_size, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)

    out = student(input_ids=input_ids, attention_mask=attention_mask)

    assert out.logits.shape == (2, 10, vocab_size)


def test_student_smaller_than_teachers():
    vocab_size = 500
    # Teachers factices : n_embd=64 et 32 (voir DebugTeacherEnsemble) ->
    # Student volontairement plus petit que les deux (hidden_size=16)
    student = build_student(vocab_size, hidden_size=16, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1,
                             intermediate_size=32, max_position_embeddings=64)
    ensemble = DebugTeacherEnsemble(vocab_size=vocab_size, num_teachers=2)

    n_student = count_parameters(student)
    n_teachers = list(ensemble.parameter_counts().values())

    assert all(n_student < n for n in n_teachers)


def test_student_gradient_flows():
    vocab_size = 200
    student = _tiny_student(vocab_size)
    input_ids = torch.randint(0, vocab_size, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    labels = torch.randint(0, vocab_size, (2, 6))

    out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()

    grads = [p.grad for p in student.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
