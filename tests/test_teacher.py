import torch

from models.teacher import build_debug_teachers, get_teacher_logits, count_parameters


def test_debug_teachers_have_different_sizes():
    teachers = build_debug_teachers(vocab_size=500, num_teachers=2)
    assert count_parameters(teachers[0]) != count_parameters(teachers[1])


def test_get_teacher_logits_shape():
    vocab_size = 500
    teachers = build_debug_teachers(vocab_size=vocab_size, num_teachers=2)
    batch, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    logits = get_teacher_logits(teachers, input_ids, attention_mask)

    assert logits.shape == (batch, seq_len, 2, vocab_size)
