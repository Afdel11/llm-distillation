import torch

from models.teacher import DebugTeacherEnsemble


def test_debug_teachers_have_different_sizes():
    ensemble = DebugTeacherEnsemble(vocab_size=500, num_teachers=2)
    counts = ensemble.parameter_counts()
    assert len(set(counts.values())) == 2, "Les deux Teachers factices doivent avoir des tailles différentes"


def test_teacher_names_property():
    ensemble = DebugTeacherEnsemble(vocab_size=500, num_teachers=3)
    assert len(ensemble.teacher_names) == 3


def test_call_returns_expected_shape():
    vocab_size = 500
    ensemble = DebugTeacherEnsemble(vocab_size=vocab_size, num_teachers=2)
    batch, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    logits = ensemble(input_ids, attention_mask)

    assert logits.shape == (batch, seq_len, 2, vocab_size)
