import torch

from arcd.confidence import gini_confidence, teacher_confidence, student_confidence


def test_deterministic_distribution():
    logits = torch.tensor([[100.0, -100.0, -100.0]])
    conf = gini_confidence(logits)
    assert torch.isclose(conf, torch.tensor([1.0]), atol=1e-4).all()


def test_uniform_distribution():
    logits = torch.zeros((1, 5))
    conf = gini_confidence(logits)
    assert torch.isclose(conf, torch.tensor([0.0]), atol=1e-4).all()


def test_single_class_no_nan():
    logits = torch.tensor([[5.0]])
    conf = gini_confidence(logits)
    assert not torch.isnan(conf).any()


def test_temperature_effect():
    logits = torch.tensor([[5.0, 2.0, 1.0]])
    c_low_temperature = gini_confidence(logits, temperature=1.0)
    c_high_temperature = gini_confidence(logits, temperature=5.0)
    assert c_high_temperature < c_low_temperature


def test_teacher_confidence_shape_vision():
    logits = torch.randn(4, 3, 10)  # (batch, num_teachers, num_classes)
    conf = teacher_confidence(logits)
    assert conf.shape == (4, 3)


def test_teacher_confidence_shape_llm():
    logits = torch.randn(2, 5, 3, 1000)  # (batch, seq_len, num_teachers, vocab_size)
    conf = teacher_confidence(logits)
    assert conf.shape == (2, 5, 3)


def test_student_confidence_shape_llm():
    logits = torch.randn(2, 5, 1000)  # (batch, seq_len, vocab_size)
    conf = student_confidence(logits)
    assert conf.shape == (2, 5)
