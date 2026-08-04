import torch

from arcd.consensus import robust_consensus


def test_vision_backward_compatibility():
    """(batch, num_teachers, num_classes) — la médiane pondérée doit suivre
    la majorité confiante, pas être trompée par un Teacher isolé."""
    teacher_logits = torch.tensor([[
        [8.0, -8.0, -8.0],
        [7.0, -7.0, -7.0],
        [-8.0, 8.0, -8.0],
    ]])
    p_median, C, T, confs = robust_consensus(teacher_logits)
    assert p_median.shape == (1, 3)
    assert p_median[0].argmax().item() == 0


def test_llm_output_shapes():
    batch, seq_len, num_teachers, vocab_size = 2, 6, 2, 500
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    p_median, C, T, confs = robust_consensus(teacher_logits)
    assert p_median.shape == (batch, seq_len, vocab_size)
    assert C.shape == (batch, seq_len)
    assert T.shape == (batch, seq_len)


def test_median_is_a_valid_probability_distribution():
    batch, seq_len, num_teachers, vocab_size = 2, 6, 2, 500
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    p_median, _, _, _ = robust_consensus(teacher_logits)
    assert torch.allclose(p_median.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-4)


def test_high_consensus_but_low_confidence_is_caught_by_T():
    """3 Teachers d'accord... sur du bruit (logits identiques -> uniforme).
    C doit être élevé (ils sont d'accord) mais T doit être proche de 0
    (ils ne savent rien) — c'est précisément le rôle du facteur T."""
    flat_logits = torch.zeros(1, 1, 3, 50)
    _, C, T, _ = robust_consensus(flat_logits)
    assert T.item() < 0.01
