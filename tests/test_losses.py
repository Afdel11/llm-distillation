import torch

from arcd.losses import ARCDLoss, IGNORE_INDEX, adaptive_lambda


def test_adaptive_lambda_bounded():
    C = torch.tensor([[0.9, 0.1, 0.9]])
    T = torch.tensor([[0.8, 0.8, 0.05]])
    S = torch.tensor([[0.1, 0.1, 0.1]])
    lam = adaptive_lambda(C, T, S)
    assert torch.all((lam >= 0) & (lam <= 1))


def test_arcd_loss_end_to_end_with_masking():
    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 8, 2, 300
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))
    labels[:, :3] = IGNORE_INDEX  # simule un prompt qu'on ne fait pas prédire

    criterion = ARCDLoss(temperature=2.0)
    loss, metrics = criterion(student_logits, teacher_logits, labels)
    loss.backward()

    assert student_logits.grad is not None
    assert not torch.isnan(loss)
    assert not torch.isnan(student_logits.grad).any()


def test_arcd_loss_all_positions_masked_returns_zero():
    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 8, 2, 300
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    labels_all_masked = torch.full((batch, seq_len), IGNORE_INDEX)

    criterion = ARCDLoss(temperature=2.0)
    loss, _ = criterion(student_logits, teacher_logits, labels_all_masked)

    assert loss.item() == 0.0
