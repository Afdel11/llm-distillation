import torch

from arcd.losses import ARCDLoss, FixedWeightConsensusLoss, IGNORE_INDEX, adaptive_lambda


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


def test_arcd_loss_top_k_reaches_robust_consensus():
    """
    Câblage de bout en bout : ARCDLoss(top_k=...) doit réellement changer le
    C loggué, pas juste le paramètre top_k de robust_consensus() pris isolément.
    Sans ce test, un futur refactor pourrait recasser le passage de top_k à
    travers ARCDLoss.forward sans qu'aucun test ne s'en aperçoive.
    """
    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 8, 2, 2000
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    criterion_full = ARCDLoss(temperature=2.0, top_k=None)
    _, metrics_full = criterion_full(student_logits, teacher_logits, labels)

    criterion_topk = ARCDLoss(temperature=2.0, top_k=10)
    _, metrics_topk = criterion_topk(student_logits, teacher_logits, labels)

    assert metrics_topk["C"] < metrics_full["C"], (
        "top_k=10 devrait révéler plus de désaccord (C plus bas) que top_k=None "
        "sur ce vocabulaire jouet — sinon top_k n'atteint plus robust_consensus."
    )


def test_fixedweight_loss_top_k_reaches_robust_consensus():
    """Même câblage que ARCDLoss, pour FixedWeightConsensusLoss (voir ci-dessus)."""
    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 8, 2, 2000
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    criterion_full = FixedWeightConsensusLoss(temperature=2.0, alpha=0.5, top_k=None)
    _, metrics_full = criterion_full(student_logits, teacher_logits, labels)

    criterion_topk = FixedWeightConsensusLoss(temperature=2.0, alpha=0.5, top_k=10)
    _, metrics_topk = criterion_topk(student_logits, teacher_logits, labels)

    assert metrics_topk["C"] < metrics_full["C"]
    # alpha reste inchangé par top_k : seule la loss (diagnostic C) est affectée.
    assert metrics_topk["alpha"] == metrics_full["alpha"] == 0.5


def test_arcd_loss_all_positions_masked_returns_zero():
    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 8, 2, 300
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    labels_all_masked = torch.full((batch, seq_len), IGNORE_INDEX)

    criterion = ARCDLoss(temperature=2.0)
    loss, _ = criterion(student_logits, teacher_logits, labels_all_masked)

    assert loss.item() == 0.0
