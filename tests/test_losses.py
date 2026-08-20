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


def test_arcd_loss_applies_causal_shift_not_copy_target():
    """
    RÉGRESSION CRITIQUE -- bug découvert lors de la rédaction du mémoire
    (chapitre 5, diagnostic de génération) : data_pipeline/prompts.py
    construit labels[i] == input_ids[i] (même position, pas décalé), en
    présupposant que le calcul de loss appliquera lui-même le décalage
    causal standard (comme le fait Qwen2ForCausalLM.forward() en interne
    pour le régime student_alone). Sans ce décalage, le Student est
    entraîné à prédire "le token qu'il vient de recevoir en contexte" au
    lieu de "le token qui vient après" -- un objectif de copie pure, qui a
    provoqué un effondrement systématique en génération autonome sur tous
    les régimes de distillation (Hinton, ARCD, tous ses variants) pendant
    plusieurs jours de ce projet, alors qu'eval_L_CE semblait raisonnable.

    Ce test vérifie directement le sens du décalage : une prédiction
    parfaite du VRAI TOKEN SUIVANT doit donner une loss quasi nulle, et une
    prédiction parfaite du TOKEN COURANT (le comportement de copie, l'ancien
    bug) doit donner une loss élevée.
    """
    import torch
    from arcd.losses import ARCDLoss

    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 1, 5, 2, 60

    input_ids = torch.tensor([[10, 20, 30, 40, 49]])
    labels = torch.tensor([[-100, -100, 30, 40, 49]])  # construction IDENTIQUE à build_example()
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    criterion = ARCDLoss(temperature=2.0)

    # Cas A : prédiction PARFAITE du vrai token suivant -> loss quasi nulle
    logits_correct = torch.full((batch, seq_len, vocab_size), -10.0)
    for i in range(seq_len - 1):
        target_next = labels[0, i + 1].item()
        if target_next != -100:
            logits_correct[0, i, target_next] = 20.0
    _, metrics_correct = criterion(logits_correct, teacher_logits, labels)
    assert metrics_correct["L_CE"] < 0.01

    # Cas B : prédiction PARFAITE du token COURANT (comportement de copie,
    # l'ancien bug) -> loss élevée, car maintenant reconnu comme FAUX.
    logits_copy = torch.full((batch, seq_len, vocab_size), -10.0)
    for i in range(seq_len):
        if labels[0, i].item() != -100:
            logits_copy[0, i, input_ids[0, i].item()] = 20.0
    _, metrics_copy = criterion(logits_copy, teacher_logits, labels)
    assert metrics_copy["L_CE"] > 5.0
