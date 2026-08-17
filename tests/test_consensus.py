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


def test_two_teachers_median_is_not_degenerate_to_winner_take_all():
    """
    RÉGRESSION -- bug découvert lors de la rédaction du mémoire (chapitre 5) :
    avec exactement 2 Teachers (notre configuration partout dans ce projet),
    la médiane pondérée désigne TOUJOURS entièrement le Teacher le plus
    confiant comme "gagnant" (p_median devient identique à sa distribution),
    et le MAD calculé via ce même mécanisme retombe alors à 0 -- donc C=1.0
    -- QUEL QUE SOIT le désaccord réel entre les deux Teachers. Ce test
    vérifie, avec des logits aléatoires continus (pas de cas construit à la
    main), que C n'est plus jamais exactement 1.0 et montre une vraie
    variance -- la signature du bug corrigé.
    """
    torch.manual_seed(42)
    batch, seq_len, num_teachers, vocab_size = 4, 20, 2, 1000
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size) * 3.0

    _, C, _, _ = robust_consensus(teacher_logits, temperature=1.0)

    assert (C == 1.0).sum().item() == 0, (
        "C ne doit plus jamais être EXACTEMENT 1.0 avec des Teachers aléatoires "
        "en désaccord -- signe que le bug de médiane dégénérée à N=2 est revenu."
    )
    assert C.std().item() > 1e-5, "C doit montrer une vraie variance entre positions."


def test_two_teachers_perfect_agreement_still_gives_high_C():
    """Vérifie que le correctif n'a pas cassé le cas simple : 2 Teachers
    strictement identiques doivent toujours donner un C proche de 1.0."""
    torch.manual_seed(0)
    logits_same = torch.randn(1, 1, 1, 500) * 3.0
    teacher_logits = logits_same.expand(1, 1, 2, 500)
    _, C, _, _ = robust_consensus(teacher_logits, temperature=1.0)
    assert C.item() > 0.999


def test_two_teachers_strong_disagreement_gives_low_C_with_topk():
    """Vérifie que le correctif détecte bien un désaccord frontal simulé
    (2 Teachers sûrs chacun d'un token différent) -- AVEC top_k, qui isole
    la position en désaccord de la masse de positions à probabilité quasi
    nulle sur lesquelles les 2 Teachers sont trivialement d'accord."""
    vocab_size = 1000
    logits_a = torch.full((1, 1, vocab_size), -10.0)
    logits_a[0, 0, 5] = 10.0
    logits_b = torch.full((1, 1, vocab_size), -10.0)
    logits_b[0, 0, 7] = 10.0
    teacher_logits = torch.stack([logits_a, logits_b], dim=2)

    _, C_full, _, _ = robust_consensus(teacher_logits, temperature=1.0, top_k=None)
    _, C_topk, _, _ = robust_consensus(teacher_logits, temperature=1.0, top_k=10)

    # Sans top_k : le désaccord existe (C < 1.0, bug corrigé) mais reste
    # dilué par les ~998 positions où les deux Teachers trivialement
    # d'accord (probabilité quasi nulle des deux côtés).
    assert C_full.item() < 1.0
    assert C_full.item() > 0.9  # dilué, mais le signal existe désormais

    # Avec top_k, restreint aux positions où le consensus concentre sa
    # probabilité, le désaccord frontal devient plus net (C nettement plus
    # bas que sur vocabulaire complet), même s'il reste partiel : les deux
    # Teachers ayant une confiance globale quasi identique (distributions
    # symétriques dans ce cas construit), la médiane elle-même moyenne
    # partiellement leurs positions de désaccord plutôt que de désigner un
    # vainqueur net.
    assert C_topk.item() < C_full.item()
    assert C_topk.item() < 0.9


def test_top_k_sharpens_disagreement_signal_vs_full_vocab():
    """Vérifie l'hypothèse de dilution (mémoire, chapitre 5, section 5.3.4) :
    à taille de vocabulaire réaliste, restreindre C aux top_k positions les
    plus probables doit révéler PLUS de désaccord (C plus bas, plus de
    variance) que la moyenne sur l'intégralité du vocabulaire."""
    torch.manual_seed(42)
    batch, seq_len, num_teachers, vocab_size = 4, 20, 2, 1000
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size) * 3.0

    _, C_full, _, _ = robust_consensus(teacher_logits, temperature=1.0, top_k=None)
    _, C_top10, _, _ = robust_consensus(teacher_logits, temperature=1.0, top_k=10)

    assert C_top10.mean().item() < C_full.mean().item()
    assert C_top10.std().item() > C_full.std().item()
