"""
arcd/consensus.py
==================
Médiane pondérée robuste + score de consensus.

Convention de forme, cas LLM :
    teacher_logits : (batch, seq_len, num_teachers, vocab_size)
    -> l'axe Teachers est l'avant-dernier (dim=-2), le vocabulaire est
       le dernier (dim=-1).

En vision, la forme était (batch, num_teachers, num_classes), donc l'axe
Teachers était dim=1 (= dim=-2 aussi, puisqu'il n'y a que 2 axes après le
batch). La convention "axe Teachers = avant-dernier" est donc rétro-
compatible : ce module n'a besoin d'aucun changement de logique, seulement
d'assumer -2 au lieu de 1 en dur.
"""

import torch

from arcd.confidence import EPS, teacher_confidence


def weighted_median(values: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Médiane pondérée le long de `dim`. Fonctionne pour n'importe quel
    nombre de dimensions précédentes (batch seul, ou batch+séquence).
    """
    weights = weights.expand_as(values)

    sorted_vals, sort_idx = torch.sort(values, dim=dim)
    sorted_weights = torch.gather(weights, dim, sort_idx)

    cum_weights = torch.cumsum(sorted_weights, dim=dim)
    total_weights = cum_weights.select(dim, -1).unsqueeze(dim)
    normalized_cum = cum_weights / (total_weights + EPS)

    reached_half = (normalized_cum >= 0.5).float()
    median_idx = reached_half.argmax(dim=dim, keepdim=True)

    median = torch.gather(sorted_vals, dim, median_idx)

    # CORRECTIF : quand le poids cumulé à median_idx tombe EXACTEMENT sur
    # 0.5 (à eps près) -- le cas de deux valeurs à poids égaux, exactement
    # notre configuration à 2 Teachers -- la médiane pondérée standard est
    # la MOYENNE de cette valeur et de la suivante, pas la valeur elle-même.
    # Sans ce correctif, argmax(reached_half) renvoie systématiquement le
    # PREMIER indice atteignant >=0.5, donc systématiquement le MINIMUM des
    # deux valeurs quand N=2 à poids égaux -- ce qui a fait dégénérer C à
    # 1.000 en permanence dans toutes les expériences de ce mémoire (voir
    # chapitre 5, section 5.3.4 : l'explication par la dilution en haute
    # dimension était plausible mais incorrecte -- la cause réelle est ce
    # bug de départage, indépendant de la taille du vocabulaire).
    n = values.size(dim)
    has_next = median_idx < (n - 1)
    next_idx = torch.clamp(median_idx + 1, max=n - 1)
    next_val = torch.gather(sorted_vals, dim, next_idx)

    cum_at_median = torch.gather(normalized_cum, dim, median_idx)
    is_exact_tie = (cum_at_median - 0.5).abs() < 1e-6

    use_average = is_exact_tie & has_next
    median = torch.where(use_average, (median + next_val) / 2.0, median)

    return median.squeeze(dim)


def weighted_mad(values: torch.Tensor, weights: torch.Tensor, median: torch.Tensor, dim: int) -> torch.Tensor:
    """
    MAD pondéré autour d'une médiane déjà calculée.

    CAS SPÉCIAL N=2 (notre configuration partout dans ce mémoire) : la
    médiane pondérée de 2 valeurs désigne TOUJOURS entièrement le Teacher le
    plus confiant comme "gagnant" (sauf égalité exacte des poids), de sorte
    que p_median devient identique à la distribution de ce Teacher. Calculer
    le MAD via CE MÊME mécanisme de médiane pondérée fait alors regagner ce
    même Teacher à 100% une seconde fois (son propre écart à p_median est
    nul), donnant un MAD nul QUEL QUE SOIT le désaccord réel entre les deux
    Teachers -- indépendamment de la taille du vocabulaire, de leur
    spécialité, ou de tout autre facteur. La médiane/MAD, en tant que
    statistiques robustes, n'ont de sens qu'à partir de N=3 observations
    (il faut au moins 3 avis pour qu'une "majorité" puisse voter contre un
    isolé). Pour N=2, la mesure de désaccord directe -- l'écart absolu entre
    les deux Teachers -- est utilisée à la place.
    """
    if values.size(dim) == 2:
        v0 = values.select(dim, 0)
        v1 = values.select(dim, 1)
        return torch.abs(v0 - v1)

    abs_dev = torch.abs(values - median.unsqueeze(dim))
    return weighted_median(abs_dev, weights, dim=dim)


def robust_consensus(teacher_logits: torch.Tensor, temperature: float = 1.0, teacher_dim: int = -2,
                      top_k: int = None):
    """
    Pipeline complet : confiance des Teachers -> médiane pondérée -> MAD -> C, T.

    Args:
        teacher_logits: (..., num_teachers, vocab_size)
                         ex. LLM: (batch, seq_len, num_teachers, vocab_size)
                             vision: (batch, num_teachers, num_classes)
        teacher_dim:    axe des Teachers (-2 par défaut : avant-dernier).
        top_k:          si fourni, restreint le calcul de C aux top_k positions
                         du vocabulaire les plus probables selon le consensus
                         (p_median), au lieu de moyenner le MAD sur l'intégralité
                         du vocabulaire. Corrige la dilution du signal de
                         désaccord identifiée empiriquement (mémoire, chapitre 5,
                         section 5.3.4) : sur un vocabulaire de ~152 000 tokens,
                         la quasi-totalité des positions ont un MAD ~0 (tous les
                         Teachers s'accordent à leur attribuer une probabilité
                         nulle), ce qui noie tout désaccord réel concentré sur
                         quelques positions à forte probabilité. N'affecte QUE
                         C -- p_median (la cible utilisée par L_KD) et T restent
                         calculés sur l'intégralité du vocabulaire, inchangés,
                         pour ne pas modifier la dynamique d'entraînement déjà
                         validée.

    Returns:
        p_median: (..., vocab_size)   — distribution cible pour L_KD (TOUJOURS pleine, top_k n'y touche pas)
        C:        (...,)              — score de consensus, dans [0, 1]
        T:        (...,)              — confiance moyenne des Teachers
        teacher_confidences: (..., num_teachers)
    """
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    confidences = teacher_confidence(teacher_logits, temperature=temperature)  # (..., num_teachers)

    weights = confidences.unsqueeze(-1)  # (..., num_teachers, 1) -> broadcast sur le vocab

    p_median = weighted_median(teacher_probs, weights, dim=teacher_dim)         # (..., vocab_size)
    mad = weighted_mad(teacher_probs, weights, p_median, dim=teacher_dim)       # (..., vocab_size)

    if top_k is not None:
        # Sélectionne les top_k positions où le CONSENSUS lui-même place le
        # plus de probabilité -- c'est là que le désaccord, s'il existe, est
        # le plus significatif. Les positions à probabilité quasi nulle
        # (l'immense majorité du vocabulaire) sont exclues de la moyenne.
        topk_idx = p_median.topk(min(top_k, p_median.size(-1)), dim=-1).indices  # (..., top_k)
        mad_selected = torch.gather(mad, dim=-1, index=topk_idx)                 # (..., top_k)
        mad_scalar = mad_selected.mean(dim=-1)                                    # (...,)
    else:
        mad_scalar = mad.mean(dim=-1)                                            # (...,) — comportement original inchangé

    C = 1.0 / (1.0 + mad_scalar)
    T = confidences.mean(dim=-1)

    return p_median, C, T, confidences
