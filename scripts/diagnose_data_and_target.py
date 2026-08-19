#!/usr/bin/env python3
"""
scripts/diagnose_data_and_target.py
======================================
Deux vérifications rapides et indépendantes du diagnostic de génération :

(A) p_median (la cible utilisée par L_KD) est-elle une vraie distribution
    de probabilité (somme ~1 sur le vocabulaire) ? Si non, la divergence KL
    calculée contre elle n'a pas le sens qu'on lui prête.

(B) Le masquage des labels est-il correct ? Affiche, pour quelques exemples
    réels, les tokens du prompt (doivent être à IGNORE_INDEX) et les tokens
    de la réponse (doivent être supervisés), pour repérer toute frontière
    mal placée ou tout token spécial mal géré.

Usage :
    python scripts/diagnose_data_and_target.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch

from arcd.consensus import robust_consensus
from data_pipeline.cache import load_teacher_cache
from data_pipeline.prompts import build_example, IGNORE_INDEX
from data_pipeline.tokenizer import get_tokenizer


def test_a_p_median_sums_to_one(cache_path="outputs/teacher_cache_val.pt", n_examples=10, top_k=10):
    print("=" * 78)
    print("TEST A -- p_median est-elle une distribution de probabilité valide ?")
    print("=" * 78)
    cache = load_teacher_cache(cache_path)
    items = list(cache.items())[:n_examples]

    all_sums = []
    for idx, teacher_logits in items:
        tl = teacher_logits.unsqueeze(0).float()
        p_median, C, T, _ = robust_consensus(tl, temperature=2.0, top_k=top_k)
        sums = p_median.sum(dim=-1)
        all_sums.extend(sums.flatten().tolist())

    all_sums = torch.tensor(all_sums)
    print(f"Sur {len(all_sums)} positions (across {n_examples} exemples) :")
    print(f"  somme min  : {all_sums.min().item():.6f}")
    print(f"  somme max  : {all_sums.max().item():.6f}")
    print(f"  somme moy  : {all_sums.mean().item():.6f}")
    if abs(all_sums.mean().item() - 1.0) < 0.01 and (all_sums - 1.0).abs().max().item() < 0.05:
        print("  -> OK, p_median est bien une distribution de probabilité valide (~1.0 partout).")
    else:
        print("  -> ATTENTION : écart significatif à 1.0 -- la cible KD n'est pas normalisée correctement.")
    print()


def test_d_label_masking(train_file="outputs/data/train.json", tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
                          n_examples=3):
    print("=" * 78)
    print("TEST D -- Masquage des labels (prompt masqué, réponse supervisée)")
    print("=" * 78)
    tokenizer = get_tokenizer(tokenizer_name)
    examples = json.load(open(train_file, encoding="utf-8"))[:n_examples]

    for i, ex in enumerate(examples):
        encoded = build_example(tokenizer, ex["prompt"], ex["response"], max_length=256)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]

        n_masked = sum(1 for l in labels if l == IGNORE_INDEX)
        n_supervised = len(labels) - n_masked

        print(f"\n--- Exemple {i + 1} ---")
        print(f"Prompt (tronqué)   : {ex['prompt'][:60]!r}...")
        print(f"Réponse (tronquée) : {ex['response'][:60]!r}...")
        print(f"Longueur totale    : {len(input_ids)} tokens")
        print(f"Tokens masqués (IGNORE_INDEX, doit == longueur du prompt) : {n_masked}")
        print(f"Tokens supervisés (doit == longueur réponse + 1 pour EOS)  : {n_supervised}")

        # Affiche les 5 premiers tokens de la frontière prompt->réponse
        boundary = n_masked
        print("Frontière prompt/réponse (5 tokens avant/après) :")
        for j in range(max(0, boundary - 3), min(len(input_ids), boundary + 5)):
            tok_str = tokenizer.decode([input_ids[j]])
            label_str = "IGNORÉ" if labels[j] == IGNORE_INDEX else f"supervisé({labels[j]})"
            marker = " <-- frontière" if j == boundary else ""
            print(f"    pos {j:>3} : token={tok_str!r:>20}  label={label_str}{marker}")

        # Vérifie que le dernier token est bien l'EOS et qu'il est supervisé
        eos_id = tokenizer.eos_token_id
        last_ok = (input_ids[-1] == eos_id) and (labels[-1] == eos_id)
        print(f"Dernier token == EOS et supervisé : {'OUI' if last_ok else 'NON -- À VÉRIFIER'}")

    print()


if __name__ == "__main__":
    test_a_p_median_sums_to_one()
    test_d_label_masking()
    print("=" * 78)
    print("Si les deux tests sont OK, la cible KD et le masquage des labels sont")
    print("mathématiquement sains -- le problème de génération est ailleurs")
    print("(voir scripts/diagnose_generation.py).")
