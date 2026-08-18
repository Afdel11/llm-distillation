#!/usr/bin/env python3
"""
scripts/diagnose_consensus.py
================================
Teste plusieurs valeurs de top_k directement sur un cache de logits
Teachers DÉJÀ CALCULÉ (outputs/teacher_cache_val.pt ou équivalent), sans
aucun réentraînement. Répond en quelques secondes à la question : "top_k
révèle-t-il un vrai désaccord entre Teachers sur ce jeu de données précis,
une fois le bug de médiane dégénérée corrigé ?"

Usage :
    python scripts/diagnose_consensus.py
    python scripts/diagnose_consensus.py --cache outputs/teacher_cache_val_diverse.pt
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arcd.consensus import robust_consensus
from data_pipeline.cache import load_teacher_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="outputs/teacher_cache_val.pt")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top_ks", type=str, default="None,200,100,50,20,10,5,1")
    args = parser.parse_args()

    print(f"Chargement du cache : {args.cache}")
    cache = load_teacher_cache(args.cache)
    print(f"  {len(cache)} exemples en cache\n")

    top_k_values = [None if v == "None" else int(v) for v in args.top_ks.split(",")]

    print(f"{'top_k':>8} | {'C moyen':>12} | {'C écart-type':>14} | {'C min':>10} | {'C max':>10}")
    print("-" * 68)

    for top_k in top_k_values:
        all_C = []
        for idx, teacher_logits in cache.items():
            # teacher_logits attendu: (seq_len, num_teachers, vocab_size)
            tl = teacher_logits.unsqueeze(0).float()  # ajoute l'axe batch -> (1, seq_len, teachers, vocab)
            _, C, _, _ = robust_consensus(tl, temperature=args.temperature, top_k=top_k)
            all_C.extend(C.flatten().tolist())

        mean_C = statistics.mean(all_C)
        std_C = statistics.stdev(all_C) if len(all_C) > 1 else 0.0
        min_C, max_C = min(all_C), max(all_C)
        label = "complet" if top_k is None else str(top_k)
        print(f"{label:>8} | {mean_C:>12.6f} | {std_C:>14.8f} | {min_C:>10.6f} | {max_C:>10.6f}")

    print()
    print("Lecture : si C moyen baisse et l'écart-type augmente à mesure que top_k")
    print("diminue, un vrai désaccord entre Teachers existe et était dilué par le")
    print("vocabulaire complet -- confirmation de l'hypothèse du chapitre 5.")
    print("Si C reste ~identique quel que soit top_k, les deux Teachers sont réellement")
    print("d'accord sur ce jeu de données, même sur leurs positions les plus probables.")


if __name__ == "__main__":
    main()
