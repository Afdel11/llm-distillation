#!/usr/bin/env python3
"""
scripts/prepare_curriculum.py
================================
Découpe un jeu d'entraînement en étapes CUMULATIVES triées par difficulté
(longueur de la réponse en caractères, un proxy simple et standard en
curriculum learning), pour un entraînement progressif : court d'abord,
puis court+moyen, puis court+moyen+long, jamais de retour en arrière sur
ce qui a déjà été vu.

Référence directe : Liu et Zhang (2025), déjà citée au chapitre 5 du
mémoire pour le diagnostic du mode collapse, proposent précisément un
cadre d'apprentissage par curriculum comme piste corrective.

Usage :
    python scripts/prepare_curriculum.py
    python scripts/prepare_curriculum.py --input outputs/data/train.json --n_stages 4
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/data/train.json")
    parser.add_argument("--output_dir", type=str, default="outputs/data_curriculum")
    parser.add_argument("--n_stages", type=int, default=4)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        examples = json.load(f)

    print(f"{len(examples)} exemples chargés depuis {args.input}")

    # Tri par difficulté croissante : longueur totale (prompt + réponse) en
    # caractères, un proxy simple -- une réponse courte est structurellement
    # plus facile à apprendre pour un petit modèle from scratch qu'une
    # réponse longue et développée.
    examples_sorted = sorted(examples, key=lambda e: len(e["prompt"]) + len(e["response"]))

    os.makedirs(args.output_dir, exist_ok=True)

    n = len(examples_sorted)
    for stage in range(1, args.n_stages + 1):
        cutoff = int(n * stage / args.n_stages)
        stage_examples = examples_sorted[:cutoff]
        path = os.path.join(args.output_dir, f"stage{stage}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stage_examples, f, ensure_ascii=False, indent=2)

        lengths = [len(e["prompt"]) + len(e["response"]) for e in stage_examples]
        print(f"Étape {stage}/{args.n_stages} : {len(stage_examples)} exemples "
              f"(longueur max {max(lengths)} caractères) -> {path}")

    print(f"\n{args.n_stages} fichiers cumulatifs écrits dans {args.output_dir}/")
    print("Lance scripts/run_curriculum.sh pour l'entraînement progressif.")


if __name__ == "__main__":
    main()
