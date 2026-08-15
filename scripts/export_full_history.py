#!/usr/bin/env python3
"""
scripts/export_full_history.py
================================
Contrairement à analyze_seeds.py (qui ne garde que le MEILLEUR epoch par
seed), ce script exporte TOUT l'historique epoch par epoch, pour tous les
seeds — nécessaire pour tracer des courbes d'entraînement (eval_L_CE vs
epoch) avec bande d'incertitude ± écart-type entre seeds.

Usage :
    python scripts/export_full_history.py
    # -> outputs/analysis/full_history.csv (colonnes: regime, seed, epoch, eval_L_CE)
"""

import argparse
import csv
import json
from pathlib import Path

REGIME_METRIC = {
    "student_alone": None,
    "hinton_kd": "eval_hinton/L_CE",
    "multi_teacher_fixed": "eval_fixedmt/L_CE",
    "arcd": "eval_arcd/L_CE",
}


def export_regime(checkpoint_root: Path, regime: str, metric_key, rows: list):
    regime_dir = checkpoint_root / regime
    if not regime_dir.is_dir():
        return

    for seed_dir in sorted(regime_dir.glob("seed_*")):
        checkpoints = sorted(seed_dir.glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[-1]))
        if not checkpoints:
            continue
        state_file = checkpoints[-1] / "trainer_state.json"
        if not state_file.exists():
            continue

        with open(state_file) as f:
            state = json.load(f)

        by_epoch = {}
        for entry in state.get("log_history", []):
            if "eval_loss" not in entry or entry.get("epoch") is None:
                continue
            by_epoch[float(entry["epoch"])] = entry

        seed = seed_dir.name.replace("seed_", "")
        for epoch, entry in sorted(by_epoch.items()):
            value = entry.get(metric_key, entry["eval_loss"]) if metric_key else entry["eval_loss"]
            rows.append({"regime": regime, "seed": seed, "epoch": epoch, "eval_L_CE": value})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=str, default="outputs/checkpoints")
    parser.add_argument("--output-csv", type=str, default="outputs/analysis/full_history.csv")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for regime, metric_key in REGIME_METRIC.items():
        export_regime(checkpoint_root, regime, metric_key, rows)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "seed", "epoch", "eval_L_CE"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} lignes exportées -> {args.output_csv}")
    print("Récupère ce fichier (scp, cat + copier-coller, ou upload dans le chat) "
          "pour générer les courbes d'entraînement.")


if __name__ == "__main__":
    main()
