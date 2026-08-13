#!/usr/bin/env python3
"""
scripts/analyze_seeds.py
==========================
Agrège les runs multi-seeds (voir scripts/run_seeds.sh) : pour chaque
régime, lit le meilleur eval_L_CE_comparable de chaque seed, puis calcule
moyenne ± écart-type. C'est CE tableau, pas un seul run, qui doit aller
dans le mémoire — un seul run par régime ne permet pas de distinguer un
vrai effet d'ARCD d'un simple coup de chance sur l'initialisation.

Usage :
    python scripts/analyze_seeds.py
    python scripts/analyze_seeds.py --checkpoint-root outputs/checkpoints
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

REGIME_METRIC = {
    "student_alone": None,             # déjà de la CE pure -> eval_loss lui-même
    "hinton_kd": "eval_hinton/L_CE",
    "arcd": "eval_arcd/L_CE",
}

REGIME_DISPLAY = {
    "student_alone": "Baseline (Student seul)",
    "hinton_kd": "Hinton KD",
    "arcd": "ARCD",
}


def best_comparable_value(state_file: Path, metric_key):
    with open(state_file) as f:
        state = json.load(f)

    by_epoch = {}
    for entry in state.get("log_history", []):
        if "eval_loss" not in entry or entry.get("epoch") is None:
            continue
        by_epoch[float(entry["epoch"])] = entry

    if not by_epoch:
        return None

    def value_of(entry):
        return entry.get(metric_key, entry["eval_loss"]) if metric_key else entry["eval_loss"]

    best_entry = min(by_epoch.values(), key=value_of)
    return value_of(best_entry), best_entry["epoch"]


def collect_regime(checkpoint_root: Path, regime: str, metric_key):
    regime_dir = checkpoint_root / regime
    if not regime_dir.is_dir():
        return []

    results = []
    for seed_dir in sorted(regime_dir.glob("seed_*")):
        checkpoints = sorted(seed_dir.glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[-1]))
        if not checkpoints:
            continue
        last_ckpt = checkpoints[-1]
        state_file = last_ckpt / "trainer_state.json"
        if not state_file.exists():
            continue

        result = best_comparable_value(state_file, metric_key)
        if result is None:
            continue
        value, epoch = result
        seed = seed_dir.name.replace("seed_", "")
        results.append({"seed": seed, "checkpoint": last_ckpt.name, "epoch": epoch, "value": value})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=str, default="outputs/checkpoints")
    parser.add_argument("--output-csv", type=str, default="outputs/analysis/seeds_summary.csv")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(" AGRÉGATION MULTI-SEEDS — eval_L_CE_comparable")
    print("=" * 70)

    summary_rows = []
    csv_rows = []

    for regime, metric_key in REGIME_METRIC.items():
        results = collect_regime(checkpoint_root, regime, metric_key)
        print(f"\n{REGIME_DISPLAY[regime]} ({regime})")
        if not results:
            print("  Aucun résultat trouvé (lance scripts/run_seeds.sh d'abord).")
            continue

        for r in results:
            print(f"  seed={r['seed']:>3} | epoch={r['epoch']:6.2f} | best={r['value']:.6f}")
            csv_rows.append({"regime": regime, "seed": r["seed"], "epoch": r["epoch"], "value": r["value"]})

        values = [r["value"] for r in results]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        print(f"  -> {len(values)} seed(s) : moyenne={mean:.6f}  écart-type={std:.6f}")
        summary_rows.append({"regime": regime, "n_seeds": len(values), "mean": mean, "std": std})

    print()
    print("=" * 70)
    print(" TABLEAU FINAL (à mettre dans le mémoire)")
    print("=" * 70)
    print(f"{'RÉGIME':<28} {'N SEEDS':>8} {'MOYENNE':>12} {'ÉCART-TYPE':>12}")
    print("-" * 64)
    for row in summary_rows:
        std_str = f"{row['std']:.6f}" if row["std"] == row["std"] else "N/A (1 seed)"
        print(f"{REGIME_DISPLAY[row['regime']]:<28} {row['n_seeds']:>8} {row['mean']:>12.6f} {std_str:>12}")

    if any(row["n_seeds"] < 3 for row in summary_rows):
        print()
        print("ATTENTION : au moins un régime a moins de 3 seeds — l'écart-type sur")
        print("si peu de points reste peu fiable. Vise au moins 3, idéalement 5.")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "seed", "epoch", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV détaillé (une ligne par seed) : {args.output_csv}")


if __name__ == "__main__":
    main()
