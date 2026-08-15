#!/usr/bin/env python3
"""
scripts/analyze_seeds.py
==========================
Agrège les runs multi-seeds (voir scripts/run_seeds.sh) : pour chaque
régime, lit le meilleur eval_L_CE_comparable de chaque seed (moyenne ±
écart-type — LE tableau de comparaison à mettre dans le mémoire), et pour
les régimes qui calculent un consensus robuste (multi_teacher_fixed, arcd),
lit aussi C et T à ce même epoch, à titre diagnostique.

Pourquoi comparer C et T entre multi_teacher_fixed et arcd alors que la
formule qui les calcule est identique dans les deux régimes (même
robust_consensus, mêmes 2 Teachers) : parce que C et T sont mesurés sur les
prédictions des Teachers, gelés, pour les positions de validation -- ils ne
dépendent QUE des Teachers, pas du Student. Ils devraient donc être quasi
identiques entre les deux régimes (aux epochs de sélection près, si les
deux runs s'arrêtent à des moments différents) ; un écart notable serait le
signe que quelque chose d'autre diverge et mériterait d'être creusé avant
de conclure quoi que ce soit sur l'apport de l'adaptivité.

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
    "multi_teacher_fixed": "eval_fixedmt/L_CE",
    "arcd": "eval_arcd/L_CE",
    "arcd_diverse": "eval_arcd/L_CE",   # même trainer/préfixe de log qu'"arcd"
}

# Préfixe utilisé pour retrouver C et T à l'epoch retenu -- None si le
# régime n'a pas de consensus robuste (student_alone, hinton_kd single-Teacher).
REGIME_CT_PREFIX = {
    "student_alone": None,
    "hinton_kd": None,
    "multi_teacher_fixed": "eval_fixedmt",
    "arcd": "eval_arcd",
    "arcd_diverse": "eval_arcd",
}

REGIME_DISPLAY = {
    "student_alone": "Baseline (Student seul)",
    "hinton_kd": "Hinton KD",
    "multi_teacher_fixed": "Multi-Teacher poids fixe (contrôle)",
    "arcd": "ARCD",
    "arcd_diverse": "ARCD (Teacher diversifié — Coder)",
}


def best_comparable_entry(state_file: Path, metric_key):
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
    return best_entry, value_of(best_entry)


def collect_regime(checkpoint_root: Path, regime: str, metric_key, ct_prefix):
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

        found = best_comparable_entry(state_file, metric_key)
        if found is None:
            continue
        best_entry, value = found
        seed = seed_dir.name.replace("seed_", "")

        row = {"seed": seed, "checkpoint": last_ckpt.name, "epoch": best_entry["epoch"], "value": value}
        if ct_prefix:
            row["C"] = best_entry.get(f"{ct_prefix}/C")
            row["T"] = best_entry.get(f"{ct_prefix}/T")
        else:
            row["C"] = None
            row["T"] = None
        results.append(row)

    return results


def fmt(x, spec=".4f"):
    return format(x, spec) if x is not None else "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=str, default="outputs/checkpoints")
    parser.add_argument("--output-csv", type=str, default="outputs/analysis/seeds_summary.csv")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(" AGRÉGATION MULTI-SEEDS — eval_L_CE_comparable (+ C, T pour les régimes concernés)")
    print("=" * 78)

    summary_rows = []
    csv_rows = []

    for regime, metric_key in REGIME_METRIC.items():
        ct_prefix = REGIME_CT_PREFIX[regime]
        results = collect_regime(checkpoint_root, regime, metric_key, ct_prefix)
        print(f"\n{REGIME_DISPLAY[regime]} ({regime})")
        if not results:
            print("  Aucun résultat trouvé (lance scripts/run_seeds.sh d'abord).")
            continue

        for r in results:
            ct_str = f" | C={fmt(r['C'])} T={fmt(r['T'])}" if ct_prefix else ""
            print(f"  seed={r['seed']:>3} | epoch={r['epoch']:6.2f} | best={r['value']:.6f}{ct_str}")
            csv_rows.append({
                "regime": regime, "seed": r["seed"], "epoch": r["epoch"], "value": r["value"],
                "C": r["C"], "T": r["T"],
            })

        values = [r["value"] for r in results]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        line = f"  -> {len(values)} seed(s) : moyenne={mean:.6f}  écart-type={std:.6f}"
        row = {"regime": regime, "n_seeds": len(values), "mean": mean, "std": std, "C_mean": None, "T_mean": None}
        if ct_prefix:
            c_vals = [r["C"] for r in results if r["C"] is not None]
            t_vals = [r["T"] for r in results if r["T"] is not None]
            if c_vals:
                row["C_mean"] = statistics.mean(c_vals)
                line += f"  | C moyen={row['C_mean']:.4f}"
            if t_vals:
                row["T_mean"] = statistics.mean(t_vals)
                line += f"  T moyen={row['T_mean']:.4f}"
        print(line)
        summary_rows.append(row)

    print()
    print("=" * 78)
    print(" TABLEAU FINAL (à mettre dans le mémoire)")
    print("=" * 78)
    print(f"{'RÉGIME':<28} {'N':>3} {'MOYENNE':>12} {'ÉCART-TYPE':>12} {'C moyen':>10} {'T moyen':>10}")
    print("-" * 78)
    for row in summary_rows:
        std_str = f"{row['std']:.6f}" if row["std"] == row["std"] else "N/A (1 seed)"
        print(f"{REGIME_DISPLAY[row['regime']]:<28} {row['n_seeds']:>3} {row['mean']:>12.6f} "
              f"{std_str:>12} {fmt(row['C_mean']):>10} {fmt(row['T_mean']):>10}")

    if any(row["n_seeds"] < 3 for row in summary_rows):
        print()
        print("ATTENTION : au moins un régime a moins de 3 seeds — l'écart-type sur")
        print("si peu de points reste peu fiable. Vise au moins 3, idéalement 5.")

    # Comparaison ciblée multi_teacher_fixed vs arcd, si les deux sont présents
    mt_row = next((r for r in summary_rows if r["regime"] == "multi_teacher_fixed"), None)
    arcd_row = next((r for r in summary_rows if r["regime"] == "arcd"), None)
    if mt_row and arcd_row:
        print()
        print("=" * 78)
        print(" LECTURE — ADAPTIVITÉ VS SIMPLE PRÉSENCE DE 2 TEACHERS")
        print("=" * 78)
        hinton_row = next((r for r in summary_rows if r["regime"] == "hinton_kd"), None)
        print(f"  Hinton (1 Teacher, poids fixe)              : {fmt(hinton_row['mean'] if hinton_row else None, '.4f')}")
        print(f"  Multi-Teacher fixe (2 Teachers, poids fixe)  : {fmt(mt_row['mean'], '.4f')}")
        print(f"  ARCD (2 Teachers, poids ADAPTATIF)           : {fmt(arcd_row['mean'], '.4f')}")
        if mt_row["C_mean"] is not None and arcd_row["C_mean"] is not None:
            print(f"  Écart de C moyen (fixe vs adaptatif)         : {abs(mt_row['C_mean'] - arcd_row['C_mean']):.4f}")
        if mt_row["T_mean"] is not None and arcd_row["T_mean"] is not None:
            print(f"  Écart de T moyen (fixe vs adaptatif)         : {abs(mt_row['T_mean'] - arcd_row['T_mean']):.4f}")
        print("  Si ARCD << Multi-Teacher fixe : l'adaptivité de lambda(x) apporte un")
        print("  gain réel, isolé de la simple présence d'un second Teacher.")
        print("  Si ARCD ≈ Multi-Teacher fixe : le gain vient surtout d'avoir 2 Teachers,")
        print("  pas de l'adaptivité -- conclusion plus modeste, mais honnête.")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "seed", "epoch", "value", "C", "T"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV détaillé (une ligne par seed, avec C/T quand disponibles) : {args.output_csv}")


if __name__ == "__main__":
    main()
