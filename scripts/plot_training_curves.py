#!/usr/bin/env python3
"""
scripts/plot_training_curves.py
=================================
Lit outputs/analysis/full_history.csv (généré par export_full_history.py)
et trace eval_L_CE en fonction de l'epoch, une courbe par régime, avec une
bande ± écart-type entre seeds à chaque epoch.

Nécessite matplotlib et pandas :
    pip install matplotlib pandas

Usage :
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --input outputs/analysis/full_history.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REGIME_DISPLAY = {
    "student_alone": "Baseline (Student seul)",
    "hinton_kd": "Hinton KD",
    "multi_teacher_fixed": "Multi-Teacher poids fixe (contrôle)",
    "arcd": "ARCD (méthode proposée)",
}
REGIME_COLOR = {
    "student_alone": "#808080",
    "hinton_kd": "#5B8FD6",
    "multi_teacher_fixed": "#E8A33D",
    "arcd": "#D65B5B",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/analysis/full_history.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for regime in ["student_alone", "hinton_kd", "multi_teacher_fixed", "arcd"]:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue

        # Regroupe par epoch (arrondie pour aligner les points entre seeds,
        # au cas où l'early stopping arrête chaque seed à un epoch différent)
        sub = sub.copy()
        sub["epoch_r"] = sub["epoch"].round().astype(int)
        grouped = sub.groupby("epoch_r")["eval_L_CE"]
        mean = grouped.mean()
        std = grouped.std()  # NaN si un seul seed encore présent à cet epoch -> non affiché

        color = REGIME_COLOR[regime]
        ax.plot(mean.index, mean.values, label=REGIME_DISPLAY[regime],
                 color=color, linewidth=2, marker="o", markersize=4)
        if std.notna().any():
            ax.fill_between(mean.index, mean.values - std.fillna(0), mean.values + std.fillna(0),
                             color=color, alpha=0.15)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("eval_L_CE (cross-entropy pure sur validation)", fontsize=11)
    ax.set_title("Évolution de la perte de validation par epoch\n(moyenne ± écart-type sur les seeds)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    png_path = out_dir / "courbes_entrainement.png"
    pdf_path = out_dir / "courbes_entrainement.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Figures sauvegardées : {png_path}, {pdf_path}")

    # Note : les epochs tardives peuvent n'avoir qu'1 seed encore actif (les
    # autres ayant déclenché l'early stopping plus tôt) -> la bande
    # d'incertitude disparaît logiquement à droite du graphique pour ces
    # régions, ce n'est pas un bug.


if __name__ == "__main__":
    main()
