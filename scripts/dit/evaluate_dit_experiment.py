#!/usr/bin/env python3
"""
scripts/dit/evaluate_dit_experiment.py
=========================================
Extrait eval_L_CE (ou eval_loss) de l'historique d'entraînement de chaque
checkpoint DIT disponible, affiche un tableau récapitulatif, et trace les
courbes de comparaison (étape 1 : 149 exemples vs étape 2 : 1980 exemples,
pour le Teacher et le Student).

Gère gracieusement les checkpoints manquants (par exemple si l'historique
d'une étape a été écrasé par l'étape suivante) -- affiche ce qui est
disponible, signale clairement ce qui ne l'est pas.

Usage :
    python scripts/dit/evaluate_dit_experiment.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (label affiché, dossier de checkpoints, sous-régime, clé de métrique)
CHECKPOINTS = [
    ("Teacher Qwen0.5B — étape 1 (149 ex., fuite train/val)", "outputs/checkpoints_dit_qwen05b_stage1_149ex", "student_alone", "eval_loss"),
    ("Teacher Qwen0.5B — étape 2 (2256 ex., fuite corrigée)", "outputs/checkpoints_dit_qwen05b", "student_alone", "eval_loss"),
    ("Teacher Qwen0.5B — étape 3 (continuité)", "outputs/checkpoints_dit_qwen05b_stage3", "student_alone", "eval_loss"),
    ("Student ARCD — étape 1 (149 ex., fuite train/val)", "outputs/checkpoints_dit_smart_stage1_149ex", "arcd_topk", "eval_arcd/L_CE"),
    ("Student ARCD — étape 2 (2256 ex., fuite corrigée)", "outputs/checkpoints_dit_smart", "arcd_topk", "eval_arcd/L_CE"),
    ("Student ARCD — étape 3 (continuité)", "outputs/checkpoints_dit_smart_stage3", "arcd_topk", "eval_arcd/L_CE"),
]

# Référence : meilleur résultat général (hors DIT) de chaque régime, pour
# rappeler POURQUOI multi_teacher_fixed avait été retenu comme base avant
# la spécialisation DIT -- voir outputs/analysis/seeds_summary.csv du
# projet général pour le détail complet.
GENERAL_REFERENCE = {
    "Baseline (student_alone), général patient": 5.531,
    "Hinton KD, général patient": 5.502,
    "Multi-Teacher fixe, général patient (retenu comme base)": 5.344,
    "ARCD, général patient": 5.642,
}


def find_trainer_state(checkpoint_dir, regime):
    """Cherche trainer_state.json dans checkpoint_dir/regime/seed_0/checkpoint-*/,
    et si absent, tente checkpoint_dir/ directement (cas d'un dossier _final
    copié isolément, sans historique)."""
    pattern = os.path.join(checkpoint_dir, regime, "seed_0", "checkpoint-*", "trainer_state.json")
    candidates = sorted(glob.glob(pattern), key=lambda p: int(p.split("checkpoint-")[-1].split("/")[0]))
    if candidates:
        return candidates[-1]  # le plus avancé
    return None


def main():
    print("=" * 78)
    print(" ÉVALUATION DE L'EXPÉRIENCE DIT — étape 1 (149 ex.) vs étape 2 (1980 ex.)")
    print("=" * 78)

    all_curves = {}

    for label, ckpt_dir, regime, metric_key in CHECKPOINTS:
        state_path = find_trainer_state(ckpt_dir, regime)
        print(f"\n{label}")
        print(f"  Dossier cherché : {ckpt_dir}/{regime}/seed_0/checkpoint-*/")
        if state_path is None:
            print("  -> INTROUVABLE (historique probablement écrasé par un entraînement suivant, "
                  "ou jamais lancé sous ce nom). Seuls les poids finaux sont peut-être disponibles "
                  "pour les tests de génération.")
            continue

        state = json.load(open(state_path, encoding="utf-8"))
        epochs, values = [], []
        for e in state["log_history"]:
            val = e.get(metric_key, e.get("eval_loss"))
            if val is not None and "epoch" in e and ("eval_loss" in e or metric_key in e):
                epochs.append(e["epoch"])
                values.append(val)

        if not epochs:
            print(f"  -> trouvé ({state_path}) mais aucune valeur '{metric_key}' dans l'historique.")
            continue

        best_idx = min(range(len(values)), key=lambda i: values[i])
        print(f"  -> {len(epochs)} points d'évaluation, epoch 1 à {epochs[-1]:.0f}")
        print(f"  -> meilleur {metric_key} = {values[best_idx]:.4f} à l'epoch {epochs[best_idx]:.0f}")
        all_curves[label] = (epochs, values)

    if not all_curves:
        print("\nAucune courbe disponible -- vérifie les chemins ci-dessus.")
    else:
        # --- Tracé des courbes disponibles ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        teacher_curves = {k: v for k, v in all_curves.items() if k.startswith("Teacher")}
        student_curves = {k: v for k, v in all_curves.items() if k.startswith("Student")}

        for ax, curves, title in [(axes[0], teacher_curves, "Teacher Qwen0.5B (fine-tuning direct)"),
                                    (axes[1], student_curves, "Student ARCD (distillation)")]:
            for label, (epochs, values) in curves.items():
                short_label = label.split("—")[-1].strip()
                ax.plot(epochs, values, marker="o", markersize=3, label=short_label)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("eval_L_CE")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle("DIT — Progression par étapes (149 -> 2256 exemples -> étape 3)")
        plt.tight_layout()

        os.makedirs("outputs/analysis", exist_ok=True)
        out_path = "outputs/analysis/dit_courbes_comparaison.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nCourbes sauvegardées -> {out_path}")

    print("\n" + "=" * 78)
    print(" RAPPEL — pourquoi Multi-Teacher fixe avait été retenu comme base générale")
    print("=" * 78)
    for label, val in GENERAL_REFERENCE.items():
        marker = "  <-- base choisie pour le fine-tuning DIT" if "retenu" in label else ""
        print(f"  {label:55s} eval_L_CE = {val:.3f}{marker}")


if __name__ == "__main__":
    main()
