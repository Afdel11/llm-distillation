import matplotlib.pyplot as plt
import numpy as np

# Données réelles issues de outputs/analysis/seeds_summary.csv (3 seeds, GPU distant)
data = {
    "Baseline\n(Student seul)": [5.387998, 5.527176, 5.511904],
    "Hinton KD": [1.163417, 1.324490, 1.316803],
    "ARCD\n(méthode proposée)": [0.746446, 0.777631, 0.762140],
}

labels = list(data.keys())
means = [np.mean(v) for v in data.values()]
stds = [np.std(v, ddof=1) for v in data.values()]
seeds_vals = list(data.values())

fig, ax = plt.subplots(figsize=(7, 5.5))

colors = ["#B0B0B0", "#5B8FD6", "#D65B5B"]
x = np.arange(len(labels))

bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black",
               linewidth=0.8, zorder=3, error_kw={"linewidth": 1.5, "zorder": 4})

# Points individuels par seed, en superposition (transparence des runs réels)
rng = np.random.default_rng(0)
for i, vals in enumerate(seeds_vals):
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), x[i]) + jitter, vals, color="black", s=28,
               zorder=5, alpha=0.75, label="Seeds individuels" if i == 0 else None)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("eval_L_CE (cross-entropy pure sur validation)", fontsize=11)
ax.set_title("Comparaison des 3 régimes — moyenne ± écart-type sur 3 seeds", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.legend(loc="upper right", fontsize=9)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(x[i], m + s + 0.15, f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("/home/claude/figures/resultats_finaux_3seeds.png", dpi=300, bbox_inches="tight")
plt.savefig("/home/claude/figures/resultats_finaux_3seeds.pdf", bbox_inches="tight")
print("Figures sauvegardées.")
