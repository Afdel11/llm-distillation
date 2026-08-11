#!/usr/bin/env bash

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="$PROJECT_ROOT/outputs/checkpoints"
ANALYSIS_DIR="$PROJECT_ROOT/outputs/analysis"

mkdir -p "$ANALYSIS_DIR"

CSV_FILE="$ANALYSIS_DIR/checkpoint_analysis.csv"
TXT_FILE="$ANALYSIS_DIR/checkpoint_analysis.txt"

# ------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------

echo "regime,checkpoint,epoch,eval_loss" > "$CSV_FILE"
: > "$TXT_FILE"

echo ""
echo "============================================================"
echo " ANALYSE DES CHECKPOINTS — KNOWLEDGE DISTILLATION"
echo "============================================================"
echo ""

echo "Analyse lancée : $(date)"
echo "Projet         : $PROJECT_ROOT"
echo "Checkpoints    : $CHECKPOINT_ROOT"
echo ""

# ------------------------------------------------------------
# Fonction d'analyse d'un régime
# ------------------------------------------------------------

analyze_regime() {

    local REGIME="$1"
    local DISPLAY_NAME="$2"
    local REGIME_DIR="$CHECKPOINT_ROOT/$REGIME"

    echo ""
    echo "============================================================"
    echo "$DISPLAY_NAME"
    echo "============================================================"
    echo ""

    {
        echo ""
        echo "============================================================"
        echo "$DISPLAY_NAME"
        echo "============================================================"
        echo ""
    } >> "$TXT_FILE"

    # Vérification du dossier
    if [ ! -d "$REGIME_DIR" ]; then
        echo "Aucun dossier trouvé : $REGIME_DIR"
        echo "Aucun dossier trouvé : $REGIME_DIR" >> "$TXT_FILE"
        return
    fi

    # Récupération des checkpoints
    mapfile -t CHECKPOINTS < <(
        find "$REGIME_DIR" \
            -maxdepth 1 \
            -type d \
            -name "checkpoint-*" \
            -printf "%f\n" 2>/dev/null |
        sort -V
    )

    if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
        echo "Aucun checkpoint trouvé."
        echo "Aucun checkpoint trouvé." >> "$TXT_FILE"
        return
    fi

    # --------------------------------------------------------
    # Extraction des métriques
    # --------------------------------------------------------

    for CHECKPOINT in "${CHECKPOINTS[@]}"; do

        STATE_FILE="$REGIME_DIR/$CHECKPOINT/trainer_state.json"

        if [ ! -f "$STATE_FILE" ]; then
            echo ""
            echo "$CHECKPOINT : trainer_state.json absent"
            echo "$CHECKPOINT : trainer_state.json absent" >> "$TXT_FILE"
            continue
        fi

        echo "$CHECKPOINT"

        echo "$CHECKPOINT" >> "$TXT_FILE"

        python - "$STATE_FILE" "$REGIME" "$CHECKPOINT" "$CSV_FILE" <<'PY'
import json
import sys
import csv

state_file = sys.argv[1]
regime = sys.argv[2]
checkpoint = sys.argv[3]
csv_file = sys.argv[4]

with open(state_file, "r") as f:
    state = json.load(f)

history = state.get("log_history", [])

# ------------------------------------------------------------
# On récupère uniquement les entrées contenant eval_loss.
# ------------------------------------------------------------

eval_entries = [
    entry
    for entry in history
    if "eval_loss" in entry
]

# ------------------------------------------------------------
# Un même epoch peut apparaître plusieurs fois dans certains
# historiques. On conserve la dernière occurrence.
# ------------------------------------------------------------

by_epoch = {}

for entry in eval_entries:
    epoch = entry.get("epoch")

    if epoch is None:
        continue

    by_epoch[float(epoch)] = entry

# ------------------------------------------------------------
# Tri chronologique
# ------------------------------------------------------------

entries = sorted(
    by_epoch.items(),
    key=lambda x: x[0]
)

# ------------------------------------------------------------
# Export CSV
# ------------------------------------------------------------

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)

    for epoch, entry in entries:

        eval_loss = entry.get("eval_loss")

        writer.writerow([
            regime,
            checkpoint,
            epoch,
            eval_loss
        ])

        print(
            f"  epoch={epoch:6.2f} | "
            f"eval_loss={eval_loss:.6f}"
        )

# ------------------------------------------------------------
# Meilleur résultat contenu dans ce checkpoint
# ------------------------------------------------------------

if entries:

    best_epoch, best_entry = min(
        entries,
        key=lambda x: x[1]["eval_loss"]
    )

    best_loss = best_entry["eval_loss"]

    print(
        f"  BEST = {best_loss:.6f} "
        f"(epoch {best_epoch:.2f})"
    )

    # Ces informations sont récupérées par Bash via stdout
    print(
        f"__BEST__|{checkpoint}|{best_epoch}|{best_loss}"
    )

else:

    print("  Aucune eval_loss trouvée.")

PY

        echo ""

        # ----------------------------------------------------
        # Résultat du checkpoint dans le rapport texte
        # ----------------------------------------------------

        python - "$STATE_FILE" "$CHECKPOINT" >> "$TXT_FILE" <<'PY'
import json
import sys

state_file = sys.argv[1]
checkpoint = sys.argv[2]

with open(state_file, "r") as f:
    state = json.load(f)

history = state.get("log_history", [])

entries = [
    x for x in history
    if "eval_loss" in x and x.get("epoch") is not None
]

if entries:

    by_epoch = {}

    for x in entries:
        by_epoch[float(x["epoch"])] = x

    entries = list(by_epoch.values())

    best = min(
        entries,
        key=lambda x: x["eval_loss"]
    )

    print(
        f"  epoch={best['epoch']:.2f} | "
        f"eval_loss={best['eval_loss']:.6f}"
    )

PY

    done

    # --------------------------------------------------------
    # Meilleur checkpoint global du régime
    # --------------------------------------------------------

    python - "$REGIME_DIR" "$DISPLAY_NAME" >> "$TXT_FILE" <<'PY'
import json
import sys
from pathlib import Path

regime_dir = Path(sys.argv[1])
display_name = sys.argv[2]

all_entries = []

for state_file in regime_dir.glob("checkpoint-*/trainer_state.json"):

    checkpoint = state_file.parent.name

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except Exception:
        continue

    for entry in state.get("log_history", []):

        if "eval_loss" not in entry:
            continue

        epoch = entry.get("epoch")

        if epoch is None:
            continue

        all_entries.append({
            "checkpoint": checkpoint,
            "epoch": float(epoch),
            "eval_loss": float(entry["eval_loss"])
        })

if not all_entries:
    print()
    print("  Aucun résultat de validation disponible.")
    sys.exit(0)

best = min(
    all_entries,
    key=lambda x: x["eval_loss"]
)

print()
print("  ----------------------------------------------------------")
print("  MEILLEUR RÉSULTAT")
print("  ----------------------------------------------------------")
print(f"  Régime     : {display_name}")
print(f"  Checkpoint : {best['checkpoint']}")
print(f"  Epoch      : {best['epoch']:.2f}")
print(f"  Eval loss  : {best['eval_loss']:.6f}")
print("  ----------------------------------------------------------")

PY

}

# ------------------------------------------------------------
# Analyse des trois régimes
# ------------------------------------------------------------

analyze_regime \
    "student_alone" \
    "BASELINE — STUDENT ALONE"

analyze_regime \
    "hinton_kd" \
    "HINTON — KNOWLEDGE DISTILLATION"

analyze_regime \
    "arcd" \
    "ARCD — ADAPTIVE ROBUST CONSENSUS DISTILLATION"

# ------------------------------------------------------------
# Tableau comparatif final
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo " COMPARAISON DES TROIS MÉTHODES"
echo "============================================================"
echo ""

{
    echo ""
    echo "============================================================"
    echo " COMPARAISON DES TROIS MÉTHODES"
    echo "============================================================"
    echo ""
} >> "$TXT_FILE"

python - "$CSV_FILE" <<'PY'

import csv
import sys
from collections import defaultdict

csv_file = sys.argv[1]

data = defaultdict(list)

with open(csv_file, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        try:
            data[row["regime"]].append({
                "checkpoint": row["checkpoint"],
                "epoch": float(row["epoch"]),
                "eval_loss": float(row["eval_loss"])
            })
        except:
            pass

print(
    f"{'RÉGIME':<20} "
    f"{'CHECKPOINT':<20} "
    f"{'EPOCH':>8} "
    f"{'BEST EVAL LOSS':>16}"
)

print("-" * 70)

for regime, entries in data.items():

    if not entries:
        continue

    best = min(
        entries,
        key=lambda x: x["eval_loss"]
    )

    print(
        f"{regime:<20} "
        f"{best['checkpoint']:<20} "
        f"{best['epoch']:>8.2f} "
        f"{best['eval_loss']:>16.6f}"
    )

PY

# ------------------------------------------------------------
# Génération d'un résumé global dans TXT
# ------------------------------------------------------------

{
    echo ""
    echo "============================================================"
    echo " FICHIERS GÉNÉRÉS"
    echo "============================================================"
    echo ""
    echo "CSV : $CSV_FILE"
    echo "TXT : $TXT_FILE"
    echo ""
    echo "Le CSV peut être utilisé pour générer les graphiques"
    echo "d'analyse du mémoire."
    echo ""
} >> "$TXT_FILE"

echo ""
echo "============================================================"
echo " FICHIERS GÉNÉRÉS"
echo "============================================================"
echo ""
echo "CSV : outputs/analysis/checkpoint_analysis.csv"
echo "TXT : outputs/analysis/checkpoint_analysis.txt"
echo ""
echo "Analyse terminée."
echo ""
