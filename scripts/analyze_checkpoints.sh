#!/usr/bin/env bash
# scripts/analyze_checkpoints.sh
# ================================
# Extrait, pour chaque régime (student_alone, hinton_kd, arcd), l'historique
# eval_loss ET une métrique COMPARABLE entre les trois : la cross-entropy
# pure sur validation (eval_L_CE_comparable).
#
# Pourquoi pas juste eval_loss : eval_loss est un mélange différent selon le
# régime (CE pure pour student_alone, alpha*KD+(1-alpha)*CE pour hinton_kd,
# lambda(x)*KD+(1-lambda(x))*CE pour arcd, avec lambda variable) -> comparer
# ces trois nombres directement, c'est comparer des grandeurs différentes qui
# portent le même nom. eval_L_CE_comparable isole la seule composante
# strictement identique dans les trois cas : la qualité de prédiction du
# texte, indépendamment de combien chaque méthode s'appuie sur ses Teachers.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="$PROJECT_ROOT/outputs/checkpoints"
ANALYSIS_DIR="$PROJECT_ROOT/outputs/analysis"

# --seed N : n'analyse que ce seed précis. Sans --seed : analyse le seed_0
# par défaut (comportement explicite plutôt que de deviner lequel prendre
# parmi plusieurs seeds disponibles).
SEED_FILTER="0"
SEED_NEXT=false
for arg in "$@"; do
  case "$arg" in
    --seed=*) SEED_FILTER="${arg#--seed=}" ;;
    --seed) SEED_NEXT=true ;;
    *) if [[ "$SEED_NEXT" == "true" ]]; then SEED_FILTER="$arg"; SEED_NEXT=false; fi ;;
  esac
done

mkdir -p "$ANALYSIS_DIR"

CSV_FILE="$ANALYSIS_DIR/checkpoint_analysis_seed${SEED_FILTER}.csv"
TXT_FILE="$ANALYSIS_DIR/checkpoint_analysis_seed${SEED_FILTER}.txt"

echo "regime,checkpoint,epoch,eval_loss,eval_L_CE_comparable" > "$CSV_FILE"
: > "$TXT_FILE"

echo ""
echo "============================================================"
echo " ANALYSE DES CHECKPOINTS — SEED $SEED_FILTER"
echo "============================================================"
echo ""
echo "Analyse lancée : $(date)"
echo "Projet         : $PROJECT_ROOT"
echo "Checkpoints    : $CHECKPOINT_ROOT/*/seed_$SEED_FILTER"
echo ""

analyze_regime() {
    local REGIME="$1"
    local DISPLAY_NAME="$2"
    local METRIC_KEY="$3"   # ex: "eval_hinton/L_CE", "eval_arcd/L_CE", ou "" (student_alone -> eval_loss lui-même)
    local REGIME_DIR="$CHECKPOINT_ROOT/$REGIME/seed_$SEED_FILTER"

    echo ""
    echo "============================================================"
    echo "$DISPLAY_NAME"
    echo "============================================================"
    { echo ""; echo "============================================================"; echo "$DISPLAY_NAME"; echo "============================================================"; echo ""; } >> "$TXT_FILE"

    if [ ! -d "$REGIME_DIR" ]; then
        echo "Aucun dossier trouvé : $REGIME_DIR"
        return
    fi

    mapfile -t CHECKPOINTS < <(find "$REGIME_DIR" -maxdepth 1 -type d -name "checkpoint-*" -printf "%f\n" 2>/dev/null | sort -V)
    if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
        echo "Aucun checkpoint trouvé."
        return
    fi

    # On ne lit que le DERNIER checkpoint : trainer_state.json y contient déjà
    # tout l'historique cumulé (comme observé dans les runs précédents).
    LAST_CHECKPOINT="${CHECKPOINTS[-1]}"
    STATE_FILE="$REGIME_DIR/$LAST_CHECKPOINT/trainer_state.json"

    if [ ! -f "$STATE_FILE" ]; then
        echo "trainer_state.json absent dans $LAST_CHECKPOINT"
        return
    fi

    echo "$LAST_CHECKPOINT (historique complet)"

    python - "$STATE_FILE" "$REGIME" "$LAST_CHECKPOINT" "$CSV_FILE" "$METRIC_KEY" <<'PY'
import json, sys, csv

state_file, regime, checkpoint, csv_file, metric_key = sys.argv[1:6]

with open(state_file) as f:
    state = json.load(f)

by_epoch = {}
for entry in state.get("log_history", []):
    if "eval_loss" not in entry or entry.get("epoch") is None:
        continue
    by_epoch[float(entry["epoch"])] = entry

entries = sorted(by_epoch.items())

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    for epoch, entry in entries:
        eval_loss = entry["eval_loss"]
        comparable = entry.get(metric_key, eval_loss) if metric_key else eval_loss
        writer.writerow([regime, checkpoint, epoch, eval_loss, comparable])
        print(f"  epoch={epoch:6.2f} | eval_loss={eval_loss:.6f} | eval_L_CE_comparable={comparable:.6f}")

if entries:
    best_epoch, best_entry = min(
        entries,
        key=lambda x: (x[1].get(metric_key, x[1]["eval_loss"]) if metric_key else x[1]["eval_loss"])
    )
    best_val = best_entry.get(metric_key, best_entry["eval_loss"]) if metric_key else best_entry["eval_loss"]
    print(f"  MEILLEUR (sur L_CE comparable) = {best_val:.6f} (epoch {best_epoch:.2f})")
PY
    echo ""
}

analyze_regime "student_alone" "BASELINE — STUDENT ALONE" ""
analyze_regime "hinton_kd"     "HINTON — KNOWLEDGE DISTILLATION" "eval_hinton/L_CE"
analyze_regime "multi_teacher_fixed" "MULTI-TEACHER POIDS FIXE — CONTRÔLE" "eval_fixedmt/L_CE"
analyze_regime "arcd"          "ARCD — ADAPTIVE ROBUST CONFIDENCE DISTILLATION" "eval_arcd/L_CE"

echo ""
echo "============================================================"
echo " COMPARAISON — MÊME MÉTRIQUE POUR LES 3 RÉGIMES (eval_L_CE_comparable)"
echo "============================================================"
echo ""

python - "$CSV_FILE" <<'PY'
import csv, sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    data = defaultdict(list)
    for row in csv.DictReader(f):
        try:
            data[row["regime"]].append({
                "checkpoint": row["checkpoint"], "epoch": float(row["epoch"]),
                "eval_loss": float(row["eval_loss"]), "comparable": float(row["eval_L_CE_comparable"]),
            })
        except (ValueError, KeyError):
            pass

print(f"{'RÉGIME':<16} {'EPOCH':>8} {'EVAL_LOSS (brut)':>18} {'EVAL_L_CE (comparable)':>24}")
print("-" * 70)
for regime, entries in data.items():
    if not entries:
        continue
    best = min(entries, key=lambda x: x["comparable"])
    print(f"{regime:<16} {best['epoch']:>8.2f} {best['eval_loss']:>18.6f} {best['comparable']:>24.6f}")

print()
print("ATTENTION : la colonne EVAL_LOSS (brut) n'est PAS comparable entre régimes")
print("(mélanges différents selon la méthode). Seule EVAL_L_CE (comparable) l'est.")
PY

echo ""
echo "CSV : $CSV_FILE"
echo "Analyse terminée."
