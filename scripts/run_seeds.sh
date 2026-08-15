#!/usr/bin/env bash
# scripts/run_seeds.sh
# ======================
# Lance un ou plusieurs régimes sur plusieurs seeds, pour pouvoir présenter
# une moyenne ± écart-type plutôt qu'un seul run par régime.
#
# Le cache Teacher (outputs/teacher_cache*.pt) ne dépend pas du seed
# (Teachers gelés) -> construit UNE SEULE FOIS, réutilisé pour tous les
# seeds ET pour multi_teacher_fixed (qui partage exactement la même cible
# de consensus qu'arcd).
#
# Chaque run est TOUJOURS lancé avec --restart : dans un balayage multi-
# seeds, on ne veut jamais reprendre accidentellement le checkpoint d'un
# autre run.
#
# Usage :
#   bash scripts/run_seeds.sh                          # 4 régimes, seeds 0 1 2
#   bash scripts/run_seeds.sh 0 1 2 3 4                 # seeds personnalisés
#   bash scripts/run_seeds.sh --skip-cache 0 1 2
#   bash scripts/run_seeds.sh --regimes multi_teacher_fixed 0 1 2
#       -> ne lance QUE le nouveau régime de contrôle, sans retoucher aux
#          baseline/hinton/arcd déjà entraînés pour ces mêmes seeds.
#   bash scripts/run_seeds.sh --regimes hinton_kd,arcd 0 1 2
#       -> plusieurs régimes précis, séparés par des virgules.

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_CACHE=false
SEEDS=()
REGIMES_ARG=""
NEXT_IS_REGIMES=false

for arg in "$@"; do
  if [[ "$NEXT_IS_REGIMES" == "true" ]]; then
    REGIMES_ARG="$arg"
    NEXT_IS_REGIMES=false
  elif [[ "$arg" == "--skip-cache" ]]; then
    SKIP_CACHE=true
  elif [[ "$arg" == "--regimes" ]]; then
    NEXT_IS_REGIMES=true
  else
    SEEDS+=("$arg")
  fi
done

if [ "${#SEEDS[@]}" -eq 0 ]; then
  SEEDS=(0 1 2)
fi

if [[ -z "$REGIMES_ARG" ]]; then
  REGIMES=(baseline hinton multi_teacher_fixed arcd)
else
  IFS=',' read -ra REGIMES <<< "$REGIMES_ARG"
fi

declare -A CONFIG_FILE=(
  [baseline]="configs/baseline.yaml"
  [hinton]="configs/hinton.yaml"
  [multi_teacher_fixed]="configs/multi_teacher_fixed.yaml"
  [arcd]="configs/arcd.yaml"
)
declare -A DISPLAY_NAME=(
  [baseline]="Baseline (student_alone)"
  [hinton]="Hinton KD"
  [multi_teacher_fixed]="Multi-Teacher poids fixe (contrôle)"
  [arcd]="ARCD"
)

section () {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

if [[ "$SKIP_CACHE" == "false" ]]; then
  section "Cache Teacher (une seule fois, partagé arcd + multi_teacher_fixed)"
  python scripts/build_teacher_cache.py --config configs/arcd.yaml
else
  echo "Cache ignoré (--skip-cache)."
fi

echo ""
echo "Régimes sélectionnés : ${REGIMES[*]}"
echo "Seeds sélectionnés   : ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  for regime in "${REGIMES[@]}"; do
    cfg="${CONFIG_FILE[$regime]:-}"
    if [[ -z "$cfg" ]]; then
      echo "Régime inconnu: $regime (attendu: baseline, hinton, multi_teacher_fixed, arcd) — ignoré."
      continue
    fi
    section "SEED $seed — ${DISPLAY_NAME[$regime]}"
    python scripts/train.py --config "$cfg" --seed "$seed" --restart
  done
done

section "Terminé — ${#SEEDS[@]} seed(s) x ${#REGIMES[@]} régime(s). Lance scripts/analyze_seeds.py pour l'agrégation."
