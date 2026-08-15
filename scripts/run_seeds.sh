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
#   bash scripts/run_seeds.sh --regimes arcd_diverse 0 1 2
#       -> variante ARCD avec Teacher diversifié (Qwen2.5-Coder), volontairement
#          ABSENTE de la liste de régimes par défaut -> jamais lancée par accident.
#   bash scripts/run_seeds.sh --regimes hinton,arcd 0 1 2
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
  REGIMES=(baseline hinton multi_teacher_fixed arcd)   # arcd_diverse volontairement ABSENT du défaut
else
  IFS=',' read -ra REGIMES <<< "$REGIMES_ARG"
fi

declare -A CONFIG_FILE=(
  [baseline]="configs/baseline.yaml"
  [hinton]="configs/hinton.yaml"
  [multi_teacher_fixed]="configs/multi_teacher_fixed.yaml"
  [arcd]="configs/arcd.yaml"
  [arcd_diverse]="configs/arcd_diverse.yaml"
)
declare -A DISPLAY_NAME=(
  [baseline]="Baseline (student_alone)"
  [hinton]="Hinton KD"
  [multi_teacher_fixed]="Multi-Teacher poids fixe (contrôle)"
  [arcd]="ARCD"
  [arcd_diverse]="ARCD (Teacher diversifié — Coder)"
)

section () {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

if [[ "$SKIP_CACHE" == "false" ]]; then
  # Construit le cache Teacher UNIQUEMENT pour les configs réellement utilisées
  # par les régimes sélectionnés : "arcd"/"multi_teacher_fixed" partagent le
  # cache standard (configs/arcd.yaml) ; "arcd_diverse" a son PROPRE cache
  # (Teacher différent -> logits différents, voir configs/arcd_diverse.yaml).
  NEEDS_STANDARD_CACHE=false
  NEEDS_DIVERSE_CACHE=false
  for regime in "${REGIMES[@]}"; do
    case "$regime" in
      arcd|multi_teacher_fixed) NEEDS_STANDARD_CACHE=true ;;
      arcd_diverse) NEEDS_DIVERSE_CACHE=true ;;
    esac
  done

  if [[ "$NEEDS_STANDARD_CACHE" == "true" ]]; then
    section "Cache Teacher standard (partagé arcd + multi_teacher_fixed)"
    python scripts/build_teacher_cache.py --config configs/arcd.yaml
  fi
  if [[ "$NEEDS_DIVERSE_CACHE" == "true" ]]; then
    section "Cache Teacher diversifié (arcd_diverse uniquement)"
    python scripts/build_teacher_cache.py --config configs/arcd_diverse.yaml
  fi
  if [[ "$NEEDS_STANDARD_CACHE" == "false" && "$NEEDS_DIVERSE_CACHE" == "false" ]]; then
    echo "Aucun régime sélectionné n'a besoin d'un cache Teacher (baseline/hinton seuls)."
  fi
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
