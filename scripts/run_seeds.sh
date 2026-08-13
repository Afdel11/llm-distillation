#!/usr/bin/env bash
# scripts/run_seeds.sh
# ======================
# Lance les 3 régimes (baseline, hinton, arcd) sur plusieurs seeds, pour
# pouvoir présenter une moyenne ± écart-type plutôt qu'un seul run par
# régime (un seul run ne permet pas de distinguer un vrai effet d'ARCD
# d'un simple coup de chance sur l'initialisation).
#
# Le cache Teacher (outputs/teacher_cache*.pt) ne dépend pas du seed
# (Teachers gelés) -> construit UNE SEULE FOIS, réutilisé pour tous les
# seeds et tous les régimes.
#
# Chaque run est TOUJOURS lancé avec --restart : dans un balayage multi-
# seeds, on ne veut jamais reprendre accidentellement le checkpoint d'un
# autre run (voir l'incident de reprise inter-versions déjà rencontré).
#
# Usage :
#   bash scripts/run_seeds.sh                # seeds 0 1 2 par défaut
#   bash scripts/run_seeds.sh 0 1 2 3 4       # seeds personnalisés
#   bash scripts/run_seeds.sh --skip-cache 0 1 2

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_CACHE=false
SEEDS=()
for arg in "$@"; do
  if [[ "$arg" == "--skip-cache" ]]; then
    SKIP_CACHE=true
  else
    SEEDS+=("$arg")
  fi
done
if [ "${#SEEDS[@]}" -eq 0 ]; then
  SEEDS=(0 1 2)
fi

section () {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

if [[ "$SKIP_CACHE" == "false" ]]; then
  section "Cache Teacher (une seule fois, indépendant du seed)"
  python scripts/build_teacher_cache.py --config configs/arcd.yaml
else
  echo "Cache ignoré (--skip-cache)."
fi

for seed in "${SEEDS[@]}"; do
  section "SEED $seed — Baseline (student_alone)"
  python scripts/train.py --config configs/baseline.yaml --seed "$seed" --restart

  section "SEED $seed — Hinton KD"
  python scripts/train.py --config configs/hinton.yaml --seed "$seed" --restart

  section "SEED $seed — ARCD"
  python scripts/train.py --config configs/arcd.yaml --seed "$seed" --restart
done

section "Terminé — ${#SEEDS[@]} seed(s) x 3 régimes. Lance scripts/analyze_seeds.sh pour l'agrégation."
