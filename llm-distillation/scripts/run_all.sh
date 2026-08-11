#!/usr/bin/env bash
# scripts/run_all.sh
# ====================
# Lance, dans l'ordre, tout ce qu'il faut pour comparer les 3 régimes :
#   1. Construction du cache Teacher (train + val) — une seule fois, réutilisé partout.
#   2. Baseline (student_alone)
#   3. Hinton KD
#   4. ARCD
#
# S'arrête au premier échec (set -e) plutôt que d'enchaîner sur un état
# incohérent. Chaque étape est clairement délimitée dans la sortie.
#
# Usage :
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --skip-cache   # si le cache existe déjà et est à jour

set -euo pipefail

cd "$(dirname "$0")/.."  # se placer à la racine du projet, peu importe d'où on lance le script

SKIP_CACHE=false
for arg in "$@"; do
  if [[ "$arg" == "--skip-cache" ]]; then
    SKIP_CACHE=true
  fi
done

section () {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

if [[ "$SKIP_CACHE" == "false" ]]; then
  section "1/4 — Construction du cache Teacher (train + val)"
  python scripts/build_teacher_cache.py --config configs/arcd.yaml
else
  echo "Cache ignoré (--skip-cache) — vérifie que outputs/teacher_cache.pt ET "
  echo "outputs/teacher_cache_val.pt existent déjà et correspondent au dataset actuel."
fi

section "2/4 — Baseline (student_alone)"
python scripts/train.py --config configs/baseline.yaml

section "3/4 — Hinton KD"
python scripts/train.py --config configs/hinton.yaml

section "4/4 — ARCD"
python scripts/train.py --config configs/arcd.yaml

section "Terminé — checkpoints dans outputs/checkpoints/{student_alone,hinton_kd,arcd}_final"
