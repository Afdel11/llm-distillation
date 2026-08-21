#!/usr/bin/env bash
# scripts/dit/run_dit_finetuning.sh
# ====================================
# Fine-tuning spécialisé DIT : repart des checkpoints généraux "patient"
# (les plus solides, post-correctif du décalage causal) et les affine sur
# le corpus DIT (outputs/data_dit/). Ordre : student_alone (baseline) ->
# hinton_kd -> multi_teacher_fixed -> arcd_topk (si le temps le permet).
#
# Usage :
#   bash scripts/dit/run_dit_finetuning.sh                # les 4 régimes
#   bash scripts/dit/run_dit_finetuning.sh student_alone hinton_kd   # un sous-ensemble

set -euo pipefail
cd "$(dirname "$0")/../.."

REGIMES=("$@")
if [ ${#REGIMES[@]} -eq 0 ]; then
  REGIMES=(student_alone hinton_kd multi_teacher_fixed arcd)
fi

echo "================================================================"
echo "  Préparation du dataset DIT"
echo "================================================================"
python scripts/dit/generate_dit_dataset.py

declare -A CONFIG_FILE=(
  [student_alone]="configs/dit/student_alone_dit.yaml"
  [hinton_kd]="configs/dit/hinton_dit.yaml"
  [multi_teacher_fixed]="configs/dit/multi_teacher_fixed_dit.yaml"
  [arcd]="configs/dit/arcd_dit.yaml"
)

declare -A BASE_CHECKPOINT=(
  [student_alone]="outputs/checkpoints_patient/student_alone_seed0_final"
  [hinton_kd]="outputs/checkpoints_patient/hinton_kd_seed0_final"
  [multi_teacher_fixed]="outputs/checkpoints_patient/multi_teacher_fixed_seed0_final"
  [arcd]="outputs/checkpoints_patient/arcd_topk_seed0_final"
)

for regime in "${REGIMES[@]}"; do
  config="${CONFIG_FILE[$regime]:-}"
  base_ckpt="${BASE_CHECKPOINT[$regime]:-}"

  if [ -z "$config" ]; then
    echo "Régime inconnu: $regime (attendu: student_alone, hinton_kd, multi_teacher_fixed, arcd) — ignoré."
    continue
  fi
  if [ ! -d "$base_ckpt" ]; then
    echo "ERREUR : checkpoint de base introuvable pour $regime -> $base_ckpt"
    echo "Vérifie que outputs/checkpoints_patient/ contient bien ce checkpoint avant de continuer."
    exit 1
  fi

  echo ""
  echo "================================================================"
  echo "  Fine-tuning DIT : $regime"
  echo "  Depuis : $base_ckpt"
  echo "================================================================"
  python scripts/train.py --config "$config" --restart --init_from "$base_ckpt"
done

echo ""
echo "================================================================"
echo "  Fine-tuning DIT terminé pour : ${REGIMES[*]}"
echo "================================================================"
echo ""
echo "Pour tester un modèle :"
echo "  python -i scripts/load_model.py --checkpoint outputs/checkpoints_dit/<regime>_seed0_final"
