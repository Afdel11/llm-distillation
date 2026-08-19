#!/usr/bin/env bash
# scripts/run_curriculum.sh
# ============================
# Entraînement progressif par curriculum : le Student apprend d'abord sur
# les exemples les plus courts (les plus faciles), puis sur un ensemble
# cumulatif de plus en plus large, en repartant à chaque étape des poids
# de l'étape précédente -- jamais from scratch après la première étape.
#
# Mode Teacher EN DIRECT (pas de cache) : le cache existant est indexé sur
# l'ordre du train.json original, incompatible avec les sous-ensembles
# réordonnés du curriculum. Plus lent, mais sans aucun risque de
# désalignement entre logits Teachers et exemples.
#
# Référence : Liu et Zhang (2025), déjà citée au chapitre 5 du mémoire pour
# le diagnostic du mode collapse, proposent précisément un cadre par
# curriculum comme piste corrective.
#
# Usage :
#   bash scripts/run_curriculum.sh
#   bash scripts/run_curriculum.sh 6 8   # 6 étapes, 8 epochs max par étape

set -euo pipefail
cd "$(dirname "$0")/.."

N_STAGES="${1:-4}"
EPOCHS_PER_STAGE="${2:-5}"

echo "================================================================"
echo "  Préparation des fichiers de curriculum ($N_STAGES étapes)"
echo "================================================================"
python scripts/prepare_curriculum.py \
    --input outputs/data/train.json \
    --output_dir outputs/data_curriculum \
    --n_stages "$N_STAGES"

PREV_CKPT=""

for stage in $(seq 1 "$N_STAGES"); do
  echo ""
  echo "================================================================"
  echo "  ÉTAPE $stage / $N_STAGES"
  echo "================================================================"

  CONFIG_PATH="configs/curriculum_stage${stage}.yaml"
  CKPT_DIR="outputs/checkpoints_curriculum/stage${stage}"

  cat > "$CONFIG_PATH" << EOF
data:
  tokenizer_name: "Qwen/Qwen2.5-0.5B-Instruct"
  examples_path: "outputs/data_curriculum/stage${stage}.json"
  val_examples_path: "outputs/data/val.json"
  max_length: 256
  batch_size: 2
  gradient_accumulation_steps: 2

models:
  teacher_names:
    large: "Qwen/Qwen2.5-1.5B-Instruct"
    small: "Qwen/Qwen2.5-0.5B-Instruct"
  student:
    hidden_size: 512
    num_hidden_layers: 4
    num_attention_heads: 8
    num_key_value_heads: 4
    intermediate_size: 2048

training:
  regime: arcd_topk
  device: cuda
  epochs: ${EPOCHS_PER_STAGE}
  early_stopping_patience: 2
  lr: 0.0005
  temperature: 2.0
  consensus_top_k: 10
  seed: 0

output:
  checkpoint_dir: ${CKPT_DIR}
  # Chemins volontairement inexistants -> force le mode Teacher en direct,
  # sans cache (voir l'en-tête de ce script pour la raison).
  teacher_cache_path: outputs/__no_cache_curriculum__.pt
  teacher_cache_val_path: outputs/__no_cache_curriculum_val__.pt
EOF

  if [ -z "$PREV_CKPT" ]; then
    echo "Étape $stage : from scratch (première étape du curriculum)"
    python scripts/train.py --config "$CONFIG_PATH" --restart
  else
    echo "Étape $stage : initialisée depuis les poids de l'étape précédente"
    echo "  ($PREV_CKPT)"
    python scripts/train.py --config "$CONFIG_PATH" --restart --init_from "$PREV_CKPT"
  fi

  PREV_CKPT="${CKPT_DIR}/arcd_topk_seed0_final"

  if [ ! -d "$PREV_CKPT" ]; then
    echo "ERREUR : checkpoint final introuvable à $PREV_CKPT — arrêt du curriculum."
    exit 1
  fi
done

echo ""
echo "================================================================"
echo "  Curriculum terminé après $N_STAGES étapes."
echo "  Checkpoint final : $PREV_CKPT"
echo "================================================================"
echo ""
echo "Pour tester ce modèle :"
echo "  python -i scripts/load_model.py --checkpoint $PREV_CKPT"
