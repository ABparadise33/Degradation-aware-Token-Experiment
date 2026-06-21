#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABELS_CSV="${LABELS_CSV:-./metadata/uieb_pseudo_labels.csv}"
DATASET_ROOT="${DATASET_ROOT:-../Underwater_Dataset/UIEB}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-./results_selfsup/convnext_tiny_slots4}"
CHECKPOINT="${CHECKPOINT:-${EXPERIMENT_DIR}/best_self_supervised_encoder.pt}"
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_DIR="${EVAL_DIR:-${EXPERIMENT_DIR}/eval}"
SELECTION_CSV="${SELECTION_CSV:-}"

python scripts/stage1_eval_self_supervised_deg.py \
  --checkpoint "$CHECKPOINT" \
  --labels-csv "$LABELS_CSV" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$EVAL_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-contents 32 \
  --num-recipes 8 \
  --num-severity-directions 4 \
  --severity-steps 7 \
  --device "$DEVICE"

VIS_ARGS=(
  --synthetic-features "$EVAL_DIR/synthetic_retrieval_features.npz"
  --severity-trajectories "$EVAL_DIR/severity_trajectories.npz"
  --output-dir "$EXPERIMENT_DIR/vis"
  --device "$DEVICE"
)
if [[ -n "$SELECTION_CSV" ]]; then
  VIS_ARGS+=(--checkpoint "$CHECKPOINT" --selection-csv "$SELECTION_CSV")
fi
python scripts/stage1_visualize_self_supervised.py "${VIS_ARGS[@]}"

echo "Self-supervised evaluation and visualization completed."
