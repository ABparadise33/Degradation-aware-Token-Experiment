#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABELS_CSV="${LABELS_CSV:-./metadata/uieb_pseudo_labels.csv}"
DATASET_ROOT="${DATASET_ROOT:-../Underwater_Dataset/UIEB}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results_v2}"
MODEL_NAME="${MODEL_NAME:-convnext_tiny_finetune}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_ROOT}/${MODEL_NAME}/best_stage1_assessor.pt}"
DEVICE="${DEVICE:-mps}"
PER_CATEGORY="${PER_CATEGORY:-5}"
SELECTION_CSV="${SELECTION_CSV:-${OUTPUT_ROOT}/${MODEL_NAME}/representative_images.csv}"

python scripts/stage1_select_representative_images.py \
  --labels-csv "$LABELS_CSV" \
  --dataset-root "$DATASET_ROOT" \
  --split test \
  --per-category "$PER_CATEGORY" \
  --output "$SELECTION_CSV"

python scripts/stage1_visualize_features.py \
  --features "${OUTPUT_ROOT}/${MODEL_NAME}/eval/features_test.npz" \
  --output-dir "${OUTPUT_ROOT}/${MODEL_NAME}/vis"

python scripts/stage1_validate_features.py \
  --features "${OUTPUT_ROOT}/${MODEL_NAME}/eval/features_test.npz" \
  --output "${OUTPUT_ROOT}/${MODEL_NAME}/eval/feature_metrics_test.csv"

python scripts/stage1_visualize_spatial_features.py \
  --checkpoint "$CHECKPOINT" \
  --selection-csv "$SELECTION_CSV" \
  --output-dir "${OUTPUT_ROOT}/${MODEL_NAME}/spatial_features" \
  --device "$DEVICE"

python scripts/stage1_batch_gradcam.py \
  --checkpoint "$CHECKPOINT" \
  --selection-csv "$SELECTION_CSV" \
  --output-dir "${OUTPUT_ROOT}/${MODEL_NAME}/gradcam_batch" \
  --device "$DEVICE"

echo "If this is a task_attention checkpoint, also run:"
echo "python scripts/stage1_visualize_task_attention.py --checkpoint \"$CHECKPOINT\" --selection-csv \"$SELECTION_CSV\" --output-dir \"${OUTPUT_ROOT}/${MODEL_NAME}/task_attention\" --device \"$DEVICE\""
