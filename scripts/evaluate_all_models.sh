#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABELS_CSV="${LABELS_CSV:-./metadata/uieb_pseudo_labels.csv}"
DATASET_ROOT="${DATASET_ROOT:-../Underwater_Dataset/UIEB}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"

for model in \
  resnet50_frozen \
  resnet50_finetune \
  convnext_tiny_frozen \
  convnext_tiny_finetune
do
  echo "===== Evaluating ${model} ====="
  python scripts/stage1_eval_assessor.py \
    --labels-csv "$LABELS_CSV" \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint "results/${model}/best_stage1_assessor.pt" \
    --split test \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output-dir "results/${model}/eval"
done

echo "All four models evaluated."
