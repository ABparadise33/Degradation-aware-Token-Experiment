#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABELS_CSV="${LABELS_CSV:-./metadata/uieb_pseudo_labels.csv}"
DATASET_ROOT="${DATASET_ROOT:-../Underwater_Dataset/UIEB}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_selfsup/convnext_tiny_slots4}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SLOTS="${NUM_SLOTS:-4}"

python scripts/stage1_train_self_supervised_deg.py \
  --labels-csv "$LABELS_CSV" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --backbone convnext_tiny \
  --latent-dim 128 \
  --num-slots "$NUM_SLOTS" \
  --num-heads 4 \
  --decoder-layers 1 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 1e-5 \
  --num-workers "$NUM_WORKERS" \
  --lambda-same 1.0 \
  --lambda-different 0.5 \
  --lambda-order 0.5 \
  --lambda-pair 0.5 \
  --lambda-variance 0.01 \
  --lambda-covariance 0.01 \
  --lambda-slot-diversity 0.01 \
  --device "$DEVICE"

echo "Self-supervised training completed."
echo "Run scripts/run_self_supervised_evaluation.sh next."
