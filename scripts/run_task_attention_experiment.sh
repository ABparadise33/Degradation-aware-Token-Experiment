#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABELS_CSV="${LABELS_CSV:-./metadata/uieb_pseudo_labels.csv}"
DATASET_ROOT="${DATASET_ROOT:-../Underwater_Dataset/UIEB}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_task_attention/convnext_tiny_finetune}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LAMBDA_CONTRAST="${LAMBDA_CONTRAST:-0.0}"
LAMBDA_ORDER="${LAMBDA_ORDER:-0.0}"
LAMBDA_BLUR_SYNTHETIC="${LAMBDA_BLUR_SYNTHETIC:-0.0}"
LAMBDA_BLUR_ORDER="${LAMBDA_BLUR_ORDER:-0.0}"
LAMBDA_TASK_DIVERSITY="${LAMBDA_TASK_DIVERSITY:-0.0}"
LAMBDA_ATTENTION_DIVERSITY="${LAMBDA_ATTENTION_DIVERSITY:-0.0}"

python scripts/stage1_train_assessor.py \
  --labels-csv "$LABELS_CSV" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --backbone convnext_tiny \
  --architecture task_attention \
  --latent-dim 128 \
  --num-heads 4 \
  --decoder-layers 1 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 1e-5 \
  --num-workers "$NUM_WORKERS" \
  --lambda-contrast "$LAMBDA_CONTRAST" \
  --lambda-order "$LAMBDA_ORDER" \
  --lambda-blur-synthetic "$LAMBDA_BLUR_SYNTHETIC" \
  --lambda-blur-order "$LAMBDA_BLUR_ORDER" \
  --lambda-task-diversity "$LAMBDA_TASK_DIVERSITY" \
  --lambda-attention-diversity "$LAMBDA_ATTENTION_DIVERSITY" \
  --device "$DEVICE"

python scripts/stage1_eval_assessor.py \
  --labels-csv "$LABELS_CSV" \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$OUTPUT_DIR/best_stage1_assessor.pt" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR/eval"
