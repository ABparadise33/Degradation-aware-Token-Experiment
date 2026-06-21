#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-./results_task_attention/convnext_tiny_finetune}"
CHECKPOINT="${CHECKPOINT:-${EXPERIMENT_DIR}/best_stage1_assessor.pt}"
FEATURES="${FEATURES:-${EXPERIMENT_DIR}/eval/features_test.npz}"
SELECTION_CSV="${SELECTION_CSV:-${EXPERIMENT_DIR}/representative_images.csv}"
DEVICE="${DEVICE:-mps}"

python scripts/stage1_validate_task_tokens.py \
  --features "$FEATURES" \
  --output-dir "${EXPERIMENT_DIR}/eval/task_token_validation"

python scripts/stage1_attention_faithfulness.py \
  --checkpoint "$CHECKPOINT" \
  --selection-csv "$SELECTION_CSV" \
  --output-dir "${EXPERIMENT_DIR}/eval/attention_faithfulness" \
  --fraction 0.1 \
  --random-repeats 5 \
  --device "$DEVICE"

python scripts/stage1_evaluate_synthetic_blur.py \
  --checkpoint "$CHECKPOINT" \
  --selection-csv "$SELECTION_CSV" \
  --output-dir "${EXPERIMENT_DIR}/eval/synthetic_blur" \
  --levels 7 \
  --device "$DEVICE"

echo "Task-aware diagnostics completed."
