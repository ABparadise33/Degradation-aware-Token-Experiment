#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABELS_CSV="${LABELS_CSV:-./metadata/uieb_pseudo_labels.csv}"
DATASET_ROOT="${DATASET_ROOT:-../Underwater_Dataset/UIEB}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-mps}"

# Let unsupported MPS operations fall back to CPU instead of terminating a run.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

run_experiment() {
  local name="$1"
  local backbone="$2"
  local learning_rate="$3"
  local freeze_flag="$4"
  local output_dir="./results/${name}"

  echo "===== Training ${name} ====="
  python scripts/stage1_train_assessor.py \
    --labels-csv "$LABELS_CSV" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$output_dir" \
    --backbone "$backbone" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$learning_rate" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    $freeze_flag

  echo "===== Testing ${name} ====="
  python scripts/stage1_eval_assessor.py \
    --labels-csv "$LABELS_CSV" \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint "$output_dir/best_stage1_assessor.pt" \
    --split test \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output-dir "$output_dir/eval"
}

run_experiment "resnet50_frozen" "resnet50" "1e-4" "--freeze-backbone"
run_experiment "resnet50_finetune" "resnet50" "1e-5" ""
run_experiment "convnext_tiny_frozen" "convnext_tiny" "1e-4" "--freeze-backbone"
run_experiment "convnext_tiny_finetune" "convnext_tiny" "1e-5" ""

echo "All four MPS experiments completed."
