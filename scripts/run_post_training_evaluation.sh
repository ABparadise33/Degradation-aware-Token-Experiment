#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"
GRADCAM_IMAGE="${GRADCAM_IMAGE:-../Underwater_Dataset/UIEB/raw-890/392_img_.png}"
BEST_MODEL="convnext_tiny_finetune"

for model in \
  resnet50_frozen \
  resnet50_finetune \
  convnext_tiny_frozen \
  convnext_tiny_finetune
do
  echo "===== Visualizing ${model} z_deg ====="
  python scripts/stage1_visualize_features.py \
    --features "results/${model}/eval/features_test.npz" \
    --output-dir "results/${model}/vis"
done

for score in s_color s_blur s_contrast s_haze q_quality
do
  echo "===== Grad-CAM ${score} ====="
  python scripts/stage1_gradcam.py \
    --checkpoint "results/${BEST_MODEL}/best_stage1_assessor.pt" \
    --image "$GRADCAM_IMAGE" \
    --score "$score" \
    --device "$DEVICE" \
    --output "results/${BEST_MODEL}/gradcam/392_${score}.png"
done

echo "Post-training evaluation completed."
