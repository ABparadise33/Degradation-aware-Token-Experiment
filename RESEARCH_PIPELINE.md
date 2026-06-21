# Degradation-aware Token Research Pipeline

這份文件把 baseline、V2 token、task-aware decoder、spatial analysis、synthetic learning、SAM region pooling 與 Stage 2 condition 串成一條可執行流程。

> 本次只完成程式碼、commands 與 smoke tests；沒有代替使用者執行正式 training。

## 0. 保留既有 baseline

既有 `results/` 是 V1 baseline：

```text
feature -> token_head -> z_deg
feature -> score_head -> scores
```

它保留作為論文比較基準。`scripts/run_all_mps_experiments.sh` 已明確加入 `--legacy-direct-score`，不會和新版混淆。

## 1. V2 supervised degradation token

新版預設架構：

```text
feature -> token_head -> z_deg -> score_head -> scores
```

五個 canonical scores：

```text
s_color
s_blur
s_contrast
s_visibility_proxy
q_quality
```

舊 metadata 中的 `s_haze` 會自動映射為 `s_visibility_proxy`，不需重算 CSV。它仍然是 weak visibility proxy，不是物理 haze ground truth。

### 執行四組 V2 training

```bash
cd /Users/ed/Research/Underwater_Stage1
conda activate FlowIE

DEVICE=mps \
OUTPUT_ROOT=./results_v2 \
EPOCHS=20 \
BATCH_SIZE=16 \
NUM_WORKERS=4 \
./scripts/run_all_v2_token_experiments.sh
```

若 Terminal 中的 PyTorch 無法使用 MPS：

```bash
DEVICE=cpu NUM_WORKERS=0 BATCH_SIZE=8 ./scripts/run_all_v2_token_experiments.sh
```

## 2. V2 evaluation

四模型 test、PCA/t-SNE、feature quantitative validation：

```bash
DEVICE=mps OUTPUT_ROOT=./results_v2 ./scripts/evaluate_all_v2_models.sh
```

每組輸出：

```text
eval/metrics_test.csv
eval/predictions_test.csv
eval/features_test.npz
eval/feature_metrics_test.csv
vis/z_deg_pca_target.png
vis/z_deg_tsne_target.png
vis/z_deg_pca_prediction.png
vis/z_deg_tsne_prediction.png
```

圖的顏色表示 degradation type，marker 表示 raw/reference。量化指標包含：

```text
intra-class distance
inter-class centroid distance
Silhouette score
Davies-Bouldin index
kNN accuracy
linear classification probe
five score regression probes
```

## 3. Representative images、spatial features、Grad-CAM

對最佳 V2 模型執行完整 qualitative pipeline：

```bash
DEVICE=mps \
OUTPUT_ROOT=./results_v2 \
MODEL_NAME=convnext_tiny_finetune \
PER_CATEGORY=5 \
./scripts/run_v2_post_training_pipeline.sh
```

它會挑選：

```text
color cast
blur
low contrast
low visibility
mixed degradation
reference
```

並輸出每個 backbone stage 的 mean absolute activation、前 8 channels，以及五個 score 的 Grad-CAM。

## 4. SFIQA-style task-aware decoder

架構：

```text
spatial feature map
  -> image tokens
  -> learnable 5 task tokens
  -> task-token self attention
  -> cross attention
  -> per-task regression heads
```

輸出：

```text
scores              [B, 5]
task_tokens         [B, 5, D]
z_deg               [B, D]
attention_maps      [B, 5, H, W]
spatial_feature     [B, C, H, W]
```

Training command：

```bash
DEVICE=mps \
OUTPUT_DIR=./results_task_attention/convnext_tiny_finetune \
./scripts/run_task_attention_experiment.sh
```

加入 synthetic representation learning：

```bash
DEVICE=mps \
LAMBDA_CONTRAST=0.1 \
LAMBDA_ORDER=0.1 \
./scripts/run_task_attention_experiment.sh
```

Task attention visualization：

```bash
python scripts/stage1_select_representative_images.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --split test \
  --per-category 5 \
  --output ./results_task_attention/convnext_tiny_finetune/representative_images.csv

python scripts/stage1_visualize_task_attention.py \
  --checkpoint ./results_task_attention/convnext_tiny_finetune/best_stage1_assessor.pt \
  --selection-csv ./results_task_attention/convnext_tiny_finetune/representative_images.csv \
  --output-dir ./results_task_attention/convnext_tiny_finetune/task_attention \
  --device mps
```

## 5. Visibility alternatives

目前主要 target 仍是：

```text
0.5 * inverse contrast
+ 0.3 * low saturation
+ 0.2 * flat brightness
```

DCP-style transmission proxy 可獨立檢查：

```bash
python scripts/stage1_visibility_proxy.py \
  --image ../Underwater_Dataset/UIEB/raw-890/392_img_.png \
  --output /tmp/392_transmission.png
```

這仍是 proxy，不能宣稱為真實水下 transmission ground truth。

## Task-token specialization、attention faithfulness、blur diagnostics

三項診斷可一次執行：

```bash
DEVICE=mps \
EXPERIMENT_DIR=./results_task_attention/convnext_tiny_finetune \
./scripts/run_task_aware_diagnostics.sh
```

### Per-task token validation

輸出：

```text
eval/task_token_validation/task_token_probe_matrix.csv
eval/task_token_validation/task_token_summary.csv
eval/task_token_validation/task_token_cosine_similarity.csv
eval/task_token_validation/task_token_probe_mae.png
eval/task_token_validation/task_token_cosine_similarity.png
```

理想狀況是 probe MAE matrix 的對角線最低，且不同 task token 的 cosine similarity 明顯低於 1。

### Attention faithfulness

對 attention top 10%、bottom 10% 與 random 10% 區域分別遮罩，再重新 forward：

```text
top absolute score change > random > bottom
```

輸出：

```text
eval/attention_faithfulness/attention_faithfulness_details.csv
eval/attention_faithfulness/attention_faithfulness_summary.csv
```

`top_selectivity > 0` 表示遮罩該 task 的 top-attention region，對自己的 score 影響大於其他四項 score。

### Synthetic blur response

對 Gaussian blur 與 motion blur severity `0→1` 檢查 predicted `s_blur`：

```text
Spearman(severity, prediction) 越接近 1 越好
increasing_step_fraction 越接近 1 越好
prediction_range 不應接近 0
```

輸出：

```text
eval/synthetic_blur/synthetic_blur_details.csv
eval/synthetic_blur/synthetic_blur_summary.csv
```

### 下一版 blur/disentanglement training

建議輸出到新的 experiment directory，不覆蓋現有 task-aware baseline：

```bash
DEVICE=mps \
OUTPUT_DIR=./results_task_attention_blur_diverse/convnext_tiny_finetune \
LAMBDA_BLUR_SYNTHETIC=0.1 \
LAMBDA_BLUR_ORDER=0.1 \
LAMBDA_TASK_DIVERSITY=0.01 \
LAMBDA_ATTENTION_DIVERSITY=0.01 \
./scripts/run_task_attention_experiment.sh
```

其中：

```text
L_blur_synthetic：要求 severe/mild predicted blur gap 接近已知 synthetic severity gap
L_blur_order：要求 severe blur score 高於 mild blur
L_task_diversity：降低五個 task token 的 cosine collapse
L_attention_diversity：降低五張 attention map 完全相同的情況
```

## 6. Synthetic degradation learning

`stage1/synthetic.py` 支援：

```text
color cast
Gaussian blur
contrast reduction
visibility/backscatter-like degradation
```

Optional losses：

```text
InfoNCE: 同 source、同 degradation 的 mild/severe token 對齊
severity order: severe 必須比 mild 距離 clean token 更遠
```

由 training flags 控制，預設權重為 0，不影響 baseline。

## 7. SAM masks 與 region pooling

安裝 optional SAM：

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

產生 masks：

```bash
python scripts/stage1_generate_sam_masks.py \
  --image ../Underwater_Dataset/UIEB/raw-890/392_img_.png \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device mps \
  --output /tmp/392_sam_masks.npz
```

Region inference：

```bash
python scripts/stage1_region_inference.py \
  --checkpoint ./results_task_attention/convnext_tiny_finetune/best_stage1_assessor.pt \
  --image ../Underwater_Dataset/UIEB/raw-890/392_img_.png \
  --masks-npz /tmp/392_sam_masks.npz \
  --output-dir /tmp/392_stage1_conditions \
  --device mps
```

輸出：

```text
stage1_conditions.npz
s_color_region_map.png
s_blur_region_map.png
s_contrast_region_map.png
s_visibility_proxy_region_map.png
```

NPZ 內含 global scores/token、region scores/token/maps、SAM masks，以及 task-aware 模型的 task tokens/attention。

## 8. 接入 Stage 2

`stage1/conditioning.py` 提供：

```python
conditions = load_stage1_conditions("stage1_conditions.npz", device)
stage2_input = build_spatial_condition(
    underwater_image,
    conditions["region_maps"],
    conditions.get("masks"),
)
```

最簡單的 Stage 2 輸入 channel：

```text
RGB image (3)
+ four degradation maps (4)
+ SAM coverage (1, optional)
= 7 or 8 channels
```

Global `z_deg` 或 `task_tokens` 可另外透過 FiLM、cross-attention 或 ControlNet condition 注入 FlowIE。

## 9. 最終 comparison

收集不同 experiment metrics：

```bash
python scripts/compare_experiment_metrics.py \
  --glob 'results/*/eval/metrics_test.csv' \
  --glob 'results_v2/*/eval/metrics_test.csv' \
  --glob 'results_task_attention/*/eval/metrics_test.csv' \
  --output comparison/all_experiments.csv
```

最終 ablation：

```text
V1 direct-score baseline
V2 supervised global z_deg
task-aware tokens
task-aware tokens + synthetic losses
Stage 2 baseline FlowIE
FlowIE + global z_deg
FlowIE + task tokens
FlowIE + region degradation maps
```

## 10. Non-training verification

```bash
python scripts/smoke_test_pipeline.py
python -m compileall -q stage1 scripts
bash -n scripts/*.sh
```
