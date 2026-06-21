# Self-supervised Degradation Representation Pipeline

這個實驗把現有 V2 與 Task-aware models 保留為 supervised baselines，另外新增不使用五個 pseudo-score 作為主訓練目標的 degradation representation learner。

> Repository 只提供 code、commands 與 evaluation pipeline。正式 training 由使用者自行執行。

## 1. Research question

現有 baselines 直接使用：

```text
s_color / s_blur / s_contrast / s_visibility_proxy / q_quality
```

訓練 regression heads。新模型改為學習：

```text
general z_deg
K anonymous degradation slots
generic magnitude m_deg
```

五個 pseudo scores 僅保留給 frozen linear probe evaluation，不參與主模型 training loss。

## 2. Architecture

```text
image
  -> ConvNeXt-Tiny spatial feature map
  -> 1x1 projection
  -> image tokens
  -> K anonymous slot queries
  -> slot self-attention
  -> cross-attention over image tokens
  -> slot_tokens [B, K, D]
  -> mean + LayerNorm
  -> z_deg [B, D]
  -> generic magnitude head
  -> m_deg [B, 1]
```

模型實作：`stage1/self_supervised_model.py`

輸出：

| Output | Shape | 用途 |
|---|---|---|
| `z_deg` | `[B, D]` | 全域 degradation representation |
| `slot_tokens` | `[B, K, D]` | anonymous degradation slots |
| `slot_attention` | `[B, K, H, W]` | slot spatial attention |
| `m_deg` | `[B, 1]` | generic degradation magnitude |
| `spatial_feature` | `[B, C, H, W]` | Stage 2 / region analysis |

## 3. Mixed degradation generation

`stage1/mixed_degradation.py` 以連續 recipe 產生混合退化：

```text
[color, blur, contrast, visibility, noise]
```

recipe 只用來建立 positive/negative/order pairs，不會作為 degradation type classification label。

每個 batch 建立：

1. Same degradation, different content。
2. Same content, different degradation。
3. Same direction, mild vs severe。
4. Real UIEB raw/reference pair。

## 4. Objective

```text
L =
  λ_same       L_InfoNCE
+ λ_different  L_separation
+ λ_order      L_severity_order
+ λ_pair       L_raw_reference_order
+ λ_variance   L_variance
+ λ_covariance L_covariance
+ λ_slot       L_slot_diversity
```

- `L_InfoNCE`：同 recipe、不同內容的 representation 對齊。
- `L_separation`：同內容、不同 recipe 的 representation 至少相隔指定 margin。
- `L_severity_order`：severe 必須比 mild 距離 clean representation 更遠。
- `L_raw_reference_order`：`m_deg(raw) > m_deg(reference)`。
- variance/covariance：避免所有 representation collapse 或 dimensions 高度冗餘。
- slot diversity：避免 anonymous slots 完全相同。

## 5. Training command

```bash
cd /Users/ed/Research/Underwater_Stage1
conda activate FlowIE

DEVICE=mps \
OUTPUT_DIR=./results_selfsup/convnext_tiny_slots4 \
EPOCHS=20 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
NUM_SLOTS=4 \
./scripts/run_self_supervised_experiment.sh
```

這個 training step 每次會合併 forward 七組 views，因此記憶體使用量高於原本 assessor。若 MPS out-of-memory：

```bash
DEVICE=mps BATCH_SIZE=4 NUM_WORKERS=2 ./scripts/run_self_supervised_experiment.sh
```

直接調整 loss：

```bash
python scripts/stage1_train_self_supervised_deg.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results_selfsup/convnext_tiny_slots8 \
  --backbone convnext_tiny \
  --num-slots 8 \
  --epochs 20 \
  --batch-size 4 \
  --lambda-same 1.0 \
  --lambda-different 0.5 \
  --lambda-order 0.5 \
  --lambda-pair 0.5 \
  --lambda-variance 0.01 \
  --lambda-covariance 0.01 \
  --lambda-slot-diversity 0.01 \
  --device mps
```

訓練輸出：

```text
results_selfsup/convnext_tiny_slots4/
  best_self_supervised_encoder.pt
  last_self_supervised_encoder.pt
  training_log.csv
```

Best checkpoint 由 validation total representation objective 選擇，不使用五個 pseudo-score MAE。

## 6. Evaluation command

Training 完成後執行：

```bash
DEVICE=mps \
EXPERIMENT_DIR=./results_selfsup/convnext_tiny_slots4 \
./scripts/run_self_supervised_evaluation.sh
```

若要同時輸出代表圖片的 anonymous slot attention：

```bash
DEVICE=mps \
EXPERIMENT_DIR=./results_selfsup/convnext_tiny_slots4 \
SELECTION_CSV=./results_task_attention/convnext_tiny_finetune/representative_images.csv \
./scripts/run_self_supervised_evaluation.sh
```

## 7. Evaluation outputs

```text
eval/evaluation_summary.csv
eval/synthetic_retrieval.csv
eval/synthetic_retrieval_features.npz
eval/severity_monotonicity.csv
eval/severity_trajectories.npz
eval/linear_probe_metrics.csv
eval/raw_reference_ranking.csv
eval/collapse_metrics.csv
eval/features_test.npz
vis/z_deg_pca_by_recipe.png
vis/z_deg_tsne_by_recipe.png
vis/severity_trajectories_pca.png
vis/slot_attention/
```

### Synthetic retrieval

Query 與 database 會排除相同 content，目標是在不同內容中找回相同 anonymous recipe：

```text
Top-1
Top-5
mAP
Recall@5
```

### Content invariance vs degradation sensitivity

報告三種 cosine distance：

```text
D(same content, same degradation, augmentation)
D(different content, same degradation)
D(same content, different degradation)
```

理想上第三項最大，第一項最小。

### Severity monotonicity

對 random mixed recipe directions 做 severity `0 -> 1` sweep：

```text
Spearman(severity, distance_to_clean)
increasing_step_fraction
distance_range
```

### Frozen linear probe

主模型 training 不使用 pseudo scores。Evaluation 才以 cross-validation ridge probe 預測五個 score，輸出：

```text
MAE / RMSE / Spearman / R²
```

### Raw/reference ranking

使用 generic `m_deg` 檢查：

```text
m_deg(raw) > m_deg(reference)
```

輸出 ranking accuracy、average margin 與 AUC。

### Collapse diagnostics

```text
dimension standard deviation
effective rank / rank fraction
mean absolute feature correlation
mean pairwise z_deg cosine
mean anonymous-slot cosine
```

## 8. Model comparison

最終建議主表：

| Model | Training target | Type labels | Retrieval mAP ↑ | Severity Spearman ↑ | Raw/ref ranking ↑ | Probe avg MAE ↓ | Effective rank ↑ | Stage 2 gain ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ImageNet feature | none | No | | | | | | |
| V2 supervised z_deg | pseudo scores | Yes | | | | | | |
| Task-aware tokens | pseudo scores | Yes | | | | | | |
| Self-supervised z_deg | contrastive/order | No | | | | | | |
| Self-supervised slots | contrastive/order | No | | | | | | |

Stage 2 最終比較仍應以 enhancement output 的 PSNR、SSIM、LPIPS、UIQM/UCIQE 與 qualitative cases 判斷 learned representation 是否真正有用。
