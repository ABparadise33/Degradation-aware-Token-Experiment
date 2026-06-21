# Underwater Stage 1

Weakly-supervised Underwater Degradation-aware Feature Learning

## Research pipeline

新版完整 pipeline（V2 supervised token、task-aware cross-attention、feature validation、spatial maps、synthetic losses、SAM region pooling、Stage 2 condition）請見：

[`RESEARCH_PIPELINE.md`](RESEARCH_PIPELINE.md)

不使用五個固定 pseudo-score 作為主訓練目標的 anonymous-slot self-supervised pipeline 請見：

[`SELF_SUPERVISED_PIPELINE.md`](SELF_SUPERVISED_PIPELINE.md)

正式 training commands 已整理於該文件；本 repository 的 code verification 不會自動啟動長時間 training。

## Current experiment status

V2 四組 20-epoch training/evaluation、Task-aware baseline、feature validation、task-token specialization、attention faithfulness 與 synthetic blur sensitivity baseline 均已完成。

| Completed model | Average 5-score MAE ↓ | Quality ranking ↑ |
|---|---:|---:|
| **Task-aware ConvNeXt-Tiny fine-tune** | **0.049293** | **1.000000** |
| V2 ConvNeXt-Tiny fine-tune | 0.061686 | 0.993103 |
| V2 ConvNeXt-Tiny frozen | 0.083868 | 0.979310 |
| V2 ResNet-50 frozen | 0.111144 | 0.510345 |
| V2 ResNet-50 fine-tune | 0.122876 | 0.496552 |

Task-aware model 的 regression 最佳，但五個 task tokens 仍高度相似，attention faithfulness 也只部分成立。Synthetic blur supervision + diversity-loss ablation 目前正在執行，尚未列入 completed 結果。

完整數字、指標定義、圖像分析與限制請見：

[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)

公開的 evaluation 圖片與精簡 CSV 位於：

- [`reports/assets`](reports/assets)
- [`reports/metrics`](reports/metrics)

Repository 不上傳 UIEB dataset、`.pt` checkpoints 或大型 feature dumps。

這個 project 是從 `Underwater_FlowIE` 拆出的獨立 Stage 1 實驗工具。目的不是直接做 underwater image enhancement，而是先訓練一個 degradation-aware assessor，用 weak supervision 判斷 backbone feature 是否能表達水下退化資訊。

模型會輸出：

```bash
s_color
s_blur
s_contrast
s_visibility_proxy
q_quality
z_deg
```

其中 `z_deg` 是 latent degradation token，後續可以接到 Stage 2 preprocessing 或 FlowIE condition。

## Project Structure

```bash
Underwater_Stage1/
├── README.md
├── requirements.txt
├── metadata/
│   └── uieb_pseudo_labels.csv
├── scripts/
│   ├── download_uieb.py
│   ├── stage1_generate_pseudo_labels.py
│   ├── stage1_train_assessor.py
│   ├── stage1_eval_assessor.py
│   ├── stage1_visualize_features.py
│   ├── stage1_validate_features.py
│   ├── stage1_visualize_spatial_features.py
│   ├── stage1_batch_gradcam.py
│   ├── stage1_generate_sam_masks.py
│   └── stage1_region_inference.py
└── stage1/
    ├── pseudo_labels.py
    ├── data.py
    ├── model.py
    ├── engine.py
    ├── metrics.py
    ├── spatial.py
    ├── synthetic.py
    ├── visibility.py
    ├── regions.py
    └── conditioning.py
```

## Setup

### 1. Install Environment

如果已經有 `FlowIE` conda environment：

```bash
conda activate FlowIE
pip install -r requirements.txt
```

或建立獨立環境：

```bash
conda create -n underwater_stage1 python=3.9 -y
conda activate underwater_stage1
pip install -r requirements.txt
```

### 2. Download UIEB

和 `Underwater_FlowIE` 一樣，UIEB 會直接從 Hugging Face 下載：

```bash
python scripts/download_uieb.py
```

如果本機已經有以下資料，就不需要再次下載：

```bash
../Underwater_Dataset/UIEB/raw-890
../Underwater_Dataset/UIEB/reference-890
```

下載完成後的結構：

```bash
datasets/
└── full/
    ├── raw/
    └── GT/
```

下載來源與 FlowIE 相同：

```bash
https://huggingface.co/datasets/Edddddd8787/UIEB
```

Stage 1 不需要實際移動或複製圖片來切分資料。完整資料保持在 `datasets/full`，
repository 內已經提供預先計算完成的：

```bash
metadata/uieb_pseudo_labels.csv
```

CSV 內的 `split` 欄位已用固定 hash 將每個 raw/GT pair 依目標比例
70% train、15% val、15% test 切分。UIEB 890 pairs 的實際固定結果為：

```bash
train: 619 pairs
val:   126 pairs
test:  145 pairs
```

raw 與對應 GT 一定在同一個 split。

因此下載完成後不需要執行 `split_dataset.py`，也不需要重新計算 pseudo-label，
可以直接開始 Training。這裡的「直接 Training」並不是使用全部 890 pairs；
training loader 只會讀取 CSV 中 `split=train` 的資料。

```bash
train: 更新模型參數
val: 每個 epoch 評估並選擇 best checkpoint
test: 模型與超參數確定後，進行最終一次評估
```

### 3. Train

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/resnet50_frozen \
  --backbone resnet50 \
  --freeze-backbone \
  --epochs 20 \
  --batch-size 16 \
  --device mps
```

### 4. Test

```bash
python scripts/stage1_eval_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --checkpoint ./results/resnet50_frozen/best_stage1_assessor.pt \
  --split test \
  --device mps \
  --output-dir ./results/resnet50_frozen/eval
```

## Apple Silicon MPS

在一般 macOS Terminal 中先確認目前 PyTorch 能使用 Metal：

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

最後一行應該是：

```bash
True
```

程式的 `--device auto` 會依照 CUDA、MPS、CPU 的順序自動選擇。Mac 上也可以明確指定：

```bash
--device mps
```

若個別 PyTorch operation 尚未支援 MPS，可以先啟用 CPU fallback：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Run All Four Experiments

以下 script 會依序完成四組實驗的 Training 和 Test：

```bash
chmod +x scripts/run_all_mps_experiments.sh
./scripts/run_all_mps_experiments.sh
```

預設設定：

```bash
dataset: ../Underwater_Dataset/UIEB
device: mps
epochs: 20
batch size: 16
workers: 4
```

可以用環境變數調整，例如先跑 1 epoch smoke test：

```bash
EPOCHS=1 BATCH_SIZE=8 NUM_WORKERS=0 ./scripts/run_all_mps_experiments.sh
```

四組等價的 Training commands 如下。

### ResNet-50 Frozen

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/resnet50_frozen \
  --backbone resnet50 \
  --freeze-backbone \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --device mps
```

### ResNet-50 Fine-tune

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/resnet50_finetune \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-5 \
  --device mps
```

### ConvNeXt-Tiny Frozen

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/convnext_tiny_frozen \
  --backbone convnext_tiny \
  --freeze-backbone \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --device mps
```

### ConvNeXt-Tiny Fine-tune

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/convnext_tiny_finetune \
  --backbone convnext_tiny \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-5 \
  --device mps
```

每組完成後，runner 會自動使用 `test` split 評估，輸出位置如下：

```bash
results/<experiment_name>/eval/metrics_test.csv
results/<experiment_name>/eval/predictions_test.csv
results/<experiment_name>/eval/features_test.npz
```

## Dataset Split Design

這個 project 使用「manifest-based split」：圖片維持在同一個資料夾，
由 CSV 決定每張圖片屬於哪一個 split。這和實際建立
`datasets/train`、`datasets/val`、`datasets/test` 的效果相同，但不會重複或搬動圖片。

預先計算 CSV 的內容包含：

```bash
image_path, image_name, pair_id, split, role, is_reference
color_raw, sharpness_raw, contrast_raw, saturation_raw
s_color, s_blur, s_contrast, s_visibility_proxy, q_quality
```

所有圖片的原始 no-reference metrics 雖然一次計算完成，但 pseudo-label 的 min-max
正規化範圍只使用 train split 擬合，再套用到 val/test，避免資料洩漏。

## Regenerate Pseudo-labels

一般建置與訓練不需要執行本節。只有修改 pseudo-label 公式、split 比例或更換
dataset 時，才需要重新產生：

```bash
python scripts/stage1_generate_pseudo_labels.py \
  --raw-dir ./datasets/full/raw \
  --reference-dir ./datasets/full/GT \
  --output ./metadata/uieb_pseudo_labels.csv \
  --portable-dataset-root ../datasets/full
```

`--portable-dataset-root` 讓 CSV 儲存相對路徑，因此 repository 移到其他電腦後，
只要資料仍下載在 `datasets/full` 就能直接使用。

## Frozen Backbone Baseline

Frozen backbone 用來測試 ImageNet pretrained feature 本身是否已包含水下退化資訊。

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/resnet50_frozen \
  --backbone resnet50 \
  --freeze-backbone \
  --epochs 20 \
  --batch-size 16
```

可替換 backbone：

```bash
--backbone convnext_tiny
--backbone edgenext_small
--backbone swin_tiny_patch4_window7_224
```

實際可用名稱取決於目前安裝的 `timm` 版本。

## Fine-tuned Backbone

Fine-tuned setting 用來測試 backbone 是否能被訓練成 degradation-aware feature extractor。

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --output-dir ./results/convnext_tiny_finetune \
  --backbone convnext_tiny \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-5
```

與 frozen setting 的差異是沒有加 `--freeze-backbone`。

## Evaluate and Export Features

```bash
python scripts/stage1_eval_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --dataset-root ../Underwater_Dataset/UIEB \
  --checkpoint ./results/resnet50_frozen/best_stage1_assessor.pt \
  --split test \
  --output-dir ./results/resnet50_frozen/eval
```

輸出：

```bash
results/resnet50_frozen/eval/metrics_test.csv
results/resnet50_frozen/eval/predictions_test.csv
results/resnet50_frozen/eval/features_test.npz
```

`metrics_test.csv` 會包含：

```bash
MAE
RMSE
Pearson correlation
Spearman correlation
R2
Ranking accuracy
Avg q_ref - q_raw
```

`features_test.npz` 會包含：

```bash
z_deg
preds
targets
image_paths
pair_ids
roles
score_columns
```

## Visualize Feature Space

```bash
python scripts/stage1_visualize_features.py \
  --features ./results/resnet50_frozen/eval/features_test.npz \
  --output-dir ./results/resnet50_frozen/vis
```

輸出：

```bash
results/resnet50_frozen/vis/z_deg_pca.png
results/resnet50_frozen/vis/z_deg_tsne.png
```

這一步會把 `z_deg` 依 degradation type 做 PCA / t-SNE 視覺化。

四組模型可以一次執行：

```bash
DEVICE=mps ./scripts/run_post_training_evaluation.sh
```

## Grad-CAM

針對指定 score 產生 Grad-CAM，例如 weak visibility proxy：

```bash
python scripts/stage1_gradcam.py \
  --checkpoint ./results/resnet50_frozen/best_stage1_assessor.pt \
  --image ./datasets/full/raw/392_img_.png \
  --score s_visibility_proxy \
  --output ./results/resnet50_frozen/gradcam/392_visibility.png
```

可用 score：

```bash
s_color
s_blur
s_contrast
s_visibility_proxy
q_quality
```

如果自動選到的 convolution layer 不適合，可以先列出可用 layer：

```bash
python scripts/stage1_gradcam.py \
  --checkpoint ./results/resnet50_frozen/best_stage1_assessor.pt \
  --image ./datasets/full/raw/392_img_.png \
  --output ./results/resnet50_frozen/gradcam/tmp.png \
  --list-layers
```

再指定：

```bash
--target-layer backbone.layer4.1.conv2
```

目前已使用最佳 `convnext_tiny_finetune` checkpoint，對 `392_img_.png` 的五個 score 完成 Grad-CAM，輸出位於：

```bash
results/convnext_tiny_finetune/gradcam/
```

## Evaluate All Trained Models

一次重新 evaluate 四組已訓練模型：

```bash
DEVICE=mps ./scripts/evaluate_all_models.sh
```

若 MPS 不可用：

```bash
DEVICE=cpu NUM_WORKERS=0 ./scripts/evaluate_all_models.sh
```

## Suggested Experiments

建議先跑這三組：

```bash
resnet50_frozen
convnext_tiny_frozen
convnext_tiny_finetune
```

比較重點：

```bash
Score prediction MAE / Spearman
Ranking accuracy
Feature PCA / t-SNE separation
Grad-CAM interpretability
Model cost
```

## Smoke Test

如果只是要確認 pipeline 可跑，可以用小圖和 random backbone 快速測試：

```bash
python scripts/stage1_train_assessor.py \
  --labels-csv ./metadata/uieb_pseudo_labels.csv \
  --output-dir /tmp/stage1_train_smoke \
  --backbone resnet18 \
  --no-pretrained \
  --freeze-backbone \
  --image-size 64 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 0 \
  --device cpu
```
