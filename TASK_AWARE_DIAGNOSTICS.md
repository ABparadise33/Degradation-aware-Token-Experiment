# Task-aware Follow-up Diagnostics

## Current baseline finding

The existing task-aware checkpoint improves five-score regression, but the task-token validation indicates representation collapse:

```text
own token is the best linear probe for 1/5 targets
mean pairwise task-token cosine similarity: approximately 0.994–0.997
```

Therefore, the current model supports task-specific output heads, but does not yet support a strong claim that the five task tokens are disentangled representations.

## 1. Run all non-training diagnostics

```bash
cd /Users/ed/Research/Underwater_Stage1
conda activate FlowIE

DEVICE=mps \
EXPERIMENT_DIR=./results_task_attention/convnext_tiny_finetune \
./scripts/run_task_aware_diagnostics.sh
```

Interpretation:

- `task_token_summary.csv`: an own token should ideally be the best probe for its corresponding target.
- `task_token_cosine_similarity.csv`: off-diagonal values should move away from `1.0`.
- `attention_faithfulness_summary.csv`: `top > random > bottom` and positive task selectivity support attention faithfulness.
- `synthetic_blur_summary.csv`: severity/prediction Spearman and monotonic-step fraction should approach `1.0`; prediction range should not collapse near zero.

## 2. Train the proposed blur/diversity ablation

This command writes to a new directory and preserves the existing task-aware baseline:

```bash
DEVICE=mps \
OUTPUT_DIR=./results_task_attention_blur_diverse/convnext_tiny_finetune \
LAMBDA_BLUR_SYNTHETIC=0.1 \
LAMBDA_BLUR_ORDER=0.1 \
LAMBDA_TASK_DIVERSITY=0.01 \
LAMBDA_ATTENTION_DIVERSITY=0.01 \
./scripts/run_task_attention_experiment.sh
```

Recommended ablations:

```text
A: existing task-aware baseline
B: + synthetic blur regression/order
C: + task/attention diversity
D: + synthetic blur + diversity
```

Compare test MAE/R², task-token diagonal specialization, attention faithfulness, and synthetic blur monotonicity. Do not select a model solely by average MAE.
