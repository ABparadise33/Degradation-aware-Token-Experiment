import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from stage1.pseudo_labels import SCORE_COLUMNS


def probe_mae(token: np.ndarray, target: np.ndarray, folds: int) -> float:
    prediction = np.zeros_like(target, dtype=np.float64)
    for train_index, test_index in KFold(n_splits=folds, shuffle=True, random_state=0).split(token):
        train_x = token[train_index]
        test_x = token[test_index]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        train_x = (train_x - mean) / scale
        test_x = (test_x - mean) / scale
        train_y = target[train_index]
        target_mean = train_y.mean()
        centered_y = train_y - target_mean
        train_tensor = torch.from_numpy(train_x)
        test_tensor = torch.from_numpy(test_x)
        target_tensor = torch.from_numpy(centered_y)
        identity = torch.eye(train_x.shape[1], dtype=torch.float64)
        weights = torch.linalg.solve(
            train_tensor.T @ train_tensor + identity,
            train_tensor.T @ target_tensor,
        )
        prediction[test_index] = (test_tensor @ weights).numpy() + target_mean
    return float(mean_absolute_error(target, prediction))


def cosine_similarity_matrix(task_tokens: np.ndarray) -> np.ndarray:
    normalized = task_tokens / np.maximum(np.linalg.norm(task_tokens, axis=2, keepdims=True), 1e-12)
    return np.einsum("ntd,nud->tu", normalized, normalized) / len(normalized)


def save_heatmap(matrix: np.ndarray, output: str, title: str, fmt: str = ".3f") -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(range(len(SCORE_COLUMNS)), SCORE_COLUMNS, rotation=35, ha="right")
    axis.set_yticks(range(len(SCORE_COLUMNS)), SCORE_COLUMNS)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, format(matrix[row, column], fmt), ha="center", va="center", color="white")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Measure task-token specialization and disentanglement.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    if "task_tokens" not in data:
        raise ValueError("The NPZ does not contain task_tokens. Evaluate a task_attention checkpoint first.")
    tokens = data["task_tokens"].astype(np.float64)
    targets = data["targets"].astype(np.float64)
    if tokens.shape[1] != len(SCORE_COLUMNS):
        raise ValueError(f"Expected {len(SCORE_COLUMNS)} task tokens, got {tokens.shape[1]}.")

    mae_matrix = np.zeros((len(SCORE_COLUMNS), len(SCORE_COLUMNS)), dtype=np.float64)
    for token_index in range(len(SCORE_COLUMNS)):
        for target_index in range(len(SCORE_COLUMNS)):
            mae_matrix[token_index, target_index] = probe_mae(
                tokens[:, token_index],
                targets[:, target_index],
                args.folds,
            )

    cosine_matrix = cosine_similarity_matrix(tokens)
    diagonal = np.diag(mae_matrix)
    best_token_per_target = np.argmin(mae_matrix, axis=0)
    own_token_wins = best_token_per_target == np.arange(len(SCORE_COLUMNS))
    other_mean = (mae_matrix.sum(axis=0) - diagonal) / (len(SCORE_COLUMNS) - 1)
    specialization_gain = other_mean - diagonal

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "task_token_probe_matrix.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["token"] + SCORE_COLUMNS)
        for index, name in enumerate(SCORE_COLUMNS):
            writer.writerow([name] + list(mae_matrix[index]))

    with open(os.path.join(args.output_dir, "task_token_summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target",
                "own_token_mae",
                "other_tokens_mean_mae",
                "specialization_gain",
                "best_token",
                "own_token_is_best",
            ],
        )
        writer.writeheader()
        for index, name in enumerate(SCORE_COLUMNS):
            writer.writerow(
                {
                    "target": name,
                    "own_token_mae": diagonal[index],
                    "other_tokens_mean_mae": other_mean[index],
                    "specialization_gain": specialization_gain[index],
                    "best_token": SCORE_COLUMNS[best_token_per_target[index]],
                    "own_token_is_best": bool(own_token_wins[index]),
                }
            )

    with open(os.path.join(args.output_dir, "task_token_cosine_similarity.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["token"] + SCORE_COLUMNS)
        for index, name in enumerate(SCORE_COLUMNS):
            writer.writerow([name] + list(cosine_matrix[index]))

    save_heatmap(
        mae_matrix,
        os.path.join(args.output_dir, "task_token_probe_mae.png"),
        "Linear-probe MAE: task token → target score",
    )
    save_heatmap(
        cosine_matrix,
        os.path.join(args.output_dir, "task_token_cosine_similarity.png"),
        "Mean task-token cosine similarity",
    )
    print(f"Own token is best for {int(own_token_wins.sum())}/{len(SCORE_COLUMNS)} targets.")
    print(f"Wrote task-token validation to {args.output_dir}")


if __name__ == "__main__":
    main()
