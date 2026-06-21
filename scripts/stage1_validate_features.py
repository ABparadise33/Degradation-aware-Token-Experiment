import argparse
import csv
import os
import sys
from itertools import combinations
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    mean_absolute_error,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stage1.pseudo_labels import SCORE_COLUMNS
from stage1_visualize_features import degradation_groups


def mean_pairwise_distance(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    diff = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    return float(distances[np.triu_indices(len(x), k=1)].mean())


def cluster_distances(z: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    classes = sorted(set(labels))
    intra = [mean_pairwise_distance(z[labels == name]) for name in classes]
    centroids = {name: z[labels == name].mean(axis=0) for name in classes}
    inter = [
        float(np.linalg.norm(centroids[a] - centroids[b]))
        for a, b in combinations(classes, 2)
    ]
    return float(np.nanmean(intra)), float(np.mean(inter))


def evaluate_labels(z: np.ndarray, labels: np.ndarray, prefix: str) -> Dict[str, float]:
    names, counts = np.unique(labels, return_counts=True)
    retained = names[counts >= 5]
    mask = np.isin(labels, retained)
    z = z[mask]
    labels = labels[mask]
    if len(retained) < 2:
        return {
            f"{prefix}_num_classes": float(len(retained)),
            f"{prefix}_num_samples": float(len(labels)),
        }
    _, counts = np.unique(labels, return_counts=True)
    folds = max(2, min(5, int(counts.min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    components = min(32, z.shape[1], len(z) - 1)
    knn = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=0),
        KNeighborsClassifier(n_neighbors=5),
    )
    linear = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=0),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
    )
    knn_pred = cross_val_predict(knn, z, labels, cv=cv)
    linear_pred = cross_val_predict(linear, z, labels, cv=cv)
    intra, inter = cluster_distances(z, labels)
    return {
        f"{prefix}_num_classes": float(len(set(labels))),
        f"{prefix}_num_samples": float(len(labels)),
        f"{prefix}_intra_class_distance": intra,
        f"{prefix}_inter_class_centroid_distance": inter,
        f"{prefix}_inter_over_intra": inter / intra if intra > 0 else float("nan"),
        f"{prefix}_silhouette": float(silhouette_score(z, labels)),
        f"{prefix}_davies_bouldin": float(davies_bouldin_score(z, labels)),
        f"{prefix}_knn_accuracy": float(accuracy_score(labels, knn_pred)),
        f"{prefix}_linear_probe_accuracy": float(accuracy_score(labels, linear_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="Quantitatively validate exported z_deg features.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    z = data["z_deg"].astype(np.float64)
    targets = data["targets"].astype(np.float64)
    preds = data["preds"].astype(np.float64)
    metrics: Dict[str, float] = {}
    metrics.update(evaluate_labels(z, degradation_groups(targets), "target_group"))
    metrics.update(evaluate_labels(z, degradation_groups(preds), "prediction_group"))

    # Five-fold linear regression probes estimate whether a frozen z_deg retains score information.
    for idx, name in enumerate(SCORE_COLUMNS):
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(32, z.shape[1], len(z) - 1), random_state=0),
            Ridge(alpha=1.0, solver="svd"),
        )
        prediction = cross_val_predict(model, z, targets[:, idx], cv=5)
        metrics[f"linear_probe_{name}_mae"] = float(mean_absolute_error(targets[:, idx], prediction))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in sorted(metrics.items()):
            writer.writerow({"metric": key, "value": value})
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
