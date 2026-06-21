import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def degradation_groups(targets: np.ndarray, roles: np.ndarray) -> np.ndarray:
    names = np.array(["color", "blur", "contrast", "haze"])
    max_idx = np.argmax(targets[:, :4], axis=1)
    groups = names[max_idx].astype(object)
    mild = targets[:, :4].mean(axis=1) < np.percentile(targets[:, :4].mean(axis=1), 30)
    groups[mild] = "mild"
    groups[roles == "reference"] = "reference"
    return groups


def plot_embedding(points: np.ndarray, labels: np.ndarray, output_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(8, 6))
    for label in sorted(set(labels)):
        mask = labels == label
        plt.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.75, label=label)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(frameon=False, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize exported z_deg feature embeddings.")
    parser.add_argument("--features", required=True, help="NPZ from stage1_eval_assessor.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    z_deg = data["z_deg"]
    targets = data["targets"]
    roles = data["roles"].astype(str)
    labels = degradation_groups(targets, roles)
    os.makedirs(args.output_dir, exist_ok=True)

    pca_points = PCA(n_components=2, random_state=0).fit_transform(z_deg)
    plot_embedding(pca_points, labels, os.path.join(args.output_dir, "z_deg_pca.png"), "Stage 1 z_deg PCA")

    perplexity = min(args.perplexity, max(5.0, (len(z_deg) - 1) / 3.0))
    tsne_points = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=0).fit_transform(z_deg)
    plot_embedding(tsne_points, labels, os.path.join(args.output_dir, "z_deg_tsne.png"), "Stage 1 z_deg t-SNE")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
