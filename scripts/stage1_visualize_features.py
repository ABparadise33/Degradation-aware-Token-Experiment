import argparse
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


DEGRADATION_NAMES = np.array(["color", "blur", "contrast", "visibility"])
COLORS: Dict[str, str] = {
    "color": "tab:red",
    "blur": "tab:blue",
    "contrast": "tab:orange",
    "visibility": "tab:purple",
    "mild": "tab:green",
}
MARKERS = {"raw": "o", "reference": "s"}


def degradation_groups(scores: np.ndarray) -> np.ndarray:
    groups = DEGRADATION_NAMES[np.argmax(scores[:, :4], axis=1)].astype(object)
    severity = scores[:, :4].mean(axis=1)
    groups[severity < np.percentile(severity, 30)] = "mild"
    return groups.astype(str)


def plot_embedding(
    points: np.ndarray,
    groups: np.ndarray,
    roles: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for group in sorted(set(groups)):
        for role in sorted(set(roles)):
            mask = (groups == group) & (roles == role)
            if not np.any(mask):
                continue
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                c=COLORS[group],
                marker=MARKERS.get(role, "o"),
                s=28,
                alpha=0.72,
                edgecolors="none",
            )

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[name], label=name)
        for name in sorted(set(groups))
    ]
    role_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS.get(role, "o"),
            linestyle="",
            color="black",
            label=role,
            markerfacecolor="none",
        )
        for role in sorted(set(roles))
    ]
    legend_groups = ax.legend(handles=color_handles, title="degradation", frameon=False, loc="upper right")
    ax.add_artist(legend_groups)
    ax.legend(handles=role_handles, title="role", frameon=False, loc="lower right")
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize z_deg using degradation color and role marker.")
    parser.add_argument("--features", required=True, help="NPZ from stage1_eval_assessor.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    z_deg = data["z_deg"].astype(np.float64)
    targets = data["targets"]
    preds = data["preds"]
    roles = data["roles"].astype(str)
    os.makedirs(args.output_dir, exist_ok=True)

    pca_points = PCA(n_components=2, random_state=0).fit_transform(z_deg)
    perplexity = min(args.perplexity, max(5.0, (len(z_deg) - 1) / 3.0))
    tsne_points = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=0,
    ).fit_transform(z_deg)

    for label_source, scores in (("target", targets), ("prediction", preds)):
        groups = degradation_groups(scores)
        plot_embedding(
            pca_points,
            groups,
            roles,
            os.path.join(args.output_dir, f"z_deg_pca_{label_source}.png"),
            f"z_deg PCA ({label_source} labels)",
        )
        plot_embedding(
            tsne_points,
            groups,
            roles,
            os.path.join(args.output_dir, f"z_deg_tsne_{label_source}.png"),
            f"z_deg t-SNE ({label_source} labels)",
        )

    print(f"Wrote target/prediction PCA and t-SNE plots to {args.output_dir}")


if __name__ == "__main__":
    main()
