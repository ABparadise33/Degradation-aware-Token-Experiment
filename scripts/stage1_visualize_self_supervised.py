import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from stage1.data import build_transform
from stage1.engine import device_from_arg
from stage1.self_supervised_model import load_self_supervised_encoder


def scatter_latent(features_path: str, output_dir: str, seed: int):
    data = np.load(features_path)
    z = data["z_deg"]
    recipe_ids = data["recipe_ids"]
    pca = PCA(n_components=2, random_state=seed).fit_transform(z)
    perplexity = max(5, min(30, len(z) // 5, len(z) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(z)
    for name, points in (("pca", pca), ("tsne", tsne)):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            points[:, 0],
            points[:, 1],
            c=recipe_ids,
            cmap="tab10",
            s=20,
            alpha=0.8,
        )
        plt.colorbar(scatter, label="Anonymous recipe ID")
        plt.title(f"Self-supervised z_deg by synthetic recipe ({name.upper()})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"z_deg_{name}_by_recipe.png"), dpi=180)
        plt.close()


def plot_severity_trajectories(path: str, output_dir: str, seed: int):
    data = np.load(path)
    features = data["features"]
    clean = data["clean_features"]
    levels = data["levels"]
    directions, contents, steps, dim = features.shape
    combined = np.concatenate([clean, features.reshape(-1, dim)], axis=0)
    projected = PCA(n_components=2, random_state=seed).fit_transform(combined)
    projected_features = projected[len(clean) :].reshape(directions, contents, steps, 2)

    plt.figure(figsize=(9, 7))
    colors = plt.get_cmap("tab10")
    for direction in range(directions):
        mean_path = projected_features[direction].mean(axis=0)
        plt.plot(
            mean_path[:, 0],
            mean_path[:, 1],
            marker="o",
            color=colors(direction),
            label=f"recipe direction {direction}",
        )
        for index, level in enumerate(levels):
            plt.annotate(f"{level:.2f}", mean_path[index], fontsize=7)
    plt.title("Mean z_deg severity trajectories in PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "severity_trajectories_pca.png"), dpi=180)
    plt.close()


def visualize_slots(
    checkpoint: str,
    selection_csv: str,
    output_dir: str,
    image_size: int,
    device_name: str,
):
    device = device_from_arg(device_name)
    model = load_self_supervised_encoder(checkpoint, device)
    with open(selection_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = build_transform(image_size, train=False)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
        base = np.asarray(image).astype(np.float32) / 255.0
        category = row.get("category", "image")
        role = row.get("role", "unknown")
        stem = f"{category}_{role}_{os.path.splitext(row['image_name'])[0]}"
        target_dir = os.path.join(output_dir, "slot_attention", stem)
        os.makedirs(target_dir, exist_ok=True)
        for index in range(model.num_slots):
            attention = output["slot_attention"][0, index].cpu().numpy()
            attention = np.asarray(
                Image.fromarray(attention.astype(np.float32)).resize(
                    image.size,
                    Image.Resampling.BICUBIC,
                )
            )
            attention = (attention - attention.min()) / (
                attention.max() - attention.min() + 1e-8
            )
            heat = plt.get_cmap("jet")(attention)[:, :, :3]
            overlay = np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)
            plt.imsave(os.path.join(target_dir, f"slot_{index}_attention.png"), overlay)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize self-supervised degradation embeddings and anonymous slots."
    )
    parser.add_argument("--synthetic-features", required=True)
    parser.add_argument("--severity-trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scatter_latent(args.synthetic_features, args.output_dir, args.seed)
    plot_severity_trajectories(args.severity_trajectories, args.output_dir, args.seed)
    if bool(args.checkpoint) != bool(args.selection_csv):
        parser.error("--checkpoint and --selection-csv must be provided together.")
    if args.checkpoint:
        visualize_slots(
            args.checkpoint,
            args.selection_csv,
            args.output_dir,
            args.image_size,
            args.device,
        )
    print(f"Wrote self-supervised visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
