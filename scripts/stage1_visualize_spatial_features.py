import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from stage1.data import build_transform
from stage1.engine import device_from_arg
from stage1.spatial import load_spatial_backbone


def normalized(array: np.ndarray) -> np.ndarray:
    array = array - array.min()
    return array / (array.max() + 1e-8)


def save_overview(image: Image.Image, features, output_path: str) -> None:
    fig, axes = plt.subplots(1, len(features) + 1, figsize=(4 * (len(features) + 1), 4))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[0].axis("off")
    for index, feature in enumerate(features):
        activation = feature[0].detach().abs().mean(dim=0).cpu().numpy()
        axes[index + 1].imshow(normalized(activation), cmap="magma")
        axes[index + 1].set_title(f"stage {index} mean |activation|")
        axes[index + 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_channels(feature: torch.Tensor, stage: int, output_path: str, count: int) -> None:
    maps = feature[0, :count].detach().cpu().numpy()
    columns = 4
    rows = int(np.ceil(len(maps) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes):
        if index < len(maps):
            axis.imshow(normalized(maps[index]), cmap="viridis")
            axis.set_title(f"stage {stage}, channel {index}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export multi-stage spatial feature-map visualizations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    model, feature_info = load_spatial_backbone(args.checkpoint, device)
    with open(args.selection_csv, newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = build_transform(args.image_size, train=False)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor)
        stem = f"{row['category']}_{row['role']}_{os.path.splitext(row['image_name'])[0]}"
        target_dir = os.path.join(args.output_dir, stem)
        os.makedirs(target_dir, exist_ok=True)
        save_overview(image, features, os.path.join(target_dir, "overview.png"))
        for stage, feature in enumerate(features):
            save_channels(
                feature,
                stage,
                os.path.join(target_dir, f"stage_{stage}_channels.png"),
                min(args.channels, feature.shape[1]),
            )
    print(f"Wrote spatial feature maps for {len(rows)} images to {args.output_dir}")
    print(f"Feature stages: {feature_info}")


if __name__ == "__main__":
    main()
