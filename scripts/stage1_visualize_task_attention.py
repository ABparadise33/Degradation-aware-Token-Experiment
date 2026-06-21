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
from stage1.model import load_assessor
from stage1.pseudo_labels import SCORE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Visualize task-token cross-attention maps.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    model = load_assessor(args.checkpoint, device)
    with open(args.selection_csv, newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = build_transform(args.image_size, train=False)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
        if "attention_maps" not in output:
            raise ValueError("Checkpoint is not a task_attention model.")
        base = np.asarray(image).astype(np.float32) / 255.0
        stem = f"{row['category']}_{row['role']}_{os.path.splitext(row['image_name'])[0]}"
        target_dir = os.path.join(args.output_dir, stem)
        os.makedirs(target_dir, exist_ok=True)
        for index, name in enumerate(SCORE_COLUMNS):
            attention = output["attention_maps"][0, index].cpu().numpy()
            attention = np.asarray(
                Image.fromarray(attention.astype(np.float32)).resize(
                    image.size,
                    Image.Resampling.BICUBIC,
                )
            )
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            heat = plt.get_cmap("jet")(attention)[:, :, :3]
            overlay = np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)
            plt.imsave(os.path.join(target_dir, f"{name}_attention.png"), overlay)
    print(f"Wrote task attention maps for {len(rows)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
