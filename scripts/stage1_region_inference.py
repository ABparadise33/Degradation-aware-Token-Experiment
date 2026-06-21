import argparse
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
from stage1.regions import compose_region_maps, masked_average_pool
from stage1.spatial import load_spatial_backbone


def main():
    parser = argparse.ArgumentParser(description="Pool assessor features inside SAM masks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--masks-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    model = load_assessor(args.checkpoint, device)
    image = Image.open(args.image).convert("RGB")
    tensor = build_transform(args.image_size, train=False)(image).unsqueeze(0).to(device)
    masks_np = np.load(args.masks_npz, allow_pickle=True)["masks"]
    masks = torch.from_numpy(masks_np).unsqueeze(0).to(device)

    with torch.no_grad():
        global_output = model(tensor)
        if "spatial_feature" in global_output:
            spatial = global_output["spatial_feature"]
        else:
            spatial_model, _ = load_spatial_backbone(args.checkpoint, device)
            spatial = spatial_model(tensor)[-1]
        region_features = masked_average_pool(spatial, masks)
        flat = region_features.reshape(-1, region_features.shape[-1])
        scored = model.score_pooled_features(flat)
        region_tokens, region_scores = scored[:2]
        region_task_tokens = scored[2] if len(scored) > 2 else None
        region_scores = region_scores.reshape(1, masks.shape[1], len(SCORE_COLUMNS))
        dense_maps = compose_region_maps(region_scores, masks, (image.height, image.width))

    os.makedirs(args.output_dir, exist_ok=True)
    arrays = {
        "global_scores": global_output["scores"].cpu().numpy(),
        "global_z_deg": global_output["z_deg"].cpu().numpy(),
        "region_scores": region_scores.cpu().numpy(),
        "region_tokens": region_tokens.reshape(1, masks.shape[1], -1).cpu().numpy(),
        "region_maps": dense_maps.cpu().numpy(),
        "masks": masks_np,
        "score_columns": np.asarray(SCORE_COLUMNS),
        "image_path": np.asarray(args.image),
    }
    if "task_tokens" in global_output:
        arrays["global_task_tokens"] = global_output["task_tokens"].cpu().numpy()
    if region_task_tokens is not None:
        arrays["region_task_tokens"] = region_task_tokens.reshape(
            1, masks.shape[1], len(SCORE_COLUMNS), -1
        ).cpu().numpy()
    if "attention_maps" in global_output:
        arrays["global_attention_maps"] = global_output["attention_maps"].cpu().numpy()
    np.savez_compressed(os.path.join(args.output_dir, "stage1_conditions.npz"), **arrays)

    base = np.asarray(image).astype(np.float32) / 255.0
    for index, name in enumerate(SCORE_COLUMNS[:4]):
        score_map = dense_maps[0, index].cpu().numpy()
        heat = plt.get_cmap("magma")(score_map)[:, :, :3]
        overlay = np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)
        plt.imsave(os.path.join(args.output_dir, f"{name}_region_map.png"), overlay)
    print(f"Wrote Stage 1 region conditions to {args.output_dir}")


if __name__ == "__main__":
    main()
