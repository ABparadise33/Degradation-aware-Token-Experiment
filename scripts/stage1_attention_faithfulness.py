import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from stage1.data import build_transform
from stage1.engine import device_from_arg
from stage1.model import load_assessor
from stage1.pseudo_labels import SCORE_COLUMNS


def binary_region(attention: torch.Tensor, fraction: float, mode: str, generator: torch.Generator) -> torch.Tensor:
    flat = attention.flatten()
    count = max(1, int(round(flat.numel() * fraction)))
    if mode == "top":
        indices = flat.topk(count).indices
    elif mode == "bottom":
        indices = (-flat).topk(count).indices
    elif mode == "random":
        indices = torch.randperm(flat.numel(), generator=generator)[:count].to(flat.device)
    else:
        raise ValueError(mode)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    return mask.reshape(attention.shape)


def mask_image(image: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    region = F.interpolate(
        region[None, None].float(),
        size=image.shape[-2:],
        mode="nearest",
    ).bool()
    # Zero is the ImageNet-normalized dataset mean, a neutral replacement.
    return image.masked_fill(region, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Test whether task attention is causally relevant to scores.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("--fraction must be between 0 and 1.")

    device = device_from_arg(args.device)
    model = load_assessor(args.checkpoint, device)
    with open(args.selection_csv, newline="") as file:
        selected = list(csv.DictReader(file))
    generator = torch.Generator().manual_seed(args.seed)
    detailed = []
    summary_values = defaultdict(list)

    for row in tqdm(selected, desc="attention faithfulness"):
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = build_transform(args.image_size, train=False)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            baseline = model(tensor)
        if "attention_maps" not in baseline:
            raise ValueError("Checkpoint is not a task_attention model.")

        for task_index, task_name in enumerate(SCORE_COLUMNS):
            attention = baseline["attention_maps"][0, task_index]
            base_scores = baseline["scores"][0]
            conditions = [("top", 0), ("bottom", 0)]
            conditions += [("random", repeat) for repeat in range(args.random_repeats)]
            for mode, repeat in conditions:
                region = binary_region(attention, args.fraction, mode, generator)
                masked = mask_image(tensor, region)
                with torch.no_grad():
                    changed_scores = model(masked)["scores"][0]
                deltas = (changed_scores - base_scores).cpu().numpy()
                own_abs_delta = float(abs(deltas[task_index]))
                collateral_abs_delta = float(np.delete(np.abs(deltas), task_index).mean())
                selectivity = own_abs_delta - collateral_abs_delta
                detailed.append(
                    {
                        "image_name": row["image_name"],
                        "category": row["category"],
                        "role": row["role"],
                        "task": task_name,
                        "mask_type": mode,
                        "repeat": repeat,
                        "baseline_score": float(base_scores[task_index]),
                        "masked_score": float(changed_scores[task_index]),
                        "signed_delta": float(deltas[task_index]),
                        "absolute_delta": own_abs_delta,
                        "collateral_absolute_delta": collateral_abs_delta,
                        "task_selectivity": selectivity,
                    }
                )
                summary_values[(task_name, mode, "absolute_delta")].append(own_abs_delta)
                summary_values[(task_name, mode, "selectivity")].append(selectivity)

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "attention_faithfulness_details.csv")
    with open(detail_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)

    summary_rows = []
    for task_name in SCORE_COLUMNS:
        values = {}
        for mode in ("top", "random", "bottom"):
            values[mode] = float(np.mean(summary_values[(task_name, mode, "absolute_delta")]))
        summary_rows.append(
            {
                "task": task_name,
                "top_absolute_delta": values["top"],
                "random_absolute_delta": values["random"],
                "bottom_absolute_delta": values["bottom"],
                "top_over_random": values["top"] / max(values["random"], 1e-12),
                "top_minus_bottom": values["top"] - values["bottom"],
                "top_selectivity": float(np.mean(summary_values[(task_name, "top", "selectivity")])),
                "faithful_order": values["top"] > values["random"] > values["bottom"],
            }
        )
    with open(os.path.join(args.output_dir, "attention_faithfulness_summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote attention faithfulness results to {args.output_dir}")


if __name__ == "__main__":
    main()
