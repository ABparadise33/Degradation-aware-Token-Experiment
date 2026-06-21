import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from stage1.data import build_transform
from stage1.engine import device_from_arg
from stage1.model import load_assessor
from stage1.synthetic import apply_synthetic_blur


def main():
    parser = argparse.ArgumentParser(description="Evaluate blur-score response to known synthetic severity.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    model = load_assessor(args.checkpoint, device)
    with open(args.selection_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    severities = np.linspace(0.0, 1.0, args.levels, dtype=np.float32)
    details = []
    summaries = []

    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = build_transform(args.image_size, train=False)(image).unsqueeze(0).to(device)
        for blur_type in ("gaussian", "motion"):
            predictions = []
            for severity in severities:
                severity_tensor = tensor.new_tensor([severity])
                degraded, _ = apply_synthetic_blur(
                    tensor,
                    severity_tensor,
                    blur_type,
                    motion_direction=0 if blur_type == "motion" else None,
                )
                with torch.no_grad():
                    prediction = float(model(degraded)["scores"][0, 1])
                predictions.append(prediction)
                details.append(
                    {
                        "image_name": row["image_name"],
                        "category": row["category"],
                        "blur_type": blur_type,
                        "severity": float(severity),
                        "predicted_s_blur": prediction,
                    }
                )
            correlation = float(spearmanr(severities, predictions).statistic)
            monotonic_steps = float(np.mean(np.diff(predictions) > 0))
            summaries.append(
                {
                    "image_name": row["image_name"],
                    "category": row["category"],
                    "blur_type": blur_type,
                    "spearman_severity_vs_prediction": correlation,
                    "increasing_step_fraction": monotonic_steps,
                    "prediction_range": max(predictions) - min(predictions),
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "synthetic_blur_details.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with open(os.path.join(args.output_dir, "synthetic_blur_summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    for blur_type in ("gaussian", "motion"):
        selected = [row for row in summaries if row["blur_type"] == blur_type]
        print(
            blur_type,
            "mean_spearman=",
            np.nanmean([row["spearman_severity_vs_prediction"] for row in selected]),
            "mean_range=",
            np.mean([row["prediction_range"] for row in selected]),
        )
    print(f"Wrote synthetic blur evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
