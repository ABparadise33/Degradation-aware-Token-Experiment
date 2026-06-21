import argparse
import csv
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stage1.data import read_label_rows, score_value
from stage1.pseudo_labels import SCORE_COLUMNS


def ranked(rows: List[Dict[str, str]], key, count: int, reverse: bool = True):
    return sorted(rows, key=key, reverse=reverse)[:count]


def main():
    parser = argparse.ArgumentParser(description="Select representative UIEB images for qualitative analysis.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_label_rows(args.labels_csv, split=args.split, dataset_root=args.dataset_root)
    raw = [row for row in rows if row["role"] == "raw"]
    reference = [row for row in rows if row["role"] == "reference"]
    selected = []

    categories = {
        "color": lambda row: score_value(row, "s_color"),
        "blur": lambda row: score_value(row, "s_blur"),
        "contrast": lambda row: score_value(row, "s_contrast"),
        "visibility": lambda row: score_value(row, "s_visibility_proxy"),
    }
    for category, key in categories.items():
        selected.extend((category, row) for row in ranked(raw, key, args.per_category))

    def mixed_score(row):
        values = np.asarray([score_value(row, name) for name in SCORE_COLUMNS[:4]])
        return float(values.mean() - values.std())

    selected.extend(("mixed", row) for row in ranked(raw, mixed_score, args.per_category))
    selected.extend(
        ("reference", row)
        for row in ranked(reference, lambda row: score_value(row, "q_quality"), args.per_category)
    )

    output_rows = []
    seen = set()
    for category, row in selected:
        key = (category, row["image_path"])
        if key in seen:
            continue
        seen.add(key)
        output_rows.append(
            {
                "category": category,
                "image_path": row["image_path"],
                "image_name": row["image_name"],
                "pair_id": row["pair_id"],
                "role": row["role"],
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} representative-image rows to {args.output}")


if __name__ == "__main__":
    main()
