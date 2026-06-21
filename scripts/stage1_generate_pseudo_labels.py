import argparse
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.pseudo_labels import build_pseudo_label_rows, write_csv


def main():
    parser = argparse.ArgumentParser(description="Generate weak underwater degradation pseudo-labels.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing degraded/raw underwater images.")
    parser.add_argument("--reference-dir", default=None, help="Directory containing paired reference images.")
    parser.add_argument("--output", required=True, help="Output pseudo-label CSV path.")
    parser.add_argument(
        "--portable-dataset-root",
        default=None,
        help=(
            "Optional dataset path stored relative to the output CSV, for example ../datasets/full. "
            "Raw/reference paths will be written below raw/ and GT/."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must satisfy train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    rows = build_pseudo_label_rows(
        raw_dir=args.raw_dir,
        reference_dir=args.reference_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if args.portable_dataset_root:
        for row in rows:
            source_root = args.raw_dir if row["role"] == "raw" else args.reference_dir
            relative_name = os.path.relpath(row["image_path"], os.path.abspath(source_root))
            role_dir = "raw" if row["role"] == "raw" else "GT"
            row["image_path"] = os.path.normpath(
                os.path.join(args.portable_dataset_root, role_dir, relative_name)
            )
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} pseudo-label rows to {args.output}")
    pair_counts = Counter((row["split"], row["pair_id"]) for row in rows)
    split_counts = Counter(split for split, _ in pair_counts)
    print(
        "Pair split: "
        + ", ".join(f"{split}={split_counts.get(split, 0)}" for split in ("train", "val", "test"))
    )


if __name__ == "__main__":
    main()
