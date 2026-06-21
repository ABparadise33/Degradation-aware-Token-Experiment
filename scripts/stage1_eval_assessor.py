import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.engine import device_from_arg, evaluate_assessor


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 degradation-aware assessor.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="UIEB root containing raw-890/reference-890 or raw/GT.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", help="Use train, val, test, or all.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = evaluate_assessor(
        labels_csv=args.labels_csv,
        split=split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device_from_arg(args.device),
        checkpoint_path=args.checkpoint,
        output_csv=os.path.join(args.output_dir, f"predictions_{args.split}.csv"),
        output_npz=os.path.join(args.output_dir, f"features_{args.split}.npz"),
        dataset_root=args.dataset_root,
    )

    metrics_path = os.path.join(args.output_dir, f"metrics_{args.split}.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in sorted(metrics.items()):
            writer.writerow({"metric": key, "value": value})
            print(f"{key}: {value}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
