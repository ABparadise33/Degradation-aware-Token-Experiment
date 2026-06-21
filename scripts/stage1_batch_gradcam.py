import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.pseudo_labels import SCORE_COLUMNS
from stage1_gradcam import create_gradcam


def main():
    parser = argparse.ArgumentParser(description="Generate all score Grad-CAMs for a selected image manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-layer", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    with open(args.selection_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        stem = f"{row['category']}_{row['role']}_{os.path.splitext(row['image_name'])[0]}"
        for score in SCORE_COLUMNS:
            output = os.path.join(args.output_dir, stem, f"{score}.png")
            layer = create_gradcam(
                checkpoint=args.checkpoint,
                image_path=row["image_path"],
                output_path=output,
                score_name=score,
                device_name=args.device,
                target_layer_name=args.target_layer,
                image_size=args.image_size,
            )
            print(f"{stem}: {score} from {layer}")


if __name__ == "__main__":
    main()
