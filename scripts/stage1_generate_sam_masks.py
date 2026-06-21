import argparse
import os

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate SAM automatic masks for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-area", type=int, default=256)
    args = parser.parse_args()

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as error:
        raise SystemExit(
            "Install the optional SAM dependency first: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from error

    image = np.asarray(Image.open(args.image).convert("RGB"))
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(args.device)
    generator = SamAutomaticMaskGenerator(sam)
    records = [record for record in generator.generate(image) if record["area"] >= args.min_area]
    if not records:
        raise RuntimeError("SAM returned no masks after min-area filtering.")
    masks = np.stack([record["segmentation"] for record in records]).astype(np.uint8)
    areas = np.asarray([record["area"] for record in records])
    stability = np.asarray([record["stability_score"] for record in records])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        masks=masks,
        areas=areas,
        stability_scores=stability,
        image_path=np.asarray(args.image),
    )
    print(f"Wrote {len(masks)} SAM masks to {args.output}")


if __name__ == "__main__":
    main()
