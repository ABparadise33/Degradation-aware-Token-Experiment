import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from PIL import Image

from stage1.data import build_transform
from stage1.visibility import transmission_proxy, visibility_score_from_transmission


def main():
    parser = argparse.ArgumentParser(description="Export an underwater dark-channel transmission proxy.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=15)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    tensor = build_transform(args.image_size, train=False)(image).unsqueeze(0)
    transmission = transmission_proxy(tensor, patch_size=args.patch_size)
    score = visibility_score_from_transmission(transmission)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.imsave(args.output, transmission[0, 0].numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    print(f"s_visibility_dcp_proxy: {float(score[0])}")
    print(f"Wrote transmission proxy to {args.output}")


if __name__ == "__main__":
    main()
