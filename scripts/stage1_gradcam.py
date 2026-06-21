import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from stage1.data import build_transform
from stage1.engine import device_from_arg
from stage1.model import load_assessor
from stage1.pseudo_labels import SCORE_COLUMNS


def module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"Layer '{name}' not found. Use --list-layers to inspect available layer names.")
    return modules[name]


def find_last_conv(model: nn.Module) -> tuple[str, nn.Module]:
    last_name: Optional[str] = None
    last_module: Optional[nn.Module] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    if last_name is None or last_module is None:
        raise ValueError("Could not auto-detect a Conv2d layer. Pass --target-layer explicitly.")
    return last_name, last_module


def overlay_cam(image: Image.Image, cam: np.ndarray, output_path: str) -> None:
    cam_img = Image.fromarray(np.uint8(cam * 255.0)).resize(image.size, Image.Resampling.BICUBIC)
    heat = plt.get_cmap("jet")(np.asarray(cam_img) / 255.0)[:, :, :3]
    base = np.asarray(image).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.imsave(output_path, overlay)


def create_gradcam(
    checkpoint: str,
    image_path: str,
    output_path: str,
    score_name: str,
    device_name: str = "auto",
    target_layer_name: Optional[str] = None,
    image_size: int = 224,
) -> str:
    device = device_from_arg(device_name)
    model = load_assessor(checkpoint, device=device)
    layer_name, target_layer = (
        (target_layer_name, module_by_name(model, target_layer_name))
        if target_layer_name
        else find_last_conv(model)
    )
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(image_size, train=False)(image).unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    target = model(tensor)["scores"][0, SCORE_COLUMNS.index(score_name)]
    target.backward()
    handle_fwd.remove()
    handle_bwd.remove()

    if not activations or not gradients:
        raise RuntimeError(f"No Grad-CAM activations captured for layer '{layer_name}'.")
    act = activations[-1][0]
    grad = gradients[-1][0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=0))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    overlay_cam(image, cam.detach().cpu().numpy(), output_path)
    return layer_name


def main():
    parser = argparse.ArgumentParser(description="Create Grad-CAM for a Stage 1 predicted degradation score.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score", default="s_visibility_proxy", choices=SCORE_COLUMNS)
    parser.add_argument("--target-layer", default=None, help="Optional model module name to hook.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--list-layers", action="store_true")
    args = parser.parse_args()

    if args.list_layers:
        device = device_from_arg(args.device)
        model = load_assessor(args.checkpoint, device=device)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(name)
        return

    layer_name = create_gradcam(
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        score_name=args.score,
        device_name=args.device,
        target_layer_name=args.target_layer,
        image_size=args.image_size,
    )
    print(f"Wrote {args.score} Grad-CAM from layer '{layer_name}' to {args.output}")


if __name__ == "__main__":
    main()
