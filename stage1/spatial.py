from typing import Dict, List, Tuple

import timm
import torch
from torch import nn

from .model import load_assessor


class SpatialBackbone(nn.Module):
    """Expose timm feature-info stages from a trained assessor backbone."""

    def __init__(self, backbone: nn.Module, module_names: List[str]):
        super().__init__()
        self.backbone = backbone
        modules: Dict[str, nn.Module] = dict(backbone.named_modules())
        missing = [name for name in module_names if name not in modules]
        if missing:
            raise ValueError(f"Backbone does not expose requested feature modules: {missing}")
        self.outputs: Dict[str, torch.Tensor] = {}
        self.handles = [
            modules[name].register_forward_hook(self._capture(name))
            for name in module_names
        ]
        self.module_names = module_names

    def _capture(self, name: str):
        def hook(_, __, output):
            self.outputs[name] = output

        return hook

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        self.outputs.clear()
        self.backbone(image)
        return [self.outputs[name] for name in self.module_names]


def load_spatial_backbone(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, List[dict]]:
    assessor = load_assessor(checkpoint_path, device)
    feature_probe = timm.create_model(
        assessor.backbone_name,
        pretrained=False,
        features_only=True,
    )
    feature_info = [
        {
            "index": index,
            "channels": int(info["num_chs"]),
            "reduction": int(info["reduction"]),
            "module": str(info["module"]),
        }
        for index, info in enumerate(feature_probe.feature_info.get_dicts())
    ]
    model = SpatialBackbone(
        assessor.backbone,
        [item["module"] for item in feature_info],
    ).to(device).eval()
    return model, feature_info
