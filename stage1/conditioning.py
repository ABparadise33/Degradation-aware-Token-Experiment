from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_stage1_conditions(path: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=True)
    device = device or torch.device("cpu")
    output = {}
    for key in (
        "global_scores",
        "global_z_deg",
        "global_task_tokens",
        "global_attention_maps",
        "region_scores",
        "region_tokens",
        "region_task_tokens",
        "region_maps",
        "masks",
    ):
        if key in data:
            output[key] = torch.from_numpy(data[key]).float().to(device)
    return output


def build_spatial_condition(
    image: torch.Tensor,
    region_maps: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Concatenate image, degradation maps, and optional mask coverage for Stage 2."""
    size = output_size or image.shape[-2:]
    image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    region_maps = F.interpolate(region_maps, size=size, mode="bilinear", align_corners=False)
    tensors = [image, region_maps]
    if masks is not None:
        mask_coverage = masks.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        mask_coverage = F.interpolate(mask_coverage, size=size, mode="nearest")
        tensors.append(mask_coverage)
    return torch.cat(tensors, dim=1)
