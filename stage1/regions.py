from typing import Tuple

import torch
import torch.nn.functional as F


def resize_masks(masks: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)
    return F.interpolate(masks.float(), size=size, mode="nearest")


def masked_average_pool(feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Pool [B,C,H,W] features into [B,K,C] using [B,K,H,W] masks."""
    masks = resize_masks(masks, feature_map.shape[-2:])
    numerator = torch.einsum("bchw,bkhw->bkc", feature_map, masks)
    denominator = masks.sum(dim=(2, 3), keepdim=False).clamp_min(1e-6).unsqueeze(-1)
    return numerator / denominator


def compose_region_maps(
    region_scores: torch.Tensor,
    masks: torch.Tensor,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Compose overlapping region scores into dense [B,T,H,W] maps."""
    masks = resize_masks(masks, output_size)
    weighted = torch.einsum("bkt,bkhw->bthw", region_scores, masks)
    coverage = masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weighted / coverage
