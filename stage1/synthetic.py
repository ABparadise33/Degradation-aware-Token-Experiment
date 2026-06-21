from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def normalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image.clamp(0.0, 1.0) - mean) / std


def gaussian_kernel(size: int, sigma: float, device, dtype):
    axis = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def apply_blur(image: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
    # A batch-shared kernel keeps this augmentation inexpensive and differentiable.
    sigma = float(0.4 + 3.0 * severity.mean().item())
    size = max(3, int(round(sigma * 4)) | 1)
    kernel = gaussian_kernel(size, sigma, image.device, image.dtype)
    weight = kernel.expand(3, 1, size, size)
    return F.conv2d(image, weight, padding=size // 2, groups=3)


def apply_synthetic_degradation(
    normalized_image: torch.Tensor,
    severity: torch.Tensor,
    degradation_type: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply color, blur, contrast, or visibility degradation to each image."""
    image = denormalize(normalized_image)
    batch = image.shape[0]
    severity = severity.reshape(batch, 1, 1, 1).clamp(0.0, 1.0)
    if degradation_type is None:
        degradation_type = torch.randint(0, 4, (batch,), device=image.device)

    color_scale = torch.cat(
        [1.0 + 0.5 * severity, 1.0 - 0.35 * severity, 1.0 - 0.55 * severity],
        dim=1,
    )
    color = image * color_scale
    blur = apply_blur(image, severity)
    luminance = image.mean(dim=(1, 2, 3), keepdim=True)
    contrast = luminance + (image - luminance) * (1.0 - 0.75 * severity)
    water_light = image.new_tensor([0.05, 0.45, 0.65]).view(1, 3, 1, 1)
    transmission = 1.0 - 0.75 * severity
    visibility = image * transmission + water_light * (1.0 - transmission)
    candidates = torch.stack([color, blur, contrast, visibility], dim=1)
    selected = candidates[torch.arange(batch, device=image.device), degradation_type]
    return normalize(selected), degradation_type


def synthetic_representation_losses(
    model,
    clean_image: torch.Tensor,
    order_margin: float = 0.1,
    temperature: float = 0.1,
):
    """Create mild/severe samples and return InfoNCE plus severity-order losses."""
    batch = clean_image.shape[0]
    degradation_type = torch.randint(0, 4, (batch,), device=clean_image.device)
    mild, _ = apply_synthetic_degradation(
        clean_image,
        clean_image.new_full((batch,), 0.25),
        degradation_type,
    )
    severe, _ = apply_synthetic_degradation(
        clean_image,
        clean_image.new_full((batch,), 0.75),
        degradation_type,
    )
    clean_z = F.normalize(model(clean_image)["z_deg"], dim=1)
    mild_z = F.normalize(model(mild)["z_deg"], dim=1)
    severe_z = F.normalize(model(severe)["z_deg"], dim=1)

    logits = mild_z @ severe_z.T / temperature
    labels = torch.arange(batch, device=clean_image.device)
    contrastive = F.cross_entropy(logits, labels)
    mild_distance = 1.0 - (clean_z * mild_z).sum(dim=1)
    severe_distance = 1.0 - (clean_z * severe_z).sum(dim=1)
    order = F.relu(order_margin - (severe_distance - mild_distance)).mean()
    return contrastive, order
