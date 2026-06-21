import torch
import torch.nn.functional as F

from .synthetic import denormalize


def underwater_dark_channel(image: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    """Approximate an underwater dark channel from green/blue channels."""
    if image.min() < 0.0 or image.max() > 1.0:
        image = denormalize(image)
    green_blue_min = image[:, 1:3].amin(dim=1, keepdim=True)
    return -F.max_pool2d(
        -green_blue_min,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )


def transmission_proxy(
    image: torch.Tensor,
    patch_size: int = 15,
    omega: float = 0.95,
) -> torch.Tensor:
    dark = underwater_dark_channel(image, patch_size=patch_size)
    atmospheric = dark.flatten(2).quantile(0.99, dim=2).view(-1, 1, 1, 1).clamp_min(1e-3)
    return (1.0 - omega * dark / atmospheric).clamp(0.0, 1.0)


def visibility_score_from_transmission(transmission: torch.Tensor) -> torch.Tensor:
    return 1.0 - transmission.mean(dim=(1, 2, 3))
