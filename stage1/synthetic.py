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
    outputs = []
    for index in range(image.shape[0]):
        value = float(severity.reshape(-1)[index].item())
        if value <= 1e-6:
            outputs.append(image[index : index + 1])
            continue
        sigma = 0.4 + 3.0 * value
        size = max(3, int(round(sigma * 4)) | 1)
        kernel = gaussian_kernel(size, sigma, image.device, image.dtype)
        weight = kernel.expand(3, 1, size, size)
        padded = F.pad(image[index : index + 1], (size // 2,) * 4, mode="reflect")
        outputs.append(F.conv2d(padded, weight, groups=3))
    return torch.cat(outputs, dim=0)


def motion_kernel(size: int, direction: int, device, dtype) -> torch.Tensor:
    kernel = torch.zeros((size, size), device=device, dtype=dtype)
    if direction == 0:
        kernel[size // 2, :] = 1.0
    elif direction == 1:
        kernel[:, size // 2] = 1.0
    elif direction == 2:
        kernel.fill_diagonal_(1.0)
    else:
        kernel = torch.flip(torch.eye(size, device=device, dtype=dtype), dims=(1,))
    return kernel / kernel.sum()


def apply_motion_blur(
    image: torch.Tensor,
    severity: torch.Tensor,
    direction: Optional[int] = None,
) -> torch.Tensor:
    outputs = []
    for index in range(image.shape[0]):
        value = float(severity.reshape(-1)[index].item())
        if value <= 1e-6:
            outputs.append(image[index : index + 1])
            continue
        length = max(3, int(round(3 + 16 * value)) | 1)
        selected_direction = (
            direction
            if direction is not None
            else int(torch.randint(0, 4, (1,), device=image.device).item())
        )
        kernel = motion_kernel(length, selected_direction, image.device, image.dtype)
        weight = kernel.expand(3, 1, length, length)
        padded = F.pad(image[index : index + 1], (length // 2,) * 4, mode="reflect")
        outputs.append(F.conv2d(padded, weight, groups=3))
    return torch.cat(outputs, dim=0)


def task_token_diversity_loss(task_tokens: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(task_tokens, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(similarity.shape[1], device=similarity.device, dtype=similarity.dtype)
    return ((similarity - identity) ** 2).mean()


def attention_diversity_loss(attention_maps: torch.Tensor) -> torch.Tensor:
    flattened = F.normalize(attention_maps.flatten(2), dim=-1)
    similarity = flattened @ flattened.transpose(1, 2)
    identity = torch.eye(similarity.shape[1], device=similarity.device, dtype=similarity.dtype)
    return ((similarity - identity) ** 2).mean()


def apply_synthetic_blur(
    normalized_image: torch.Tensor,
    severity: torch.Tensor,
    blur_type: Optional[str] = None,
    motion_direction: Optional[int] = None,
) -> Tuple[torch.Tensor, str]:
    image = denormalize(normalized_image)
    if blur_type is None or blur_type == "random":
        blur_type = "gaussian" if bool(torch.randint(0, 2, (1,), device=image.device).item()) else "motion"
    if blur_type == "gaussian":
        blurred = apply_blur(image, severity)
    elif blur_type == "motion":
        blurred = apply_motion_blur(image, severity, direction=motion_direction)
    else:
        raise ValueError(f"Unsupported blur type: {blur_type}")
    return normalize(blurred), blur_type


def synthetic_blur_losses(
    model,
    clean_image: torch.Tensor,
    margin: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Supervise blur using known mild/severe Gaussian or motion blur ordering."""
    batch = clean_image.shape[0]
    mild_severity = clean_image.new_empty(batch).uniform_(0.15, 0.35)
    severe_severity = clean_image.new_empty(batch).uniform_(0.65, 0.95)
    blur_type = "gaussian" if bool(torch.randint(0, 2, (1,), device=clean_image.device).item()) else "motion"
    motion_direction = (
        int(torch.randint(0, 4, (1,), device=clean_image.device).item())
        if blur_type == "motion"
        else None
    )
    mild, _ = apply_synthetic_blur(clean_image, mild_severity, blur_type, motion_direction)
    severe, _ = apply_synthetic_blur(clean_image, severe_severity, blur_type, motion_direction)
    mild_prediction = model(mild)["scores"][:, 1]
    severe_prediction = model(severe)["scores"][:, 1]
    predicted_gap = severe_prediction - mild_prediction
    target_gap = severe_severity - mild_severity
    regression = F.smooth_l1_loss(predicted_gap, target_gap)
    order = F.relu(margin - predicted_gap).mean()
    return regression, order


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
