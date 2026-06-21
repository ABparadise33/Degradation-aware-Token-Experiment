from typing import Dict, Optional, Tuple

import torch

from .synthetic import apply_blur, denormalize, normalize


RECIPE_NAMES = (
    "color_strength",
    "blur_strength",
    "contrast_strength",
    "visibility_strength",
    "noise_strength",
)


def sample_mixed_recipe(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    min_strength: float = 0.0,
    max_strength: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    recipe = torch.rand(
        batch_size,
        len(RECIPE_NAMES),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    recipe = min_strength + (max_strength - min_strength) * recipe
    active = torch.rand(
        batch_size,
        len(RECIPE_NAMES),
        device=device,
        dtype=dtype,
        generator=generator,
    ) > 0.35
    recipe = recipe * active
    empty = recipe.sum(dim=1) <= 1e-6
    if empty.any():
        selected = torch.randint(
            0,
            len(RECIPE_NAMES),
            (int(empty.sum()),),
            device=device,
            generator=generator,
        )
        recipe[empty] = 0.0
        recipe[empty, selected] = max(min_strength, 0.25)
    return recipe


def recipe_severity(recipe: torch.Tensor) -> torch.Tensor:
    """Generic magnitude; recipe dimensions remain anonymous to the encoder."""
    return torch.sqrt(torch.mean(recipe.clamp(0.0, 1.0) ** 2, dim=1))


def scale_recipe(recipe: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 1:
        scale = scale[:, None]
    return (recipe * scale).clamp(0.0, 1.0)


def apply_random_mixed_degradation(
    normalized_image: torch.Tensor,
    recipe: Optional[torch.Tensor] = None,
    noise_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a continuous mixture without exposing degradation-type labels."""
    batch = normalized_image.shape[0]
    if recipe is None:
        recipe = sample_mixed_recipe(
            batch,
            normalized_image.device,
            normalized_image.dtype,
        )
    if recipe.shape != (batch, len(RECIPE_NAMES)):
        raise ValueError(
            f"Expected recipe shape {(batch, len(RECIPE_NAMES))}, got {tuple(recipe.shape)}"
        )
    recipe = recipe.to(device=normalized_image.device, dtype=normalized_image.dtype).clamp(0.0, 1.0)
    image = denormalize(normalized_image)

    color = recipe[:, 0].view(batch, 1, 1, 1)
    color_direction = image.new_tensor([0.55, -0.30, -0.50]).view(1, 3, 1, 1)
    image = image * (1.0 + color * color_direction)

    image = apply_blur(image.clamp(0.0, 1.0), recipe[:, 1])

    contrast = recipe[:, 2].view(batch, 1, 1, 1)
    luminance = image.mean(dim=(1, 2, 3), keepdim=True)
    image = luminance + (image - luminance) * (1.0 - 0.75 * contrast)

    visibility = recipe[:, 3].view(batch, 1, 1, 1)
    water_light = image.new_tensor([0.05, 0.45, 0.65]).view(1, 3, 1, 1)
    transmission = 1.0 - 0.75 * visibility
    image = image * transmission + water_light * (1.0 - transmission)

    noise = recipe[:, 4].view(batch, 1, 1, 1)
    if noise_seed is None:
        random_noise = torch.randn_like(image)
    else:
        generator = torch.Generator(device=image.device)
        generator.manual_seed(noise_seed)
        random_noise = torch.randn(
            image.shape,
            device=image.device,
            dtype=image.dtype,
            generator=generator,
        )
    image = image + random_noise * (0.08 * noise)

    return normalize(image.clamp(0.0, 1.0)), recipe, recipe_severity(recipe)


def recipe_dict(recipe: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {name: recipe[:, index] for index, name in enumerate(RECIPE_NAMES)}
