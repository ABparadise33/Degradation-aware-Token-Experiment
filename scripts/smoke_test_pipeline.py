import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from stage1.model import DegradationAssessor, TaskAwareDegradationAssessor
from stage1.mixed_degradation import apply_random_mixed_degradation, sample_mixed_recipe
from stage1.regions import compose_region_maps, masked_average_pool
from stage1.self_supervised_engine import forward_self_supervised_views
from stage1.self_supervised_losses import self_supervised_objective
from stage1.self_supervised_model import SelfSupervisedDegradationEncoder
from stage1.synthetic import synthetic_representation_losses
from stage1.synthetic import (
    attention_diversity_loss,
    synthetic_blur_losses,
    task_token_diversity_loss,
)


def main():
    image = torch.randn(2, 3, 64, 64)

    token_model = DegradationAssessor(
        backbone="resnet18",
        pretrained=False,
        latent_dim=32,
        score_from_token=True,
    )
    output = token_model(image)
    output["scores"].sum().backward()
    token_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in token_model.token_head.parameters()
        if parameter.grad is not None
    )
    assert token_gradient > 0.0
    assert output["scores"].shape == (2, 5)
    assert output["z_deg"].shape == (2, 32)

    task_model = TaskAwareDegradationAssessor(
        backbone="resnet18",
        pretrained=False,
        latent_dim=32,
        num_heads=4,
    )
    task_output = task_model(image)
    assert task_output["task_tokens"].shape == (2, 5, 32)
    assert task_output["attention_maps"].shape[:2] == (2, 5)
    contrastive, order = synthetic_representation_losses(task_model, image)
    blur_regression, blur_order = synthetic_blur_losses(task_model, image)
    diversity = task_token_diversity_loss(task_output["task_tokens"])
    attention_diversity = attention_diversity_loss(task_output["attention_maps"])
    assert torch.isfinite(
        contrastive + order + blur_regression + blur_order + diversity + attention_diversity
    )

    selfsup_model = SelfSupervisedDegradationEncoder(
        backbone="resnet18",
        pretrained=False,
        latent_dim=32,
        num_slots=4,
        num_heads=4,
    )
    recipe = sample_mixed_recipe(2, image.device, image.dtype)
    degraded, returned_recipe, severity = apply_random_mixed_degradation(image, recipe)
    assert degraded.shape == image.shape
    assert returned_recipe.shape == (2, 5)
    assert severity.shape == (2,)
    selfsup_output = selfsup_model(image)
    assert selfsup_output["z_deg"].shape == (2, 32)
    assert selfsup_output["slot_tokens"].shape == (2, 4, 32)
    assert selfsup_output["slot_attention"].shape[:2] == (2, 4)
    assert selfsup_output["m_deg"].shape == (2, 1)
    selfsup_views = forward_self_supervised_views(selfsup_model, image, image)
    selfsup_losses = self_supervised_objective(selfsup_views)
    assert torch.isfinite(selfsup_losses["total"])
    selfsup_losses["total"].backward()

    feature = torch.randn(2, 16, 8, 8)
    masks = torch.zeros(2, 3, 64, 64)
    masks[:, 0, :32, :32] = 1
    masks[:, 1, :32, 32:] = 1
    masks[:, 2, 32:, :] = 1
    regions = masked_average_pool(feature, masks)
    maps = compose_region_maps(torch.rand(2, 3, 4), masks, (64, 64))
    assert regions.shape == (2, 3, 16)
    assert maps.shape == (2, 4, 64, 64)
    print("Stage 1 pipeline smoke test passed.")


if __name__ == "__main__":
    main()
