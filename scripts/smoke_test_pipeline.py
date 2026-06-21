import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from stage1.model import DegradationAssessor, TaskAwareDegradationAssessor
from stage1.regions import compose_region_maps, masked_average_pool
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
