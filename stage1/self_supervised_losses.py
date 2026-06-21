from typing import Dict

import torch
import torch.nn.functional as F


def symmetric_info_nce(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def different_degradation_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    distance = 1.0 - F.cosine_similarity(first, second, dim=1)
    return F.relu(margin - distance).mean()


def severity_order_loss(
    clean: torch.Tensor,
    mild: torch.Tensor,
    severe: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    clean = F.normalize(clean, dim=1)
    mild = F.normalize(mild, dim=1)
    severe = F.normalize(severe, dim=1)
    mild_distance = 1.0 - (clean * mild).sum(dim=1)
    severe_distance = 1.0 - (clean * severe).sum(dim=1)
    return F.relu(margin - (severe_distance - mild_distance)).mean()


def raw_reference_order_loss(
    raw_magnitude: torch.Tensor,
    reference_magnitude: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    gap = raw_magnitude.reshape(-1) - reference_magnitude.reshape(-1)
    return F.relu(margin - gap).mean()


def variance_loss(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros(())
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / (z.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag(diagonal)
    return off_diagonal.pow(2).sum() / z.shape[1]


def slot_diversity_loss(slot_tokens: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(slot_tokens, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(
        similarity.shape[1],
        device=similarity.device,
        dtype=similarity.dtype,
    )
    return ((similarity - identity) ** 2).mean()


def self_supervised_objective(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    temperature: float = 0.1,
    separation_margin: float = 0.5,
    order_margin: float = 0.1,
    ranking_margin: float = 0.1,
    lambda_same: float = 1.0,
    lambda_different: float = 0.5,
    lambda_order: float = 0.5,
    lambda_pair: float = 0.5,
    lambda_variance: float = 0.01,
    lambda_covariance: float = 0.01,
    lambda_slot_diversity: float = 0.01,
) -> Dict[str, torch.Tensor]:
    losses = {
        "same": symmetric_info_nce(
            outputs["same_a"]["z_deg"],
            outputs["same_b"]["z_deg"],
            temperature=temperature,
        ),
        "different": different_degradation_loss(
            outputs["different_a"]["z_deg"],
            outputs["different_b"]["z_deg"],
            margin=separation_margin,
        ),
        "order": severity_order_loss(
            outputs["clean"]["z_deg"],
            outputs["mild"]["z_deg"],
            outputs["severe"]["z_deg"],
            margin=order_margin,
        ),
        "pair": raw_reference_order_loss(
            outputs["raw"]["m_deg"],
            outputs["clean"]["m_deg"],
            margin=ranking_margin,
        ),
    }
    representation = torch.cat(
        [
            outputs["same_a"]["z_deg"],
            outputs["same_b"]["z_deg"],
            outputs["different_a"]["z_deg"],
            outputs["different_b"]["z_deg"],
            outputs["mild"]["z_deg"],
            outputs["severe"]["z_deg"],
        ],
        dim=0,
    )
    losses["variance"] = variance_loss(representation)
    losses["covariance"] = covariance_loss(representation)
    losses["slot_diversity"] = slot_diversity_loss(
        torch.cat(
            [
                outputs["same_a"]["slot_tokens"],
                outputs["same_b"]["slot_tokens"],
            ],
            dim=0,
        )
    )
    losses["total"] = (
        lambda_same * losses["same"]
        + lambda_different * losses["different"]
        + lambda_order * losses["order"]
        + lambda_pair * losses["pair"]
        + lambda_variance * losses["variance"]
        + lambda_covariance * losses["covariance"]
        + lambda_slot_diversity * losses["slot_diversity"]
    )
    return losses
