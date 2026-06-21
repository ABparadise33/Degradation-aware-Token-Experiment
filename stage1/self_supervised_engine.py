import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Stage1PairDataset
from .engine import device_from_arg
from .mixed_degradation import (
    apply_random_mixed_degradation,
    sample_mixed_recipe,
    scale_recipe,
)
from .self_supervised_losses import self_supervised_objective
from .self_supervised_model import SelfSupervisedDegradationEncoder


def _slice_output(output: Dict[str, torch.Tensor], start: int, end: int):
    return {
        key: value[start:end]
        for key, value in output.items()
        if torch.is_tensor(value) and value.shape[0] >= end
    }


def build_self_supervised_views(
    raw_image: torch.Tensor,
    reference_image: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
    batch = reference_image.shape[0]
    recipe = sample_mixed_recipe(
        batch,
        reference_image.device,
        reference_image.dtype,
        min_strength=0.15,
        max_strength=0.9,
    )
    different_recipe = sample_mixed_recipe(
        batch,
        reference_image.device,
        reference_image.dtype,
        min_strength=0.15,
        max_strength=0.9,
    )
    recipe_distance = torch.linalg.vector_norm(different_recipe - recipe, dim=1)
    too_similar = recipe_distance < 0.5
    if too_similar.any():
        different_recipe[too_similar] = (1.0 - recipe[too_similar]).clamp(0.15, 0.9)
    mild_scale = reference_image.new_empty(batch).uniform_(0.2, 0.4)
    severe_scale = reference_image.new_empty(batch).uniform_(0.75, 1.0)

    same_a, _, _ = apply_random_mixed_degradation(reference_image, recipe)
    shifted_reference = torch.roll(reference_image, shifts=1, dims=0)
    same_b, _, _ = apply_random_mixed_degradation(shifted_reference, recipe)
    different_b, _, _ = apply_random_mixed_degradation(reference_image, different_recipe)
    mild, _, _ = apply_random_mixed_degradation(
        reference_image,
        scale_recipe(recipe, mild_scale),
    )
    severe, _, _ = apply_random_mixed_degradation(
        reference_image,
        scale_recipe(recipe, severe_scale),
    )

    named_images = [
        ("clean", reference_image),
        ("raw", raw_image),
        ("same_a", same_a),
        ("same_b", same_b),
        ("different_b", different_b),
        ("mild", mild),
        ("severe", severe),
    ]
    offsets: Dict[str, Tuple[int, int]] = {}
    images = []
    cursor = 0
    for name, image in named_images:
        images.append(image)
        offsets[name] = (cursor, cursor + batch)
        cursor += batch
    return torch.cat(images, dim=0), offsets


def forward_self_supervised_views(
    model: SelfSupervisedDegradationEncoder,
    raw_image: torch.Tensor,
    reference_image: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    images, offsets = build_self_supervised_views(raw_image, reference_image)
    combined = model(images)
    outputs = {
        name: _slice_output(combined, start, end)
        for name, (start, end) in offsets.items()
    }
    outputs["different_a"] = outputs["same_a"]
    return outputs


def _batch_diagnostics(outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    same_a = F.normalize(outputs["same_a"]["z_deg"], dim=1)
    same_b = F.normalize(outputs["same_b"]["z_deg"], dim=1)
    logits = same_a @ same_b.T
    top1 = (logits.argmax(dim=1) == torch.arange(logits.shape[0], device=logits.device)).float().mean()

    clean = F.normalize(outputs["clean"]["z_deg"], dim=1)
    mild = F.normalize(outputs["mild"]["z_deg"], dim=1)
    severe = F.normalize(outputs["severe"]["z_deg"], dim=1)
    mild_distance = 1.0 - (clean * mild).sum(dim=1)
    severe_distance = 1.0 - (clean * severe).sum(dim=1)
    severity_accuracy = (severe_distance > mild_distance).float().mean()

    pair_margin = outputs["raw"]["m_deg"].reshape(-1) - outputs["clean"]["m_deg"].reshape(-1)
    return {
        "same_recipe_top1": float(top1.detach().cpu()),
        "severity_order_acc": float(severity_accuracy.detach().cpu()),
        "raw_reference_ranking_acc": float((pair_margin > 0).float().mean().detach().cpu()),
        "raw_reference_margin": float(pair_margin.mean().detach().cpu()),
    }


@torch.no_grad()
def validate_self_supervised(
    model: SelfSupervisedDegradationEncoder,
    loader: DataLoader,
    device: torch.device,
    objective_kwargs: Dict[str, float],
) -> Dict[str, float]:
    model.eval()
    records: Dict[str, List[float]] = {}
    for batch in tqdm(loader, desc="validating self-supervised", leave=False):
        raw_image = batch["raw_image"].to(device)
        reference_image = batch["ref_image"].to(device)
        outputs = forward_self_supervised_views(model, raw_image, reference_image)
        losses = self_supervised_objective(outputs, **objective_kwargs)
        diagnostics = _batch_diagnostics(outputs)
        for key, value in losses.items():
            records.setdefault(f"loss_{key}", []).append(float(value.detach().cpu()))
        for key, value in diagnostics.items():
            records.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(values)) for key, values in records.items()}


def train_self_supervised_encoder(
    labels_csv: str,
    output_dir: str,
    dataset_root: Optional[str] = None,
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    image_size: int = 224,
    latent_dim: int = 128,
    num_slots: int = 4,
    num_heads: int = 4,
    decoder_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device_name: str = "auto",
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
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    device = device_from_arg(device_name)
    print(f"Using device: {device}")

    train_set = Stage1PairDataset(
        labels_csv,
        split="train",
        image_size=image_size,
        train=True,
        dataset_root=dataset_root,
    )
    val_set = Stage1PairDataset(
        labels_csv,
        split="val",
        image_size=image_size,
        train=False,
        dataset_root=dataset_root,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = SelfSupervisedDegradationEncoder(
        backbone=backbone,
        pretrained=pretrained,
        latent_dim=latent_dim,
        num_slots=num_slots,
        num_heads=num_heads,
        decoder_layers=decoder_layers,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    objective_kwargs = {
        "temperature": temperature,
        "separation_margin": separation_margin,
        "order_margin": order_margin,
        "ranking_margin": ranking_margin,
        "lambda_same": lambda_same,
        "lambda_different": lambda_different,
        "lambda_order": lambda_order,
        "lambda_pair": lambda_pair,
        "lambda_variance": lambda_variance,
        "lambda_covariance": lambda_covariance,
        "lambda_slot_diversity": lambda_slot_diversity,
    }
    config = {
        "architecture": "self_supervised_slots",
        "backbone": backbone,
        "image_size": image_size,
        "latent_dim": latent_dim,
        "num_slots": num_slots,
        "num_heads": num_heads,
        "decoder_layers": decoder_layers,
        "dropout": dropout,
        "freeze_backbone": freeze_backbone,
        **objective_kwargs,
    }

    log_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_same_recipe_top1",
        "val_severity_order_acc",
        "val_raw_reference_ranking_acc",
        "val_raw_reference_margin",
    ]
    best_metric = float("inf")
    best_path = os.path.join(output_dir, "best_self_supervised_encoder.pt")
    log_path = os.path.join(output_dir, "training_log.csv")

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses: List[float] = []
            for batch in tqdm(train_loader, desc=f"selfsup epoch {epoch}/{epochs}", leave=False):
                raw_image = batch["raw_image"].to(device)
                reference_image = batch["ref_image"].to(device)
                outputs = forward_self_supervised_views(model, raw_image, reference_image)
                losses = self_supervised_objective(outputs, **objective_kwargs)
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
                epoch_losses.append(float(losses["total"].detach().cpu()))

            val = validate_self_supervised(model, val_loader, device, objective_kwargs)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_loss": val["loss_total"],
                "val_same_recipe_top1": val["same_recipe_top1"],
                "val_severity_order_acc": val["severity_order_acc"],
                "val_raw_reference_ranking_acc": val["raw_reference_ranking_acc"],
                "val_raw_reference_margin": val["raw_reference_margin"],
            }
            writer.writerow(row)
            log_file.flush()
            print(row)

            checkpoint = {
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_metrics": val,
            }
            torch.save(checkpoint, os.path.join(output_dir, "last_self_supervised_encoder.pt"))
            if val["loss_total"] < best_metric:
                best_metric = val["loss_total"]
                torch.save(checkpoint, best_path)
    return best_path
