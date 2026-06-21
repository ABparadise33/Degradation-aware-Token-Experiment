import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Stage1ImageDataset, Stage1PairDataset
from .metrics import ranking_metrics, regression_metrics
from .model import DegradationAssessor, TaskAwareDegradationAssessor, load_assessor
from .pseudo_labels import SCORE_COLUMNS
from .synthetic import synthetic_representation_losses


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is not available. Verify that PyTorch is running on an "
            "Apple Silicon Mac with an MPS-enabled build."
        )
    return torch.device(name)


def train_assessor(
    labels_csv: str,
    output_dir: str,
    backbone: str,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    image_size: int = 224,
    latent_dim: int = 128,
    dropout: float = 0.2,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    rank_margin: float = 0.1,
    lambda_rank: float = 1.0,
    num_workers: int = 4,
    device_name: str = "auto",
    dataset_root: Optional[str] = None,
    score_from_token: bool = True,
    architecture: str = "token_mlp",
    num_heads: int = 4,
    decoder_layers: int = 1,
    lambda_contrast: float = 0.0,
    lambda_order: float = 0.0,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    device = device_from_arg(device_name)
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"

    train_set = Stage1PairDataset(
        labels_csv,
        split="train",
        image_size=image_size,
        train=True,
        dataset_root=dataset_root,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if architecture == "task_attention":
        model = TaskAwareDegradationAssessor(
            backbone=backbone,
            pretrained=pretrained,
            latent_dim=latent_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            num_heads=num_heads,
            decoder_layers=decoder_layers,
        ).to(device)
    else:
        model = DegradationAssessor(
            backbone=backbone,
            pretrained=pretrained,
            latent_dim=latent_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            score_from_token=score_from_token,
        ).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)

    config = {
        "backbone": backbone,
        "latent_dim": latent_dim,
        "dropout": dropout,
        "image_size": image_size,
        "score_columns": SCORE_COLUMNS,
        "freeze_backbone": freeze_backbone,
        "score_from_token": score_from_token,
        "architecture_version": 3 if architecture == "task_attention" else (2 if score_from_token else 1),
        "architecture": architecture,
        "num_heads": num_heads,
        "decoder_layers": decoder_layers,
        "lambda_contrast": lambda_contrast,
        "lambda_order": lambda_order,
    }
    best_metric = float("inf")
    best_path = os.path.join(output_dir, "best_stage1_assessor.pt")
    log_path = os.path.join(output_dir, "training_log.csv")

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_score_mae", "val_ranking_acc"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            losses: List[float] = []
            for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
                raw_image = batch["raw_image"].to(device)
                ref_image = batch["ref_image"].to(device)
                raw_scores = batch["raw_scores"].to(device)
                ref_scores = batch["ref_scores"].to(device)

                raw_out = model(raw_image)
                ref_out = model(ref_image)
                score_loss = F.smooth_l1_loss(raw_out["scores"], raw_scores) + F.smooth_l1_loss(ref_out["scores"], ref_scores)
                rank_loss = F.relu(rank_margin - (ref_out["scores"][:, 4] - raw_out["scores"][:, 4])).mean()
                loss = score_loss + lambda_rank * rank_loss
                if lambda_contrast > 0.0 or lambda_order > 0.0:
                    contrast_loss, order_loss = synthetic_representation_losses(model, ref_image)
                    loss = loss + lambda_contrast * contrast_loss + lambda_order * order_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_metrics = evaluate_assessor(
                model=model,
                labels_csv=labels_csv,
                split="val",
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                dataset_root=dataset_root,
            )
            val_mae = float(np.mean([val_metrics[f"{name}_mae"] for name in SCORE_COLUMNS]))
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_score_mae": val_mae,
                    "val_ranking_acc": val_metrics["ranking_acc"],
                }
            )
            log_file.flush()

            checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_metrics": val_metrics}
            torch.save(checkpoint, os.path.join(output_dir, "last_stage1_assessor.pt"))
            if val_mae < best_metric:
                best_metric = val_mae
                torch.save(checkpoint, best_path)

    return best_path


@torch.no_grad()
def evaluate_assessor(
    labels_csv: str,
    split: Optional[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: Optional[DegradationAssessor] = None,
    checkpoint_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_npz: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> Dict[str, float]:
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided.")
        model = load_assessor(checkpoint_path, device=device)
    model.eval()

    dataset = Stage1ImageDataset(
        labels_csv=labels_csv,
        split=split,
        image_size=image_size,
        dataset_root=dataset_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    targets: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    tokens: List[np.ndarray] = []
    task_tokens: List[np.ndarray] = []
    attention_maps: List[np.ndarray] = []
    records: List[Tuple[str, str, float]] = []
    csv_rows: List[Dict[str, object]] = []

    for batch in tqdm(loader, desc="evaluating", leave=False):
        image = batch["image"].to(device)
        out = model(image)
        pred = out["scores"].detach().cpu().numpy()
        token = out["z_deg"].detach().cpu().numpy()
        target = batch["scores"].numpy()
        targets.append(target)
        preds.append(pred)
        tokens.append(token)
        if "task_tokens" in out:
            task_tokens.append(out["task_tokens"].detach().cpu().numpy())
        if "attention_maps" in out:
            attention_maps.append(out["attention_maps"].detach().cpu().numpy())
        for i in range(pred.shape[0]):
            pair_id = batch["pair_id"][i]
            role = batch["role"][i]
            records.append((pair_id, role, float(pred[i, 4])))
            row = {
                "image_path": batch["image_path"][i],
                "image_name": batch["image_name"][i],
                "pair_id": pair_id,
                "role": role,
                "split": batch["split"][i],
            }
            for j, name in enumerate(SCORE_COLUMNS):
                row[f"{name}_target"] = float(target[i, j])
                row[f"{name}_pred"] = float(pred[i, j])
            csv_rows.append(row)

    targets_arr = np.concatenate(targets, axis=0)
    preds_arr = np.concatenate(preds, axis=0)
    tokens_arr = np.concatenate(tokens, axis=0)
    metrics = regression_metrics(targets_arr, preds_arr)
    metrics.update(ranking_metrics(records))

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    if output_npz:
        os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
        arrays = dict(
            z_deg=tokens_arr,
            preds=preds_arr,
            targets=targets_arr,
            image_paths=np.asarray([row["image_path"] for row in csv_rows]),
            pair_ids=np.asarray([row["pair_id"] for row in csv_rows]),
            roles=np.asarray([row["role"] for row in csv_rows]),
            score_columns=np.asarray(SCORE_COLUMNS),
        )
        if task_tokens:
            arrays["task_tokens"] = np.concatenate(task_tokens, axis=0)
        if attention_maps:
            arrays["attention_maps"] = np.concatenate(attention_maps, axis=0)
        np.savez_compressed(output_npz, **arrays)

    return metrics
