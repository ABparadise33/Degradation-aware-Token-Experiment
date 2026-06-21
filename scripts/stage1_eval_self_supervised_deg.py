import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1.data import Stage1ImageDataset, Stage1PairDataset
from stage1.engine import device_from_arg
from stage1.mixed_degradation import (
    RECIPE_NAMES,
    apply_random_mixed_degradation,
    sample_mixed_recipe,
    scale_recipe,
)
from stage1.pseudo_labels import SCORE_COLUMNS
from stage1.self_supervised_model import load_self_supervised_encoder


def write_metric_csv(path: str, metrics: Dict[str, float]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in sorted(metrics.items()):
            writer.writerow({"metric": key, "value": value})


@torch.no_grad()
def encode_batches(model, images: torch.Tensor, batch_size: int) -> Dict[str, np.ndarray]:
    collected: Dict[str, List[np.ndarray]] = {}
    for start in range(0, len(images), batch_size):
        output = model(images[start : start + batch_size])
        for key in ("z_deg", "slot_tokens", "m_deg"):
            collected.setdefault(key, []).append(output[key].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def cosine_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return 1.0 - np.sum(first * second, axis=-1)


def cross_validated_ridge_prediction(
    features: np.ndarray,
    target: np.ndarray,
    folds: KFold,
    alpha: float = 1.0,
) -> np.ndarray:
    prediction = np.empty(len(features), dtype=np.float64)
    for train_index, test_index in folds.split(features):
        train_x = torch.from_numpy(features[train_index]).double()
        test_x = torch.from_numpy(features[test_index]).double()
        train_y = torch.from_numpy(target[train_index]).double()
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0, unbiased=False).clamp_min(1e-8)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std
        train_x = torch.cat([train_x, torch.ones(len(train_x), 1, dtype=train_x.dtype)], dim=1)
        test_x = torch.cat([test_x, torch.ones(len(test_x), 1, dtype=test_x.dtype)], dim=1)
        identity = torch.eye(train_x.shape[1], dtype=train_x.dtype)
        identity[-1, -1] = 0.0
        weights = torch.linalg.solve(
            train_x.T @ train_x + alpha * identity,
            train_x.T @ train_y,
        )
        prediction[test_index] = (test_x @ weights).numpy()
    return prediction


def retrieval_metrics(
    features: np.ndarray,
    recipe_ids: np.ndarray,
    content_ids: np.ndarray,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    features = np.asarray(features, dtype=np.float64)
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    similarities = normalized @ normalized.T
    average_precisions = []
    top1 = []
    top5 = []
    recall5 = []
    rows = []
    for index in range(len(features)):
        valid = content_ids != content_ids[index]
        candidates = np.flatnonzero(valid)
        order = candidates[np.argsort(-similarities[index, candidates])]
        relevant = recipe_ids[order] == recipe_ids[index]
        total_relevant = int(np.sum(recipe_ids[valid] == recipe_ids[index]))
        if total_relevant == 0:
            continue
        ranks = np.flatnonzero(relevant) + 1
        precision = np.arange(1, len(ranks) + 1) / ranks
        average_precision = float(precision.mean())
        top1_hit = float(relevant[:1].any())
        top5_hit = float(relevant[:5].any())
        recall_at_5 = float(relevant[:5].sum() / total_relevant)
        average_precisions.append(average_precision)
        top1.append(top1_hit)
        top5.append(top5_hit)
        recall5.append(recall_at_5)
        rows.append(
            {
                "query_index": index,
                "recipe_id": int(recipe_ids[index]),
                "content_id": int(content_ids[index]),
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "average_precision": average_precision,
                "recall_at_5": recall_at_5,
                "num_relevant": total_relevant,
            }
        )
    return {
        "retrieval_top1": float(np.mean(top1)),
        "retrieval_top5": float(np.mean(top5)),
        "retrieval_map": float(np.mean(average_precisions)),
        "retrieval_recall_at_5": float(np.mean(recall5)),
    }, rows


def synthetic_retrieval_evaluation(
    model,
    references: torch.Tensor,
    num_recipes: int,
    encode_batch_size: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], List[Dict[str, object]]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    recipes = sample_mixed_recipe(
        num_recipes,
        torch.device("cpu"),
        torch.float32,
        min_strength=0.15,
        max_strength=0.9,
        generator=generator,
    ).to(device=references.device, dtype=references.dtype)
    degraded_batches = []
    flipped_batches = []
    recipe_ids = []
    content_ids = []
    for recipe_id in range(num_recipes):
        repeated = recipes[recipe_id : recipe_id + 1].expand(len(references), -1)
        degraded, _, _ = apply_random_mixed_degradation(references, repeated)
        degraded_batches.append(degraded)
        flipped_batches.append(torch.flip(degraded, dims=(3,)))
        recipe_ids.extend([recipe_id] * len(references))
        content_ids.extend(range(len(references)))
    degraded = torch.cat(degraded_batches, dim=0)
    flipped = torch.cat(flipped_batches, dim=0)
    encoded = encode_batches(model, degraded, encode_batch_size)
    augmented = encode_batches(model, flipped, encode_batch_size)
    recipe_ids_arr = np.asarray(recipe_ids)
    content_ids_arr = np.asarray(content_ids)
    metrics, retrieval_rows = retrieval_metrics(
        encoded["z_deg"],
        recipe_ids_arr,
        content_ids_arr,
    )

    z = encoded["z_deg"].reshape(num_recipes, len(references), -1)
    z_aug = augmented["z_deg"].reshape(num_recipes, len(references), -1)
    same_deg_diff_content = cosine_distance(z, np.roll(z, shift=1, axis=1)).mean()
    same_content_diff_deg = cosine_distance(z, np.roll(z, shift=1, axis=0)).mean()
    same_content_same_deg_aug = cosine_distance(z, z_aug).mean()
    metrics.update(
        {
            "distance_same_content_same_degradation_augmentation": float(
                same_content_same_deg_aug
            ),
            "distance_same_degradation_different_content": float(same_deg_diff_content),
            "distance_same_content_different_degradation": float(same_content_diff_deg),
            "degradation_over_content_distance_ratio": float(
                same_content_diff_deg / max(same_deg_diff_content, 1e-12)
            ),
        }
    )
    arrays = {
        **encoded,
        "recipe_ids": recipe_ids_arr,
        "content_ids": content_ids_arr,
        "recipes": recipes.detach().cpu().numpy(),
    }
    return metrics, arrays, retrieval_rows


def severity_evaluation(
    model,
    references: torch.Tensor,
    num_directions: int,
    levels: np.ndarray,
    encode_batch_size: int,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, np.ndarray]]:
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    directions = sample_mixed_recipe(
        num_directions,
        torch.device("cpu"),
        torch.float32,
        min_strength=0.3,
        max_strength=1.0,
        generator=generator,
    ).to(device=references.device, dtype=references.dtype)
    clean = encode_batches(model, references, encode_batch_size)["z_deg"]
    all_features = []
    rows: List[Dict[str, object]] = []
    spearman_values = []
    step_values = []
    range_values = []

    for direction_id, direction in enumerate(directions):
        level_features = []
        for level in levels:
            recipe = scale_recipe(
                direction[None].expand(len(references), -1),
                references.new_full((len(references),), float(level)),
            )
            degraded, _, _ = apply_random_mixed_degradation(references, recipe)
            level_features.append(encode_batches(model, degraded, encode_batch_size)["z_deg"])
        trajectory = np.stack(level_features, axis=1)
        all_features.append(trajectory)
        for content_id in range(len(references)):
            distances = cosine_distance(
                np.repeat(clean[content_id : content_id + 1], len(levels), axis=0),
                trajectory[content_id],
            )
            correlation = float(spearmanr(levels, distances).statistic)
            increasing = float(np.mean(np.diff(distances) > 0))
            distance_range = float(distances.max() - distances.min())
            spearman_values.append(correlation)
            step_values.append(increasing)
            range_values.append(distance_range)
            for level, distance in zip(levels, distances):
                rows.append(
                    {
                        "direction_id": direction_id,
                        "content_id": content_id,
                        "severity": float(level),
                        "distance_to_clean": float(distance),
                    }
                )
    metrics = {
        "severity_spearman_mean": float(np.nanmean(spearman_values)),
        "severity_increasing_step_fraction": float(np.mean(step_values)),
        "severity_distance_range_mean": float(np.mean(range_values)),
    }
    arrays = {
        "features": np.stack(all_features, axis=0),
        "clean_features": clean,
        "levels": levels,
        "directions": directions.detach().cpu().numpy(),
    }
    return metrics, rows, arrays


@torch.no_grad()
def real_feature_evaluation(
    model,
    labels_csv: str,
    dataset_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], List[Dict[str, object]]]:
    dataset = Stage1ImageDataset(
        labels_csv,
        split=split,
        image_size=image_size,
        dataset_root=dataset_root,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    arrays: Dict[str, List[np.ndarray]] = {"z_deg": [], "slot_tokens": [], "m_deg": [], "targets": []}
    metadata = []
    for batch in tqdm(loader, desc="encoding real test images", leave=False):
        output = model(batch["image"].to(device))
        for key in ("z_deg", "slot_tokens", "m_deg"):
            arrays[key].append(output[key].detach().cpu().numpy())
        arrays["targets"].append(batch["scores"].numpy())
        for index in range(len(batch["image_name"])):
            metadata.append(
                {
                    "image_name": batch["image_name"][index],
                    "image_path": batch["image_path"][index],
                    "pair_id": batch["pair_id"][index],
                    "role": batch["role"][index],
                }
            )
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    z = np.asarray(merged["z_deg"], dtype=np.float64)
    targets = np.asarray(merged["targets"], dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite z_deg values found during real-image evaluation.")
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    metrics: Dict[str, float] = {}
    for index, name in enumerate(SCORE_COLUMNS):
        prediction = cross_validated_ridge_prediction(
            z,
            targets[:, index],
            folds,
            alpha=1.0,
        )
        error = prediction - targets[:, index]
        ss_total = np.sum((targets[:, index] - targets[:, index].mean()) ** 2)
        metrics[f"linear_probe_{name}_mae"] = float(mean_absolute_error(targets[:, index], prediction))
        metrics[f"linear_probe_{name}_rmse"] = float(
            np.sqrt(mean_squared_error(targets[:, index], prediction))
        )
        metrics[f"linear_probe_{name}_spearman"] = float(
            spearmanr(targets[:, index], prediction).statistic
        )
        metrics[f"linear_probe_{name}_r2"] = float(
            1.0 - np.sum(error**2) / ss_total
        )
    metrics["linear_probe_average_mae"] = float(
        np.mean([metrics[f"linear_probe_{name}_mae"] for name in SCORE_COLUMNS])
    )
    return metrics, merged, metadata


def pair_ranking_and_collapse(
    real_arrays: Dict[str, np.ndarray],
    metadata: List[Dict[str, object]],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    pair_map: Dict[str, Dict[str, int]] = {}
    for index, row in enumerate(metadata):
        pair_map.setdefault(str(row["pair_id"]), {})[str(row["role"])] = index
    details = []
    margins = []
    for pair_id, indices in pair_map.items():
        if "raw" not in indices or "reference" not in indices:
            continue
        raw = float(real_arrays["m_deg"][indices["raw"], 0])
        reference = float(real_arrays["m_deg"][indices["reference"], 0])
        margin = raw - reference
        margins.append(margin)
        details.append(
            {
                "pair_id": pair_id,
                "raw_magnitude": raw,
                "reference_magnitude": reference,
                "raw_minus_reference": margin,
                "correct": margin > 0,
            }
        )
    roles = np.asarray([row["role"] for row in metadata])
    labels = (roles == "raw").astype(int)
    magnitudes = np.asarray(real_arrays["m_deg"], dtype=np.float64).reshape(-1)
    metrics = {
        "raw_reference_ranking_accuracy": float(np.mean(np.asarray(margins) > 0)),
        "raw_reference_average_margin": float(np.mean(margins)),
        "raw_reference_pairwise_auc": float(roc_auc_score(labels, magnitudes)),
    }

    z = np.asarray(real_arrays["z_deg"], dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite z_deg values found during collapse evaluation.")
    z_tensor = torch.from_numpy(z).double()
    centered = z_tensor - z_tensor.mean(dim=0)
    covariance = centered.T @ centered / max(len(z) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).numpy()
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12))))
    correlation = np.corrcoef(z, rowvar=False)
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    normalized = F.normalize(z_tensor, dim=1)
    cosine = (normalized @ normalized.T).numpy()
    pairwise_cosine = cosine[np.triu_indices(len(z), k=1)]
    slots = np.asarray(real_arrays["slot_tokens"], dtype=np.float64)
    slots_tensor = F.normalize(torch.from_numpy(slots).double(), dim=-1)
    slot_similarity = (slots_tensor @ slots_tensor.transpose(1, 2)).numpy()
    slot_mask = ~np.eye(slots.shape[1], dtype=bool)
    metrics.update(
        {
            "collapse_dimension_std_mean": float(z.std(axis=0).mean()),
            "collapse_effective_rank": effective_rank,
            "collapse_effective_rank_fraction": effective_rank / z.shape[1],
            "collapse_mean_absolute_feature_correlation": float(np.nanmean(np.abs(off_diagonal))),
            "collapse_mean_pairwise_cosine": float(pairwise_cosine.mean()),
            "collapse_mean_slot_cosine": float(slot_similarity[:, slot_mask].mean()),
        }
    )
    return metrics, details


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate self-supervised degradation representations."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-contents", type=int, default=32)
    parser.add_argument("--num-recipes", type=int, default=8)
    parser.add_argument("--num-severity-directions", type=int, default=4)
    parser.add_argument("--severity-steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device_from_arg(args.device)
    model = load_self_supervised_encoder(args.checkpoint, device)

    pair_dataset = Stage1PairDataset(
        args.labels_csv,
        split=args.split,
        image_size=args.image_size,
        train=False,
        dataset_root=args.dataset_root,
    )
    content_count = min(args.num_contents, len(pair_dataset))
    references = torch.stack(
        [pair_dataset[index]["ref_image"] for index in range(content_count)]
    ).to(device)

    retrieval, synthetic_arrays, retrieval_rows = synthetic_retrieval_evaluation(
        model,
        references,
        args.num_recipes,
        args.batch_size,
        args.seed,
    )
    levels = np.linspace(0.0, 1.0, args.severity_steps)
    severity, severity_rows, severity_arrays = severity_evaluation(
        model,
        references,
        args.num_severity_directions,
        levels,
        args.batch_size,
        args.seed,
    )
    probe, real_arrays, metadata = real_feature_evaluation(
        model,
        args.labels_csv,
        args.dataset_root,
        args.split,
        args.image_size,
        args.batch_size,
        args.num_workers,
        device,
    )
    ranking_collapse, ranking_rows = pair_ranking_and_collapse(real_arrays, metadata)
    metrics = {**retrieval, **severity, **probe, **ranking_collapse}
    write_metric_csv(os.path.join(args.output_dir, "evaluation_summary.csv"), metrics)
    write_metric_csv(
        os.path.join(args.output_dir, "linear_probe_metrics.csv"),
        {key: value for key, value in probe.items()},
    )
    write_metric_csv(
        os.path.join(args.output_dir, "collapse_metrics.csv"),
        {
            key: value
            for key, value in ranking_collapse.items()
            if key.startswith("collapse_")
        },
    )

    with open(os.path.join(args.output_dir, "synthetic_retrieval.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(retrieval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(retrieval_rows)
    with open(os.path.join(args.output_dir, "severity_monotonicity.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(severity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(severity_rows)
    with open(os.path.join(args.output_dir, "raw_reference_ranking.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ranking_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranking_rows)

    np.savez_compressed(
        os.path.join(args.output_dir, "synthetic_retrieval_features.npz"),
        **synthetic_arrays,
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "severity_trajectories.npz"),
        **severity_arrays,
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "features_test.npz"),
        **real_arrays,
        image_names=np.asarray([row["image_name"] for row in metadata]),
        pair_ids=np.asarray([row["pair_id"] for row in metadata]),
        roles=np.asarray([row["role"] for row in metadata]),
        score_columns=np.asarray(SCORE_COLUMNS),
    )
    print(f"Wrote self-supervised evaluation to {args.output_dir}")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
