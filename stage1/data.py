import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .pseudo_labels import LEGACY_SCORE_ALIASES, SCORE_COLUMNS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def _resolve_dataset_path(row: Dict[str, str], labels_dir: str, dataset_root: Optional[str]) -> str:
    if dataset_root:
        dataset_root = os.path.abspath(dataset_root)
        role_dirs = ("raw-890", "raw") if row["role"] == "raw" else ("reference-890", "GT")
        for role_dir in role_dirs:
            candidate = os.path.join(dataset_root, role_dir, row["image_name"])
            if os.path.isfile(candidate):
                return candidate
        expected = ", ".join(os.path.join(dataset_root, role_dir, row["image_name"]) for role_dir in role_dirs)
        raise FileNotFoundError(f"Could not find '{row['image_name']}'. Checked: {expected}")

    image_path = row["image_path"]
    if not os.path.isabs(image_path):
        image_path = os.path.normpath(os.path.join(labels_dir, image_path))
    return image_path


def read_label_rows(
    labels_csv: str,
    split: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> List[Dict[str, str]]:
    labels_csv = os.path.abspath(labels_csv)
    labels_dir = os.path.dirname(labels_csv)
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["image_path"] = _resolve_dataset_path(row, labels_dir, dataset_root)
    if split:
        rows = [row for row in rows if row.get("split") == split]
    return rows


class Stage1PairDataset(Dataset):
    def __init__(
        self,
        labels_csv: str,
        split: str,
        image_size: int = 224,
        train: bool = False,
        dataset_root: Optional[str] = None,
    ):
        self.transform = build_transform(image_size=image_size, train=train)
        rows = read_label_rows(labels_csv, split=split, dataset_root=dataset_root)
        grouped: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
        for row in rows:
            grouped[row["pair_id"]][row["role"]] = row
        self.pairs = [
            {"pair_id": pair_id, "raw": item["raw"], "reference": item["reference"]}
            for pair_id, item in grouped.items()
            if "raw" in item and "reference" in item
        ]
        if not self.pairs:
            raise ValueError(f"No raw/reference pairs found in split '{split}'.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    @staticmethod
    def _scores(row: Dict[str, str]) -> torch.Tensor:
        return torch.tensor([score_value(row, name) for name in SCORE_COLUMNS], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.pairs[idx]
        raw = item["raw"]
        ref = item["reference"]
        return {
            "pair_id": item["pair_id"],
            "raw_image": self._load_image(raw["image_path"]),
            "ref_image": self._load_image(ref["image_path"]),
            "raw_scores": self._scores(raw),
            "ref_scores": self._scores(ref),
            "raw_path": raw["image_path"],
            "ref_path": ref["image_path"],
        }


class Stage1ImageDataset(Dataset):
    def __init__(
        self,
        labels_csv: str,
        split: Optional[str] = None,
        image_size: int = 224,
        dataset_root: Optional[str] = None,
    ):
        self.rows = read_label_rows(labels_csv, split=split, dataset_root=dataset_root)
        self.transform = build_transform(image_size=image_size, train=False)
        if not self.rows:
            raise ValueError("No image rows found for feature export.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "scores": torch.tensor([score_value(row, name) for name in SCORE_COLUMNS], dtype=torch.float32),
            "image_path": row["image_path"],
            "image_name": row["image_name"],
            "pair_id": row["pair_id"],
            "role": row["role"],
            "split": row["split"],
        }


def score_value(row: Dict[str, str], name: str) -> float:
    """Read a canonical score while accepting legacy metadata column names."""
    if name in row and row[name] != "":
        return float(row[name])
    legacy_name = LEGACY_SCORE_ALIASES.get(name)
    if legacy_name and legacy_name in row:
        return float(row[legacy_name])
    raise KeyError(f"Missing score column '{name}' (legacy alias: {legacy_name!r}).")
