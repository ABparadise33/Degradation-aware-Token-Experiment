import csv
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SCORE_COLUMNS = ["s_color", "s_blur", "s_contrast", "s_haze", "q_quality"]


@dataclass
class ImageMetrics:
    image_path: str
    image_name: str
    pair_id: str
    split: str
    role: str
    is_reference: int
    color_raw: float
    sharpness_raw: float
    contrast_raw: float
    saturation_raw: float
    brightness_std_raw: float
    uiqm_raw: float
    uciqe_raw: float


def list_images(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(dirpath, filename))
    return sorted(files)


def stable_split(pair_id: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> str:
    digest = hashlib.md5(pair_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _trimmed_mean(values: np.ndarray, trim: float = 0.1) -> float:
    flat = np.sort(values.reshape(-1).astype(np.float64))
    lo = int(len(flat) * trim)
    hi = int(len(flat) * (1.0 - trim))
    if hi <= lo:
        return float(flat.mean())
    return float(flat[lo:hi].mean())


def _eme(gray: np.ndarray, block_size: int = 10) -> float:
    gray = gray.astype(np.float64) + 1e-6
    h_blocks = gray.shape[0] // block_size
    w_blocks = gray.shape[1] // block_size
    if h_blocks == 0 or w_blocks == 0:
        return 0.0
    value = 0.0
    for y in range(h_blocks):
        for x in range(w_blocks):
            block = gray[y * block_size : (y + 1) * block_size, x * block_size : (x + 1) * block_size]
            block_max = float(block.max())
            block_min = float(block.min())
            if block_max > 0.0 and block_min > 0.0:
                value += np.log(block_max / block_min)
    return float(2.0 * value / (h_blocks * w_blocks))


def uiqm(rgb: np.ndarray) -> float:
    img = rgb.astype(np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    mean_rg = _trimmed_mean(rg)
    mean_yb = _trimmed_mean(yb)
    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    uicm = -0.0268 * np.sqrt(mean_rg**2 + mean_yb**2) + 0.1586 * np.sqrt(std_rg**2 + std_yb**2)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharp = np.hypot(sobel_x, sobel_y)
    uism = _eme(sharp * gray, block_size=10)
    uiconm = _eme(gray, block_size=10)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def uciqe(rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    l_chan = lab[:, :, 0] / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    chroma_std = float(np.std(chroma))
    luminance_contrast = float(np.percentile(l_chan, 99) - np.percentile(l_chan, 1))
    saturation_mean = float((hsv[:, :, 1] / 255.0).mean())
    return float(0.4680 * chroma_std + 0.2745 * luminance_contrast + 0.2576 * saturation_mean)


def compute_raw_metrics(path: str, pair_id: str, role: str, split: str) -> ImageMetrics:
    rgb = read_rgb(path)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    mean_a = float(lab[:, :, 1].mean() - 128.0)
    mean_b = float(lab[:, :, 2].mean() - 128.0)
    lab_shift = np.sqrt(mean_a * mean_a + mean_b * mean_b)
    rgb_imbalance = float(np.std(rgb.reshape(-1, 3).mean(axis=0)))
    color_raw = float(lab_shift + rgb_imbalance)

    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_raw = float(ycrcb[:, :, 0].std())
    saturation_raw = float((hsv[:, :, 1] / 255.0).mean())
    brightness_std_raw = float((hsv[:, :, 2] / 255.0).std())

    return ImageMetrics(
        image_path=os.path.abspath(path),
        image_name=os.path.basename(path),
        pair_id=pair_id,
        split=split,
        role=role,
        is_reference=1 if role == "reference" else 0,
        color_raw=color_raw,
        sharpness_raw=sharpness_raw,
        contrast_raw=contrast_raw,
        saturation_raw=saturation_raw,
        brightness_std_raw=brightness_std_raw,
        uiqm_raw=uiqm(rgb),
        uciqe_raw=uciqe(rgb),
    )


def _minmax(
    values: Sequence[float],
    fit_mask: Optional[Sequence[bool]] = None,
    invert: bool = False,
) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    fit_arr = arr if fit_mask is None else arr[np.asarray(fit_mask, dtype=bool)]
    if fit_arr.size == 0:
        raise ValueError("Cannot normalize pseudo-labels without training samples.")
    lo = float(np.nanmin(fit_arr))
    hi = float(np.nanmax(fit_arr))
    if hi - lo < 1e-12:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    if invert:
        norm = 1.0 - norm
    return [float(np.clip(v, 0.0, 1.0)) for v in norm]


def _match_pairs(raw_dir: str, reference_dir: Optional[str]) -> List[Tuple[str, Optional[str], str]]:
    raw_paths = list_images(raw_dir)
    if not reference_dir:
        return [(path, None, os.path.splitext(os.path.basename(path))[0]) for path in raw_paths]

    ref_map = {
        os.path.basename(path): path
        for path in list_images(reference_dir)
    }
    pairs = []
    for raw_path in raw_paths:
        filename = os.path.basename(raw_path)
        if filename in ref_map:
            pair_id = os.path.splitext(filename)[0]
            pairs.append((raw_path, ref_map[filename], pair_id))
    return pairs


def build_pseudo_label_rows(
    raw_dir: str,
    reference_dir: Optional[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> List[Dict[str, object]]:
    metric_rows: List[ImageMetrics] = []
    pairs = _match_pairs(raw_dir, reference_dir)
    if not pairs:
        raise ValueError("No matching images found for pseudo-label generation.")

    for raw_path, ref_path, pair_id in pairs:
        split = stable_split(pair_id, train_ratio=train_ratio, val_ratio=val_ratio)
        metric_rows.append(compute_raw_metrics(raw_path, pair_id, "raw", split))
        if ref_path:
            metric_rows.append(compute_raw_metrics(ref_path, pair_id, "reference", split))

    # Fit normalization only on training data, then transform every split.
    # This keeps validation/test statistics out of model development.
    train_mask = [row.split == "train" for row in metric_rows]
    color_scores = _minmax([row.color_raw for row in metric_rows], fit_mask=train_mask)
    blur_scores = _minmax([row.sharpness_raw for row in metric_rows], fit_mask=train_mask, invert=True)
    contrast_scores = _minmax([row.contrast_raw for row in metric_rows], fit_mask=train_mask, invert=True)
    low_sat_scores = _minmax([row.saturation_raw for row in metric_rows], fit_mask=train_mask, invert=True)
    flat_scores = _minmax([row.brightness_std_raw for row in metric_rows], fit_mask=train_mask, invert=True)
    quality_uiqm = _minmax([row.uiqm_raw for row in metric_rows], fit_mask=train_mask)
    quality_uciqe = _minmax([row.uciqe_raw for row in metric_rows], fit_mask=train_mask)

    rows: List[Dict[str, object]] = []
    for i, row in enumerate(metric_rows):
        s_haze = 0.5 * contrast_scores[i] + 0.3 * low_sat_scores[i] + 0.2 * flat_scores[i]
        q_quality = 0.5 * quality_uiqm[i] + 0.5 * quality_uciqe[i]
        rows.append(
            {
                **row.__dict__,
                "s_color": color_scores[i],
                "s_blur": blur_scores[i],
                "s_contrast": contrast_scores[i],
                "s_haze": float(np.clip(s_haze, 0.0, 1.0)),
                "q_quality": float(np.clip(q_quality, 0.0, 1.0)),
            }
        )
    return rows


def write_csv(rows: Iterable[Dict[str, object]], output_path: str) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty pseudo-label CSV.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
