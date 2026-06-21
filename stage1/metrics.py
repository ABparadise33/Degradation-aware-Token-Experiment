from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .pseudo_labels import SCORE_COLUMNS


def _safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(fn(y_true, y_pred).statistic)


def regression_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx, name in enumerate(SCORE_COLUMNS):
        y_true = targets[:, idx]
        y_pred = preds[:, idx]
        err = y_pred - y_true
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        out[f"{name}_mae"] = float(np.mean(np.abs(err)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))
        out[f"{name}_pearson"] = _safe_corr(pearsonr, y_true, y_pred)
        out[f"{name}_spearman"] = _safe_corr(spearmanr, y_true, y_pred)
        out[f"{name}_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return out


def ranking_metrics(records: Iterable[Tuple[str, str, float]]) -> Dict[str, float]:
    pair_scores: Dict[str, Dict[str, float]] = {}
    for pair_id, role, q_pred in records:
        pair_scores.setdefault(pair_id, {})[role] = q_pred
    margins = []
    correct = 0
    for item in pair_scores.values():
        if "raw" not in item or "reference" not in item:
            continue
        margin = item["reference"] - item["raw"]
        margins.append(float(margin))
        correct += int(margin > 0.0)
    if not margins:
        return {"ranking_acc": float("nan"), "avg_q_ref_minus_raw": float("nan"), "num_pairs": 0}
    return {
        "ranking_acc": float(correct / len(margins)),
        "avg_q_ref_minus_raw": float(np.mean(margins)),
        "num_pairs": len(margins),
    }

