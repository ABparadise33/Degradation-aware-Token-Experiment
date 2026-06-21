import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.pseudo_labels import SCORE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Collect metrics from baseline/V2/task experiments.")
    parser.add_argument("--glob", action="append", required=True, dest="patterns")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for pattern in args.patterns:
        for path in sorted(glob.glob(pattern)):
            metrics = {row["metric"]: float(row["value"]) for row in csv.DictReader(open(path))}
            aliases = {}
            for suffix in ("mae", "rmse", "pearson", "spearman", "r2"):
                canonical = f"s_visibility_proxy_{suffix}"
                aliases[canonical] = metrics.get(canonical, metrics.get(f"s_haze_{suffix}"))
            metrics.update({key: value for key, value in aliases.items() if value is not None})
            maes = [metrics[f"{name}_mae"] for name in SCORE_COLUMNS]
            rows.append(
                {
                    "experiment": os.path.normpath(path).split(os.sep)[-3],
                    "metrics_path": path,
                    "average_5_score_mae": sum(maes) / len(maes),
                    "ranking_acc": metrics["ranking_acc"],
                    "avg_q_ref_minus_raw": metrics["avg_q_ref_minus_raw"],
                    **{f"{name}_mae": metrics[f"{name}_mae"] for name in SCORE_COLUMNS},
                    **{f"{name}_spearman": metrics[f"{name}_spearman"] for name in SCORE_COLUMNS},
                }
            )
    if not rows:
        raise ValueError("No metrics files matched.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment rows to {args.output}")


if __name__ == "__main__":
    main()
