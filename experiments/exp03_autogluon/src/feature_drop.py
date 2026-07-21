from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_REPORT = Path("experiments/exp03_autogluon/outputs/gpu/feature_importances_report.csv")
DEFAULT_DROP_LIST = Path("experiments/exp03_autogluon/configs/dropped_features_list.txt")


def drop_low_importance_features(
    report_path: Path,
    output_path: Path,
    threshold: float = 0.05,
) -> list[str]:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Feature importance report not found: {report_path}. "
            "Run validation first to create it."
        )

    report = pd.read_csv(report_path)
    required = {"feature", "importance"}
    if not required.issubset(report.columns):
        raise ValueError(f"Report must contain columns {sorted(required)}: {report_path}")

    dropped = report.loc[report["importance"] < threshold].sort_values("importance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{feature}\n" for feature in dropped["feature"]),
        encoding="utf-8",
    )
    print(
        f"Saved {len(dropped)} features with importance < {threshold} "
        f"to {output_path}"
    )
    return dropped["feature"].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the tracked Exp03 feature-drop list")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DROP_LIST)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    drop_low_importance_features(args.report, args.output, args.threshold)
