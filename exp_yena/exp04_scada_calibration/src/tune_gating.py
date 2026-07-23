from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from shared.constants import CAPACITY_KWH
from exp_yena.exp02_catboost_feature.src.run_experiment import calculate_group_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune M2 gating and group calibration from saved OOF predictions."
    )
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def evaluate(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    scores = []
    for target, group in frame.assign(candidate=prediction).groupby("target"):
        scores.append(
            calculate_group_metrics(group["y_true"], group["candidate"], target)
        )
    one_minus_nmae = float(np.mean([score["one_minus_nmae"] for score in scores]))
    ficr = float(np.mean([score["ficr"] for score in scores]))
    return {
        "total_score": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def gate_predictions(
    frame: pd.DataFrame,
    mode: str,
    low: float,
    high: float,
) -> np.ndarray:
    probability = frame["generation_probability"].to_numpy(float)
    regression = frame["regression_prediction"].to_numpy(float)
    if mode == "hard":
        gate = (probability >= high).astype(float)
    else:
        gate = np.clip((probability - low) / (high - low), 0.0, 1.0)
    return gate * regression


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.oof)
    required = {
        "target",
        "y_true",
        "generation_probability",
        "regression_prediction",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(
            f"OOF file is missing {missing}; rerun M2 validation with diagnostics enabled."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    for threshold in np.arange(0.05, 0.55, 0.05):
        prediction = gate_predictions(frame, "hard", 0.0, float(threshold))
        candidates.append({
            "mode": "hard",
            "low": 0.0,
            "high": float(threshold),
            **evaluate(frame, prediction),
        })
    for low, high in product(
        (0.0, 0.05, 0.1, 0.15, 0.2),
        (0.25, 0.35, 0.45, 0.55, 0.65),
    ):
        if low >= high:
            continue
        prediction = gate_predictions(frame, "soft", low, high)
        candidates.append({
            "mode": "soft",
            "low": low,
            "high": high,
            **evaluate(frame, prediction),
        })
    search = pd.DataFrame(candidates).sort_values(
        ["total_score", "ficr"], ascending=False
    )
    best = search.iloc[0]
    base_prediction = gate_predictions(
        frame, str(best["mode"]), float(best["low"]), float(best["high"])
    )

    calibration: dict[str, dict[str, float]] = {}
    calibrated = base_prediction.copy()
    for target, indices in frame.groupby("target").groups.items():
        indices = np.asarray(list(indices))
        capacity = CAPACITY_KWH[target]
        group_frame = frame.loc[indices]
        rows = []
        for scale, bias_fraction in product(
            np.arange(0.85, 1.151, 0.025),
            np.arange(-0.04, 0.041, 0.01),
        ):
            bias = float(bias_fraction * capacity)
            prediction = np.maximum(0.0, base_prediction[indices] * scale + bias)
            score = calculate_group_metrics(
                group_frame["y_true"], prediction, target
            )
            rows.append({
                "target": target,
                "scale": float(scale),
                "bias": bias,
                **score,
            })
        group_search = pd.DataFrame(rows).sort_values(
            ["total_score", "ficr"], ascending=False
        )
        group_search.to_csv(
            args.output_root / f"{target}_calibration_search.csv", index=False
        )
        group_best = group_search.iloc[0]
        scale = float(group_best["scale"])
        bias = float(group_best["bias"])
        calibration[target] = {"scale": scale, "bias": bias}
        calibrated[indices] = np.maximum(
            0.0, base_prediction[indices] * scale + bias
        )

    final_score = evaluate(frame, calibrated)
    result = {
        "base_config": "exp_yena/exp04_scada_calibration/configs/m2_two_stage.yaml",
        "two_stage": {
            "gating_mode": str(best["mode"]),
            "probability_threshold": float(best["high"]),
            "soft_low_threshold": float(best["low"]),
            "soft_high_threshold": float(best["high"]),
            "group_calibration": calibration,
        },
        "oof_score": final_score,
    }
    search.to_csv(args.output_root / "gating_search.csv", index=False)
    frame.assign(tuned_prediction=calibrated).to_csv(
        args.output_root / "tuned_oof_predictions.csv", index=False
    )
    (args.output_root / "best_gating.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
