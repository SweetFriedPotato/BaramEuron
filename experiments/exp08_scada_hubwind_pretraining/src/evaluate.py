"""Stage-1 physical metrics and Stage-2 official-score acceptance tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp04_raw_grid_spatiotemporal.src.blend import residual_correlations

from .scada_contract import TARGET_NAMES


EXP04_ROLLING_SCORE = 0.647439599391
EXP04_FOLD_B_SCORE = 0.650288
EXP04_PUBLIC_SCORE = 0.634005715
CHAMPION_THRESHOLD = 0.649440


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    if len(left) < 2 or np.nanstd(left) == 0 or np.nanstd(right) == 0:
        return np.nan
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy()
        right = pd.Series(right).rank(method="average").to_numpy()
    return float(np.corrcoef(left, right)[0, 1])


def stage1_metric_tables(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, pd.DataFrame]:
    if prediction.shape != target.shape or target.shape != mask.shape or target.shape[-2:] != (3, 4):
        raise ValueError("Stage-1 metric arrays must share [...,3,4]")
    time = pd.DatetimeIndex(timestamps.reshape(-1))
    rows = []
    point_rows = []
    for group in range(3):
        for target_index, target_name in enumerate(TARGET_NAMES):
            valid = mask[..., group, target_index].reshape(-1) & np.isfinite(target[..., group, target_index].reshape(-1))
            true = target[..., group, target_index].reshape(-1)[valid]
            pred = prediction[..., group, target_index].reshape(-1)[valid]
            if len(true) == 0:
                continue
            rows.append({
                "group_id": group + 1,
                "target": target_name,
                "samples": int(len(true)),
                "mae": float(np.mean(np.abs(pred - true))),
                "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
                "pearson": _correlation(pred, true),
                "spearman": _correlation(pred, true, rank=True),
                "predicted_mean": float(np.mean(pred)),
                "observed_mean": float(np.mean(true)),
                "calibration_ratio": float(np.mean(pred) / max(np.mean(true), 1e-8)),
            })
            part = pd.DataFrame({
                TIME_COL: time[valid], "group_id": group + 1, "target": target_name,
                "y_true_mps": true, "y_pred_mps": pred,
            })
            point_rows.append(part)
    points = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    groups = pd.DataFrame(rows)
    if points.empty:
        return {"group": groups, "month": pd.DataFrame(), "lead": pd.DataFrame(), "wind_regime": pd.DataFrame(), "points": points}
    points["month"] = pd.to_datetime(points[TIME_COL]).dt.month
    # The issue contract is always 24 steps; preserve 1..24 for interpretable lead slices.
    points["lead_hour"] = np.tile(np.arange(1, 25), len(points) // 24 + 1)[:len(points)]
    median_points = points.loc[points["target"].eq("hub_ws_median")].copy()
    median_points["wind_regime"] = pd.cut(
        median_points["y_true_mps"], [-np.inf, 4.0, 10.0, np.inf], labels=["low", "mid", "high"]
    )

    def sliced(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        sliced_rows = []
        for keys, part in data.groupby(columns, observed=True, sort=True):
            values = keys if isinstance(keys, tuple) else (keys,)
            row = {name: value for name, value in zip(columns, values)}
            row.update({
                "samples": int(len(part)),
                "mae": float(np.mean(np.abs(part["y_pred_mps"] - part["y_true_mps"]))),
                "rmse": float(np.sqrt(np.mean((part["y_pred_mps"] - part["y_true_mps"]) ** 2))),
                "pearson": _correlation(part["y_pred_mps"].to_numpy(), part["y_true_mps"].to_numpy()),
            })
            sliced_rows.append(row)
        return pd.DataFrame(sliced_rows)

    return {
        "group": groups,
        "month": sliced(points, ["group_id", "target", "month"]),
        "lead": sliced(median_points, ["group_id", "lead_hour"]),
        "wind_regime": sliced(median_points, ["group_id", "wind_regime"]),
        "points": points,
    }


def reproduce_exp04_reference(prediction_path: Path, output_path: Path | None = None) -> dict:
    frame = pd.read_csv(prediction_path, parse_dates=[TIME_COL])
    summary, _ = score_available_groups(frame)
    observed = float(summary["total_score"])
    payload = {
        "reference": "Exp04 Exp03/raw rolling blend",
        "expected_score": EXP04_ROLLING_SCORE,
        "observed_score": observed,
        "absolute_difference": abs(observed - EXP04_ROLLING_SCORE),
        "exact_within_1e-12": abs(observed - EXP04_ROLLING_SCORE) <= 1e-12,
        "fold_b_reference": EXP04_FOLD_B_SCORE,
        "public_context_only": EXP04_PUBLIC_SCORE,
        "public_used_for_selection": False,
    }
    if not payload["exact_within_1e-12"]:
        raise ValueError(f"Exp04 exact reproduction failed: {payload}")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize_power_candidate(predictions: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    summary, groups = score_available_groups(predictions)
    quarter_rows = []
    for quarter, part in predictions.groupby("quarter" if "quarter" in predictions else "fold", sort=True):
        score, _ = score_available_groups(part)
        quarter_rows.append({"quarter": quarter, **score})
    quarters = pd.DataFrame(quarter_rows)
    result = {**summary}
    result.update({
        "equal_quarter_mean": float(quarters["total_score"].mean()),
        "worst_quarter": float(quarters["total_score"].min()),
        "group_3_score": float(groups.loc[groups["group_id"].eq(3), "score"].iloc[0]),
        "quarter_scores": quarters,
        "group_scores": groups,
    })
    if reference is not None:
        reference_rows = []
        for quarter, part in reference.groupby("quarter" if "quarter" in reference else "fold", sort=True):
            score, _ = score_available_groups(part)
            reference_rows.append({"quarter": quarter, "reference_score": score["total_score"]})
        comparison = quarters.merge(pd.DataFrame(reference_rows), on="quarter", validate="one_to_one")
        result["improved_quarters"] = int((comparison["total_score"] >= comparison["reference_score"]).sum())
        result["worst_quarter_degradation"] = float((comparison["reference_score"] - comparison["total_score"]).max())
        result["quarter_comparison"] = comparison
        result["residual_correlation"] = float(
            residual_correlations(reference, predictions).loc[lambda x: x["slice"].eq("overall"), "residual_pearson"].iloc[0]
        )
    return result


def acceptance(candidate: dict, reference: dict, seed_scores: list[float]) -> dict:
    mean_seed = float(np.mean(seed_scores)) if seed_scores else np.nan
    checks = {
        "rolling_at_least_0_649440": candidate["total_score"] >= CHAMPION_THRESHOLD,
        "improvement_at_least_0_002": candidate["total_score"] - reference["total_score"] >= 0.002,
        "improved_quarters_at_least_6": candidate.get("improved_quarters", 0) >= 6,
        "worst_quarter_degradation_at_most_0_002": candidate.get("worst_quarter_degradation", np.inf) <= 0.002,
        "ficr_maintained": candidate["ficr"] >= reference["ficr"],
        "one_minus_nmae_within_0_0005": candidate["one_minus_nmae"] >= reference["one_minus_nmae"] - 0.0005,
        "group_3_maintained": candidate["group_3_score"] >= reference["group_3_score"],
        "three_seed_mean_improves": len(seed_scores) == 3 and mean_seed > reference["total_score"],
        "not_single_seed_dependent": len(seed_scores) == 3 and sum(score > reference["total_score"] for score in seed_scores) >= 2,
    }
    return {"accepted": bool(all(checks.values())), "checks": checks, "seed_mean": mean_seed}
