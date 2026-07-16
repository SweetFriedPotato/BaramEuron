"""Official rolling, group, and operational-slice evaluation for Exp05."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups


def prediction_view(data: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    required = {TIME_COL, "target", "group_id", "y_true_kwh", prediction_column}
    if missing := required - set(data):
        raise ValueError(f"prediction view missing columns: {sorted(missing)}")
    return data[list(required)].rename(columns={prediction_column: "y_pred_kwh"})


def rolling_metrics(
    data: pd.DataFrame,
    prediction_column: str,
    reference_column: str = "global_blend_prediction",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    overall, groups = score_available_groups(prediction_view(data, prediction_column))
    rows = []
    for quarter, part in data.groupby("quarter", sort=True):
        metric, _ = score_available_groups(prediction_view(part, prediction_column))
        reference, _ = score_available_groups(prediction_view(part, reference_column))
        rows.append({
            "quarter": quarter,
            **metric,
            "reference_score": reference["total_score"],
            "delta_vs_exp04": metric["total_score"] - reference["total_score"],
            "improved_or_equal": metric["total_score"] >= reference["total_score"] - 1e-12,
        })
    quarters = pd.DataFrame(rows)
    summary = {
        **overall,
        "equal_quarter_mean": float(quarters["total_score"].mean()),
        "worst_quarter": float(quarters["total_score"].min()),
        "improved_quarters": int(quarters["improved_or_equal"].sum()),
        "quarter_count": int(len(quarters)),
    }
    return summary, quarters, groups


def slice_metrics(data: pd.DataFrame, prediction_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = data.copy()
    values[TIME_COL] = pd.to_datetime(values[TIME_COL])
    january = values.loc[values[TIME_COL].dt.month.eq(1)]
    high_wind = values.loc[values.get("high_wind_mask", False).astype(bool)]
    january_score, january_groups = score_available_groups(prediction_view(january, prediction_column))
    high_score, high_groups = score_available_groups(prediction_view(high_wind, prediction_column))
    return (
        pd.DataFrame([{"slice": "january", **january_score}]).join(
            january_groups.set_index("target")["score"].rename(lambda x: f"{x}_score").to_frame().T.reset_index(drop=True)
        ),
        pd.DataFrame([{"slice": "high_wind", **high_score}]).join(
            high_groups.set_index("target")["score"].rename(lambda x: f"{x}_score").to_frame().T.reset_index(drop=True)
        ),
    )


def stage_d_decision(
    candidate_summary: dict,
    minimum_score: float = 0.649440,
    minimum_group3_score: float = 0.619185,
    reference_worst: float = 0.605463,
) -> dict:
    conditions = {
        "aggregate_at_least_0_649440": candidate_summary["total_score"] >= minimum_score,
        "group3_at_least_0_619185": candidate_summary["group3_score"] >= minimum_group3_score,
        "worst_not_below_exp04": candidate_summary["worst_quarter"] >= reference_worst - 1e-12,
    }
    return {
        "conditions": conditions,
        "satisfied": int(sum(conditions.values())),
        "skip_cross_group_attention": int(sum(conditions.values())) >= 2,
    }


def convex_search(
    data: pd.DataFrame,
    candidates: list[str],
    step: float = 0.05,
    maximum_models: int = 3,
) -> pd.DataFrame:
    """Small exhaustive nonnegative convex search over at most three candidates."""
    if not 1 <= len(candidates) <= maximum_models:
        raise ValueError("convex search accepts one to three candidates")
    ticks = int(round(1.0 / step))
    rows = []
    if len(candidates) == 1:
        vectors = [(ticks,)]
    elif len(candidates) == 2:
        vectors = [(a, ticks-a) for a in range(ticks + 1)]
    else:
        vectors = [(a, b, ticks-a-b) for a in range(ticks + 1) for b in range(ticks-a+1)]
    for vector in vectors:
        weights = np.asarray(vector, dtype=float) / ticks
        prediction = sum(weights[index] * data[column] for index, column in enumerate(candidates))
        work = data.copy(); work["ensemble_prediction"] = prediction
        summary, _, _ = rolling_metrics(work, "ensemble_prediction")
        rows.append({**{f"weight_{column}": weights[index] for index, column in enumerate(candidates)}, **summary})
    return pd.DataFrame(rows).sort_values(
        ["total_score", "equal_quarter_mean", "worst_quarter"], ascending=False
    ).reset_index(drop=True)
