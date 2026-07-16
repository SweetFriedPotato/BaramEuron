"""Leakage-safe issue-block quarterly backtest helpers."""

from __future__ import annotations

import pandas as pd

from .evaluate import score_available_groups


TIME_COL = "forecast_kst_dtm"
ROLLING_QUARTERS = [f"{year}Q{quarter}" for year in (2023, 2024) for quarter in (1, 2, 3, 4)]


def expanding_quarter_window(quarter: str) -> dict[str, pd.Timestamp]:
    if quarter not in ROLLING_QUARTERS:
        raise ValueError(f"unsupported rolling quarter: {quarter}")
    period = pd.Period(quarter, freq="Q")
    valid_start = period.start_time + pd.Timedelta(hours=1)
    next_quarter_start = (period + 1).start_time
    return {
        "train_start": pd.Timestamp("2022-01-01 01:00:00"),
        "train_end": period.start_time,
        "valid_start": valid_start,
        "valid_end": next_quarter_start,
    }


def issue_quarter(timestamps: pd.Series) -> pd.Series:
    """Assign 01:00..next-day 00:00 forecast blocks to one calendar quarter."""
    return (pd.to_datetime(timestamps) - pd.Timedelta(hours=1)).dt.to_period("Q").astype(str)


def quarterly_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["quarter"] = issue_quarter(data[TIME_COL])
    rows = []
    for (model_id, quarter), part in data.groupby(["model_id", "quarter"], sort=True):
        summary, _ = score_available_groups(part)
        rows.append({"model_id": model_id, "quarter": quarter, **summary})
    return pd.DataFrame(rows)


def rolling_selection_splits(quarters: list[str]) -> list[tuple[list[str], str]]:
    ordered = sorted(set(quarters))
    return [(ordered[:index], ordered[index]) for index in range(1, len(ordered))]


def assert_selection_precedes_evaluation(fit_quarters: list[str], evaluation_quarter: str) -> None:
    if not fit_quarters:
        raise ValueError("calibration fit quarters cannot be empty")
    if max(pd.Period(q, freq="Q") for q in fit_quarters) >= pd.Period(evaluation_quarter, freq="Q"):
        raise ValueError("calibration selection leaks into its evaluation quarter")
