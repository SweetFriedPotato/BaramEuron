"""Validation metrics and diagnostic slices for exp02."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL


def add_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["absolute_error"] = (out["y_true_kwh"] - out["y_pred_kwh"]).abs()
    out["capacity_kwh"] = out["target"].map(CAPACITY_KWH)
    out["nmae_contribution"] = out["absolute_error"] / out["capacity_kwh"]
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    return out


def metric_tables(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = add_error_columns(predictions)
    keys = [column for column in ["experiment_id", "seed", "ensemble", "fold"] if column in data.columns]
    group = (
        data.groupby([*keys, "target", "group_id"], dropna=False, sort=False)
        .agg(mae=("absolute_error", "mean"), nmae=("nmae_contribution", "mean"), rows=("absolute_error", "size"))
        .reset_index()
    )
    macro = (
        group.groupby(keys, dropna=False, sort=False)
        .agg(macro_mae=("mae", "mean"), macro_nmae=("nmae", "mean"), groups=("group_id", "nunique"))
        .reset_index()
    )
    monthly = (
        data.assign(month=data[TIME_COL].dt.month)
        .groupby([*keys, "target", "group_id", "month"], dropna=False, sort=False)
        .agg(mae=("absolute_error", "mean"), nmae=("nmae_contribution", "mean"), rows=("absolute_error", "size"))
        .reset_index()
    )
    hourly = (
        data.assign(hour=data[TIME_COL].dt.hour)
        .groupby([*keys, "target", "group_id", "hour"], dropna=False, sort=False)
        .agg(mae=("absolute_error", "mean"), nmae=("nmae_contribution", "mean"), rows=("absolute_error", "size"))
        .reset_index()
    )
    january_data = data[data[TIME_COL].dt.month == 1]
    january = (
        january_data.groupby([*keys, "target", "group_id"], dropna=False, sort=False)
        .agg(mae=("absolute_error", "mean"), nmae=("nmae_contribution", "mean"), rows=("absolute_error", "size"))
        .reset_index()
    )
    if "high_wind_mask" in data:
        high = data[data["high_wind_mask"].astype(bool)]
        high_wind = (
            high.groupby([*keys, "target", "group_id"], dropna=False, sort=False)
            .agg(
                mae=("absolute_error", "mean"), nmae=("nmae_contribution", "mean"),
                rows=("absolute_error", "size"), train_wind_p90_mps=("train_wind_p90_mps", "first")
            )
            .reset_index()
        )
    else:
        high_wind = pd.DataFrame()
    return {"macro": macro, "group": group, "monthly": monthly, "hourly": hourly,
            "january": january, "high_wind": high_wind}


def prediction_diagnostics(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    keys = [TIME_COL, "target"]
    left = add_error_columns(reference)[[*keys, "y_true_kwh", "y_pred_kwh", "absolute_error"]].rename(
        columns={"y_pred_kwh": "reference_prediction", "absolute_error": "reference_ae"}
    )
    right = add_error_columns(candidate)[[*keys, "y_pred_kwh", "absolute_error"]].rename(
        columns={"y_pred_kwh": "candidate_prediction", "absolute_error": "candidate_ae"}
    )
    merged = left.merge(right, on=keys, validate="one_to_one")
    ref_residual = merged["reference_prediction"] - merged["y_true_kwh"]
    candidate_residual = merged["candidate_prediction"] - merged["y_true_kwh"]
    return {
        "rows": int(len(merged)),
        "residual_pearson": float(ref_residual.corr(candidate_residual)),
        "absolute_error_correlation": float(merged["reference_ae"].corr(merged["candidate_ae"])),
        "reference_better_fraction": float((merged["reference_ae"] < merged["candidate_ae"]).mean()),
        "candidate_better_fraction": float((merged["candidate_ae"] < merged["reference_ae"]).mean()),
        "ties_fraction": float((merged["candidate_ae"] == merged["reference_ae"]).mean()),
    }
