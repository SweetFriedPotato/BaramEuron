"""Official-score evaluation tables and threshold diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from official.dacon_baram_metric.metric import CAPACITY_KWH, TARGET_COLS


TIME_COL = "forecast_kst_dtm"


def add_official_components(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    out["capacity_kwh"] = out["target"].map(CAPACITY_KWH).astype(float)
    out["official_mask"] = out["y_true_kwh"] >= 0.10 * out["capacity_kwh"]
    out["absolute_error_kwh"] = (out["y_pred_kwh"] - out["y_true_kwh"]).abs()
    out["error_rate"] = out["absolute_error_kwh"] / out["capacity_kwh"]
    out["unit_price"] = np.select(
        [out["error_rate"] <= 0.06, out["error_rate"] <= 0.08], [4.0, 3.0], default=0.0
    )
    out["earned_settlement"] = out["y_true_kwh"] * out["unit_price"]
    out["max_settlement"] = out["y_true_kwh"] * 4.0
    return out


def score_available_groups(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    data = add_official_components(predictions)
    rows = []
    for (target, group_id), part in data.groupby(["target", "group_id"], sort=True):
        valid = part.loc[part["official_mask"]]
        if valid.empty:
            continue
        nmae = float(valid["error_rate"].mean())
        ficr = float(valid["earned_settlement"].sum() / valid["max_settlement"].sum())
        rows.append(
            {
                "target": target,
                "group_id": int(group_id),
                "nmae": nmae,
                "one_minus_nmae": 1.0 - nmae,
                "ficr": ficr,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "evaluated_samples": int(len(valid)),
                "total_samples": int(len(part)),
                "evaluated_rate": float(len(valid) / len(part)),
            }
        )
    groups = pd.DataFrame(rows)
    if groups.empty:
        raise ValueError("no samples meet the official evaluation mask")
    summary = {
        "total_score": float(groups["score"].mean()),
        "one_minus_nmae": float(groups["one_minus_nmae"].mean()),
        "ficr": float(groups["ficr"].mean()),
        "groups_available": int(len(groups)),
        "is_official_three_group_score": bool(set(groups["target"]) == set(TARGET_COLS)),
        "evaluated_samples": int(groups["evaluated_samples"].sum()),
        "total_samples": int(groups["total_samples"].sum()),
        "evaluated_rate": float(groups["evaluated_samples"].sum() / groups["total_samples"].sum()),
    }
    return summary, groups


def evaluate_models(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, groups = [], []
    for (model_id, fold), part in predictions.groupby(["model_id", "fold"], sort=True):
        summary, group = score_available_groups(part)
        unmasked_group = (
            add_official_components(part)
            .groupby(["target", "group_id"], sort=True)["error_rate"]
            .mean()
        )
        summary.update(
            {
                "model_id": model_id,
                "fold": fold,
                "unmasked_one_minus_nmae": float(1.0 - unmasked_group.mean()),
                "unmasked_macro_nmae": float(unmasked_group.mean()),
                "official_masked_macro_nmae": float(1.0 - summary["one_minus_nmae"]),
            }
        )
        summaries.append(summary)
        group.insert(0, "fold", fold)
        group.insert(0, "model_id", model_id)
        groups.append(group)
    return pd.DataFrame(summaries), pd.concat(groups, ignore_index=True)


def evaluation_mask_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    data = add_official_components(predictions)
    reference_model = sorted(data["model_id"].unique())[0]
    data = data.loc[data["model_id"].eq(reference_model)]
    data["month"] = data[TIME_COL].dt.month
    return (
        data.groupby(["fold", "target", "group_id", "month"], sort=True)
        .agg(
            total_samples=("official_mask", "size"),
            evaluated_samples=("official_mask", "sum"),
            actual_mean_kwh=("y_true_kwh", "mean"),
        )
        .reset_index()
        .assign(evaluated_rate=lambda x: x["evaluated_samples"] / x["total_samples"])
    )


def slice_scores(predictions: pd.DataFrame, slice_name: str, slice_values: pd.Series) -> pd.DataFrame:
    data = predictions.copy()
    data[slice_name] = np.asarray(slice_values)
    rows = []
    for keys, part in data.groupby(["model_id", "fold", slice_name], sort=True):
        try:
            summary, _ = score_available_groups(part)
        except ValueError:
            continue
        rows.append({"model_id": keys[0], "fold": keys[1], slice_name: keys[2], **summary})
    return pd.DataFrame(rows)


def ficr_threshold_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    data = add_official_components(predictions)
    data = data.loc[data["official_mask"]].copy()
    next_threshold = np.select(
        [data["error_rate"] <= 0.06, data["error_rate"] <= 0.08],
        [0.06, 0.08],
        default=np.nan,
    )
    data["distance_to_next_threshold"] = next_threshold - data["error_rate"]
    data["within_0_5pp_below"] = data["distance_to_next_threshold"].between(0.0, 0.005)
    data["within_1pp_below"] = data["distance_to_next_threshold"].between(0.0, 0.01)
    distance_above = np.minimum(np.abs(data["error_rate"] - 0.06), np.abs(data["error_rate"] - 0.08))
    data["just_above_threshold"] = (
        ((data["error_rate"] > 0.06) & (data["error_rate"] <= 0.065))
        | ((data["error_rate"] > 0.08) & (data["error_rate"] <= 0.085))
    )
    data["nearest_threshold_distance"] = distance_above
    data["month"] = data[TIME_COL].dt.month
    data["wind_band"] = pd.cut(
        data.get("validation_wind_mps", pd.Series(np.nan, index=data.index)),
        [-np.inf, 4.0, 8.0, np.inf], labels=["low", "mid", "high"],
    )
    return (
        data.groupby(["model_id", "fold", "target", "group_id", "month", "wind_band"],
                     observed=True, dropna=False, sort=True)
        .agg(
            samples=("error_rate", "size"),
            ficr_full_rate=("error_rate", lambda x: float((x <= 0.06).mean())),
            ficr_partial_rate=("error_rate", lambda x: float(((x > 0.06) & (x <= 0.08)).mean())),
            ficr_zero_rate=("error_rate", lambda x: float((x > 0.08).mean())),
            within_0_5pp_below=("within_0_5pp_below", "sum"),
            within_1pp_below=("within_1pp_below", "sum"),
            just_above_threshold=("just_above_threshold", "sum"),
            mean_nearest_threshold_distance=("nearest_threshold_distance", "mean"),
        )
        .reset_index()
    )
