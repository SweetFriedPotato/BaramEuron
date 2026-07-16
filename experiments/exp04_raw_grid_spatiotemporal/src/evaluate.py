"""Official-score tables and prediction-frame helpers for exp04."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import (
    evaluate_models as official_evaluate_models,
    score_available_groups,
)


def prediction_frame(
    timestamps: np.ndarray,
    target_cf: np.ndarray,
    label_mask: np.ndarray,
    prediction_cf: np.ndarray,
    model_id: str,
    fold: str,
    seed: int,
    validation_wind_mps: np.ndarray | None = None,
    high_wind_threshold: float | None = None,
) -> pd.DataFrame:
    if target_cf.shape != prediction_cf.shape or target_cf.shape != label_mask.shape:
        raise ValueError("prediction/target/mask shapes differ")
    parts = []
    for index, target in enumerate(TARGETS):
        mask = label_mask[..., index].reshape(-1)
        capacity = float(CAPACITY_KWH[target])
        frame = pd.DataFrame(
            {
                TIME_COL: timestamps.reshape(-1)[mask],
                "target": target,
                "group_id": index + 1,
                "y_true_kwh": (target_cf[..., index] * capacity).reshape(-1)[mask],
                "y_pred_kwh": (np.maximum(prediction_cf[..., index], 0.0) * capacity).reshape(-1)[mask],
                "model_id": model_id,
                "fold": fold,
                "seed": int(seed),
            }
        )
        if validation_wind_mps is not None:
            frame["validation_wind_mps"] = validation_wind_mps.reshape(-1)[mask]
            if high_wind_threshold is not None:
                frame["train_wind_p90_mps"] = float(high_wind_threshold)
                frame["high_wind_mask"] = frame["validation_wind_mps"] >= float(high_wind_threshold)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def seed_ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "fold", TIME_COL, "target", "group_id"]
    aggregations = {"y_true_kwh": "first", "y_pred_kwh": "mean"}
    for name in ("validation_wind_mps", "train_wind_p90_mps", "high_wind_mask", "quarter"):
        if name in predictions:
            aggregations[name] = "first"
    result = predictions.groupby(keys, sort=False).agg(aggregations).reset_index()
    result["seed"] = -1
    return result


def official_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return official_evaluate_models(predictions)


def sliced_scores(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = predictions.copy()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data["month"] = data[TIME_COL].dt.month
    capacity = data["target"].map(CAPACITY_KWH).astype(float)
    data["capacity_factor"] = data["y_true_kwh"] / capacity
    data["capacity_factor_band"] = pd.cut(
        data["capacity_factor"], [-np.inf, 0.33, 0.66, np.inf], labels=["low", "mid", "high"]
    )
    outputs = {}
    for name in ("month", "capacity_factor_band"):
        rows = []
        for keys, part in data.groupby(["model_id", "fold", name], observed=True, sort=True):
            try:
                summary, _ = score_available_groups(part)
            except ValueError:
                continue
            rows.append({"model_id": keys[0], "fold": keys[1], name: keys[2], **summary})
        outputs[name] = pd.DataFrame(rows)
    january = data.loc[data["month"].eq(1)]
    high_wind = data.loc[data.get("high_wind_mask", False).astype(bool)] if "high_wind_mask" in data else data.iloc[0:0]
    for name, subset in (("january", january), ("high_wind", high_wind)):
        rows = []
        for (model_id, fold), part in subset.groupby(["model_id", "fold"], sort=True):
            try:
                summary, _ = score_available_groups(part)
            except ValueError:
                continue
            rows.append({"model_id": model_id, "fold": fold, **summary})
        outputs[name] = pd.DataFrame(rows)
    return outputs


def rolling_quarter_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, quarter), part in predictions.groupby(["model_id", "quarter"], sort=True):
        summary, _ = score_available_groups(part)
        rows.append({"model_id": model_id, "quarter": quarter, **summary})
    return pd.DataFrame(rows)
