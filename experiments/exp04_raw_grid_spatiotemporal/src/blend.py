"""Residual diagnostics and leakage-safe global Exp03/raw blending."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups


ALIGN_KEYS = ["fold", TIME_COL, "target", "group_id"]


def align_predictions(exp03: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    left = exp03[ALIGN_KEYS + ["y_true_kwh", "y_pred_kwh"]].rename(
        columns={"y_pred_kwh": "exp03_prediction"}
    )
    right = raw[ALIGN_KEYS + ["y_pred_kwh"]].rename(columns={"y_pred_kwh": "raw_prediction"})
    out = left.merge(right, on=ALIGN_KEYS, validate="one_to_one")
    if len(out) != len(left) or len(out) != len(right):
        raise ValueError("Exp03/raw prediction keys differ")
    return out


def blended_prediction(aligned: pd.DataFrame, raw_weight: float, model_id: str = "exp03_raw_blend") -> pd.DataFrame:
    out = aligned[ALIGN_KEYS + ["y_true_kwh"]].copy()
    out["y_pred_kwh"] = (
        (1.0 - float(raw_weight)) * aligned["exp03_prediction"]
        + float(raw_weight) * aligned["raw_prediction"]
    )
    out["model_id"] = model_id
    return out


def search_blend(exp03: pd.DataFrame, raw: pd.DataFrame, weights: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_predictions(exp03, raw)
    rows, candidates = [], []
    for weight in weights:
        frame = blended_prediction(aligned, float(weight))
        summary, _ = score_available_groups(frame)
        rows.append({"raw_weight": float(weight), **summary})
        frame["raw_weight"] = float(weight); candidates.append(frame)
    search = pd.DataFrame(rows).sort_values(["total_score", "raw_weight"], ascending=[False, True])
    return search, pd.concat(candidates, ignore_index=True)


def residual_correlations(exp03: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    aligned = align_predictions(exp03, raw)
    if "high_wind_mask" in raw:
        aligned = aligned.merge(
            raw[ALIGN_KEYS + ["high_wind_mask"]], on=ALIGN_KEYS, how="left", validate="one_to_one"
        )
    aligned["exp03_residual"] = aligned["exp03_prediction"] - aligned["y_true_kwh"]
    aligned["raw_residual"] = aligned["raw_prediction"] - aligned["y_true_kwh"]
    aligned["exp03_absolute_error"] = aligned["exp03_residual"].abs()
    aligned["raw_absolute_error"] = aligned["raw_residual"].abs()
    aligned["month"] = pd.to_datetime(aligned[TIME_COL]).dt.month
    rows = []
    groupings = [("overall", []) , ("group", ["group_id"]), ("month", ["month"])]
    if "high_wind_mask" in aligned:
        groupings.append(("high_wind", []))
    for slice_name, columns in groupings:
        source = aligned.loc[aligned["high_wind_mask"].fillna(False)] if slice_name == "high_wind" else aligned
        grouped = [((), source)] if not columns else source.groupby(columns, sort=True)
        for key, part in grouped:
            values = key if isinstance(key, tuple) else (key,)
            row = {"slice": slice_name, "samples": len(part)}
            row.update({column: value for column, value in zip(columns, values)})
            row["residual_pearson"] = float(part["exp03_residual"].corr(part["raw_residual"]))
            row["absolute_error_correlation"] = float(
                part["exp03_absolute_error"].corr(part["raw_absolute_error"])
            )
            rows.append(row)
    return pd.DataFrame(rows)
