"""Timestamp-safe CatBoost/TCN blending and weight search."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TIME_COL

from .evaluate import add_error_columns


def align_predictions(reference: pd.DataFrame, tcn: pd.DataFrame) -> pd.DataFrame:
    keys = [TIME_COL, "target", "group_id", "fold"]
    reference = reference.copy(); tcn = tcn.copy()
    reference[TIME_COL] = pd.to_datetime(reference[TIME_COL]); tcn[TIME_COL] = pd.to_datetime(tcn[TIME_COL])
    left = reference[[*keys, "y_true_kwh", "y_pred_kwh"]].rename(columns={"y_pred_kwh": "catboost_prediction"})
    right = tcn[[*keys, "y_pred_kwh"]].rename(columns={"y_pred_kwh": "tcn_prediction"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("CatBoost/TCN validation timestamp or row order differs")
    return merged.sort_values(keys).reset_index(drop=True)


def search_blend_weights(reference: pd.DataFrame, tcn: pd.DataFrame, weights: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_predictions(reference, tcn)
    result_rows = []
    prediction_parts = []
    for weight in weights:
        candidate = aligned.copy()
        candidate["y_pred_kwh"] = (
            (1.0 - float(weight)) * candidate["catboost_prediction"]
            + float(weight) * candidate["tcn_prediction"]
        )
        candidate["tcn_weight"] = float(weight)
        candidate = add_error_columns(candidate)
        for fold, fold_part in candidate.groupby("fold"):
            groups = fold_part.groupby("group_id")["nmae_contribution"].mean()
            january = fold_part[fold_part[TIME_COL].dt.month == 1].groupby("group_id")["nmae_contribution"].mean()
            row = {
                "tcn_weight": float(weight),
                "fold": fold,
                "macro_nmae": float(groups.mean()),
                "january_macro_nmae": float(january.mean()),
            }
            for group_id in groups.index:
                row[f"group_{int(group_id)}_nmae"] = float(groups.loc[group_id])
                row[f"group_{int(group_id)}_january_nmae"] = float(january.loc[group_id])
            result_rows.append(row)
        prediction_parts.append(candidate)
    return pd.DataFrame(result_rows), pd.concat(prediction_parts, ignore_index=True)


def select_blend_weight(search: pd.DataFrame) -> tuple[float, dict]:
    fold_b = search[search["fold"] == "fold_b"].copy()
    reference = fold_b.loc[fold_b["tcn_weight"] == 0.0].iloc[0]
    fold_a = search[search["fold"] == "fold_a"].set_index("tcn_weight")
    fold_a_reference = fold_a.loc[0.0]
    fold_b["january_not_worse"] = fold_b["january_macro_nmae"] <= reference["january_macro_nmae"] + 1e-12
    fold_b["group3_not_large_worse"] = fold_b.get("group_3_nmae", np.nan) <= reference.get("group_3_nmae", np.nan) + 0.003
    fold_b["fold_a_not_large_worse"] = fold_b["tcn_weight"].map(
        lambda weight: float(fold_a.loc[weight, "macro_nmae"]) <= float(fold_a_reference["macro_nmae"]) + 0.003
    )
    eligible = fold_b[
        fold_b["january_not_worse"] & fold_b["group3_not_large_worse"] & fold_b["fold_a_not_large_worse"]
    ]
    if eligible.empty:
        eligible = fold_b
    best = eligible.sort_values(["macro_nmae", "tcn_weight"]).iloc[0]
    return float(best["tcn_weight"]), {
        "fold_b_macro_nmae": float(best["macro_nmae"]),
        "fold_b_january_macro_nmae": float(best["january_macro_nmae"]),
        "january_not_worse": bool(best["january_not_worse"]),
        "group3_not_large_worse": bool(best["group3_not_large_worse"]),
        "fold_a_not_large_worse": bool(best["fold_a_not_large_worse"]),
    }
