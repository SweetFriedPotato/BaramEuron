"""Prediction-only blend, seed aggregation, and affine calibration search."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from official.dacon_baram_metric.metric import CAPACITY_KWH

from .backtest import assert_selection_precedes_evaluation, issue_quarter
from .evaluate import score_available_groups
from .prediction_loader import KEY_COLS, align_models


def blend_predictions(left: pd.DataFrame, right: pd.DataFrame, right_weight: float, model_id: str) -> pd.DataFrame:
    if not 0.0 <= right_weight <= 1.0:
        raise ValueError("blend weight must be in [0, 1]")
    merged = align_models(left, right)
    merged["y_pred_kwh"] = (
        (1.0 - right_weight) * merged["left_prediction"]
        + right_weight * merged["right_prediction"]
    )
    merged["model_id"] = model_id
    merged["blend_weight"] = float(right_weight)
    return merged[[*KEY_COLS, "y_true_kwh", "y_pred_kwh", "model_id", "blend_weight"]]


def search_global_blend(left: pd.DataFrame, right: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for weight in weights:
        candidate = blend_predictions(left, right, float(weight), f"blend_{weight:.3f}")
        summary, _ = score_available_groups(candidate)
        rows.append({"right_weight": float(weight), **summary})
    return pd.DataFrame(rows).sort_values(["total_score", "right_weight"], ascending=[False, True])


def apply_affine(predictions: pd.DataFrame, parameters: dict[str, tuple[float, float]], model_id: str) -> pd.DataFrame:
    out = predictions.copy().reset_index(drop=True)
    calibrated = np.empty(len(out), dtype=float)
    for target, indices in out.groupby("target").groups.items():
        if target not in parameters:
            raise ValueError(f"affine parameters missing for {target}")
        scale, offset = parameters[target]
        calibrated[indices] = out.loc[indices, "y_pred_kwh"] * scale + offset
    out["y_pred_kwh"] = np.maximum(calibrated, 0.0)
    out["model_id"] = model_id
    return out


def _group_objective(part: pd.DataFrame) -> float:
    summary, _ = score_available_groups(part)
    return float(summary["total_score"])


def search_affine_group(
    predictions: pd.DataFrame,
    target: str,
    scales: np.ndarray,
    offsets: np.ndarray,
) -> pd.DataFrame:
    part = predictions.loc[predictions["target"].eq(target)].copy()
    rows = []
    for scale, offset in itertools.product(scales, offsets):
        candidate = part.copy()
        candidate["y_pred_kwh"] = np.maximum(candidate["y_pred_kwh"] * scale + offset, 0.0)
        rows.append({"target": target, "scale": scale, "offset_kwh": offset,
                     "total_score": _group_objective(candidate)})
    return pd.DataFrame(rows).sort_values(
        ["total_score", "scale", "offset_kwh"], ascending=[False, True, True]
    )


def select_affine_parameters(predictions: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    parameters, searches = {}, []
    for target in sorted(predictions["target"].unique()):
        capacity = float(CAPACITY_KWH[target])
        coarse = search_affine_group(
            predictions,
            target,
            np.round(np.arange(0.90, 1.1001, 0.01), 4),
            np.linspace(-0.03 * capacity, 0.03 * capacity, 13),
        )
        best = coarse.iloc[0]
        fine = search_affine_group(
            predictions,
            target,
            np.linspace(max(0.90, best["scale"] - 0.01), min(1.10, best["scale"] + 0.01), 11),
            np.linspace(max(-0.03 * capacity, best["offset_kwh"] - 0.005 * capacity),
                        min(0.03 * capacity, best["offset_kwh"] + 0.005 * capacity), 11),
        )
        fine["stage"] = "fine"
        coarse["stage"] = "coarse"
        chosen = fine.iloc[0]
        parameters[target] = (float(chosen["scale"]), float(chosen["offset_kwh"]))
        searches.extend([coarse, fine])
    return parameters, pd.concat(searches, ignore_index=True)


def rolling_affine_backtest(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = predictions.copy()
    data["quarter"] = issue_quarter(data["forecast_kst_dtm"])
    quarters = sorted(data["quarter"].unique())
    result_rows, search_parts = [], []
    for evaluation_quarter in quarters[1:]:
        fit_quarters = [quarter for quarter in quarters if quarter < evaluation_quarter]
        assert_selection_precedes_evaluation(fit_quarters, evaluation_quarter)
        fit = data.loc[data["quarter"].isin(fit_quarters)].copy()
        evaluation = data.loc[data["quarter"].eq(evaluation_quarter)].copy()
        parameters, search = select_affine_parameters(fit)
        for target in evaluation["target"].unique():
            parameters.setdefault(target, (1.0, 0.0))
        search["evaluation_quarter"] = evaluation_quarter
        search["fit_through"] = max(fit_quarters)
        search_parts.append(search)
        baseline, _ = score_available_groups(evaluation)
        calibrated = apply_affine(evaluation, parameters, "rolling_affine")
        candidate, _ = score_available_groups(calibrated)
        result_rows.append(
            {
                "evaluation_quarter": evaluation_quarter,
                "fit_through": max(fit_quarters),
                "fit_quarters": ",".join(fit_quarters),
                "baseline_score": baseline["total_score"],
                "calibrated_score": candidate["total_score"],
                "score_delta": candidate["total_score"] - baseline["total_score"],
                "baseline_one_minus_nmae": baseline["one_minus_nmae"],
                "calibrated_one_minus_nmae": candidate["one_minus_nmae"],
                "baseline_ficr": baseline["ficr"],
                "calibrated_ficr": candidate["ficr"],
                "parameters": repr(parameters),
            }
        )
    return pd.DataFrame(result_rows), pd.concat(search_parts, ignore_index=True)


SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def apply_seasonal_affine(
    predictions: pd.DataFrame,
    parameters: dict[str, dict[str, tuple[float, float]]],
    model_id: str,
) -> pd.DataFrame:
    out = predictions.copy().reset_index(drop=True)
    seasons = pd.to_datetime(out["forecast_kst_dtm"]).dt.month.map(SEASON_BY_MONTH)
    calibrated = out["y_pred_kwh"].to_numpy(dtype=float).copy()
    for index, (season, target, value) in enumerate(zip(seasons, out["target"], calibrated)):
        scale, offset = parameters.get(season, {}).get(target, (1.0, 0.0))
        calibrated[index] = max(value * scale + offset, 0.0)
    out["y_pred_kwh"] = calibrated; out["model_id"] = model_id
    return out


def rolling_seasonal_affine_backtest(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = predictions.copy()
    data["quarter"] = issue_quarter(data["forecast_kst_dtm"])
    data["season"] = pd.to_datetime(data["forecast_kst_dtm"]).dt.month.map(SEASON_BY_MONTH)
    rows, parameter_rows = [], []
    # A prior occurrence of every season exists only in the second OOF year.
    for evaluation_quarter in [quarter for quarter in sorted(data["quarter"].unique()) if quarter >= "2024Q1"]:
        fit = data.loc[data["quarter"] < evaluation_quarter]
        evaluation = data.loc[data["quarter"].eq(evaluation_quarter)]
        parameters: dict[str, dict[str, tuple[float, float]]] = {}
        for season in sorted(evaluation["season"].unique()):
            seasonal_fit = fit.loc[fit["season"].eq(season)]
            if seasonal_fit.empty:
                continue
            selected, _ = select_affine_parameters(seasonal_fit)
            parameters[season] = selected
            for target, (scale, offset) in selected.items():
                parameter_rows.append({"evaluation_quarter": evaluation_quarter, "fit_through": fit["quarter"].max(),
                                       "season": season, "target": target, "scale": scale,
                                       "offset_kwh": offset})
        baseline, _ = score_available_groups(evaluation)
        calibrated = apply_seasonal_affine(evaluation, parameters, "seasonal_affine")
        candidate, _ = score_available_groups(calibrated)
        rows.append({"evaluation_quarter": evaluation_quarter, "fit_through": fit["quarter"].max(),
                     "baseline_score": baseline["total_score"], "seasonal_score": candidate["total_score"],
                     "score_delta": candidate["total_score"] - baseline["total_score"],
                     "improved": candidate["total_score"] > baseline["total_score"]})
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows)
