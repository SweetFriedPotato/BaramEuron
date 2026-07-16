"""Nested, regularized group-specific Exp03/raw blending."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS

from .nested_rolling import ORDERED_QUARTERS, assert_nested_order, preceding_quarters
from .oof_contract import score_prediction


@dataclass(frozen=True)
class Penalties:
    lambda_global: float
    lambda_spread: float
    lambda_instability: float

    @property
    def key(self) -> str:
        return f"g{self.lambda_global:.3f}_s{self.lambda_spread:.3f}_i{self.lambda_instability:.3f}"


DEFAULT_PENALTIES = Penalties(0.005, 0.003, 0.003)


def regularization_penalty(
    weights: np.ndarray,
    penalties: Penalties,
    instability: float | np.ndarray = 0.0,
) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("group weight array must end in three groups")
    return (
        penalties.lambda_global * np.square(values - 0.4).sum(axis=-1)
        + penalties.lambda_spread * np.var(values, axis=-1)
        + penalties.lambda_instability * np.asarray(instability, dtype=float)
    )


def apply_group_weights(data: pd.DataFrame, weights: dict[str, float], column: str) -> pd.DataFrame:
    out = data.copy()
    raw_weight = out["target"].map(weights).astype(float)
    out[column] = (1.0 - raw_weight) * out["exp03_prediction"] + raw_weight * out["raw_prediction"]
    return out


def _single_group_score(part: pd.DataFrame, prediction: np.ndarray) -> float:
    if part.empty:
        return np.nan
    capacity = float(CAPACITY_KWH[str(part["target"].iloc[0])])
    actual = part["y_true_kwh"].to_numpy(dtype=float)
    valid = actual >= 0.10 * capacity
    if not valid.any():
        return np.nan
    error = np.abs(prediction[valid] - actual[valid]) / capacity
    unit_price = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual[valid] * unit_price) / np.sum(actual[valid] * 4.0))
    return 0.5 * (one_minus_nmae + ficr)


def _score_curves(data: pd.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = np.full((3, len(grid)), np.nan, dtype=float)
    available = np.zeros(3, dtype=bool)
    for group_index, target in enumerate(TARGETS):
        part = data.loc[data["target"].eq(target)]
        if part.empty:
            continue
        available[group_index] = True
        left = part["exp03_prediction"].to_numpy(dtype=float)
        right = part["raw_prediction"].to_numpy(dtype=float)
        for index, weight in enumerate(grid):
            curves[group_index, index] = _single_group_score(part, (1-weight)*left + weight*right)
    return curves, available


def search_group_weights(
    fit: pd.DataFrame,
    penalties: Penalties,
    prior_weights: list[np.ndarray] | None = None,
    coarse_grid: np.ndarray | None = None,
) -> tuple[dict[str, float], dict, pd.DataFrame]:
    coarse_grid = np.round(np.arange(0.0, 0.8001, 0.05), 3) if coarse_grid is None else coarse_grid
    prior_weights = prior_weights or []
    rows = []
    def candidate_instability(weights: np.ndarray) -> np.ndarray:
        if not prior_weights:
            return np.zeros(len(weights), dtype=float)
        history = np.vstack(prior_weights)
        count = len(history) + 1.0
        mean = (history.sum(axis=0)[None, :] + weights) / count
        variance = (
            (np.square(history).sum(axis=0)[None, :] + np.square(weights)) / count
            - np.square(mean)
        )
        return variance.mean(axis=1)

    def select_vectorized(
        axes: list[np.ndarray], curve_maps: list[dict[float, float]], available: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        meshes = np.meshgrid(*axes, indexing="ij")
        weights = np.column_stack([mesh.reshape(-1) for mesh in meshes])
        score_columns = []
        for group in np.flatnonzero(available):
            score_columns.append(
                np.asarray([curve_maps[group][float(value)] for value in weights[:, group]], dtype=float)
            )
        official = np.column_stack(score_columns).mean(axis=1)
        objective = official - regularization_penalty(
            weights, penalties, candidate_instability(weights)
        )
        index = int(np.argmax(objective))
        return weights[index], float(objective[index]), float(official[index])

    def search(grid: np.ndarray, stage: str) -> tuple[np.ndarray, float, float]:
        curves, available = _score_curves(fit, grid)
        curve_maps = [
            {float(weight): float(curves[group, index]) for index, weight in enumerate(grid)}
            for group in range(3)
        ]
        best_weights, best_objective, best_score = select_vectorized(
            [grid, grid, grid], curve_maps, available
        )
        rows.append({
            "stage": stage, "penalty_key": penalties.key,
            "lambda_global": penalties.lambda_global, "lambda_spread": penalties.lambda_spread,
            "lambda_instability": penalties.lambda_instability,
            "weight_g1": best_weights[0], "weight_g2": best_weights[1], "weight_g3": best_weights[2],
            "fit_official_score": best_score, "objective": best_objective,
            "groups_available": int(available.sum()),
        })
        return best_weights, best_objective, best_score
    coarse, _, _ = search(coarse_grid, "coarse")
    fine_axes = [
        np.round(np.arange(max(0.0, value-0.05), min(0.8, value+0.05)+1e-9, 0.01), 3)
        for value in coarse
    ]
    # Fine axes can differ, so evaluate their Cartesian product directly.
    available_groups = sorted(int(value) for value in fit["group_id"].unique())
    curve_maps = {}
    for group_id, target, axis in zip((1, 2, 3), TARGETS, fine_axes):
        part = fit.loc[fit["target"].eq(target)]
        curve_maps[group_id] = {
            float(weight): _single_group_score(
                part,
                (1-weight)*part["exp03_prediction"].to_numpy(dtype=float)
                + weight*part["raw_prediction"].to_numpy(dtype=float),
            )
            for weight in axis
        }
    available = np.asarray([group in available_groups for group in (1, 2, 3)], dtype=bool)
    ordered_curve_maps = [curve_maps[group] for group in (1, 2, 3)]
    best_weights, best_objective, best_score = select_vectorized(
        fine_axes, ordered_curve_maps, available
    )
    rows.append({
        "stage": "fine", "penalty_key": penalties.key,
        "lambda_global": penalties.lambda_global, "lambda_spread": penalties.lambda_spread,
        "lambda_instability": penalties.lambda_instability,
        "weight_g1": best_weights[0], "weight_g2": best_weights[1], "weight_g3": best_weights[2],
        "fit_official_score": best_score, "objective": best_objective,
        "groups_available": len(available_groups),
    })
    mapping = {target: float(best_weights[index]) for index, target in enumerate(TARGETS)}
    return mapping, rows[-1], pd.DataFrame(rows)


def _inner_penalty_score(data: pd.DataFrame, penalties: Penalties) -> float:
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(data["quarter"])]
    if len(quarters) < 2:
        return -np.inf
    parts, history = [], []
    for index, quarter in enumerate(quarters):
        evaluation = data.loc[data["quarter"].eq(quarter)]
        if index == 0:
            weights = {target: 0.4 for target in TARGETS}
        else:
            fit_quarters = quarters[:index]; assert_nested_order(fit_quarters, quarter)
            weights, _, _ = search_group_weights(
                data.loc[data["quarter"].isin(fit_quarters)], penalties, history
            )
            history.append(np.asarray(list(weights.values())))
        parts.append(apply_group_weights(evaluation, weights, "candidate_prediction"))
    return score_prediction(pd.concat(parts, ignore_index=True), "candidate_prediction")["total_score"]


def nested_constrained_blend(
    data: pd.DataFrame,
    penalties: list[Penalties],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(data["quarter"])]
    prediction_parts, weight_rows, search_parts = [], [], []
    selected_history: list[np.ndarray] = []
    selected_penalties: list[Penalties] = []
    for index, quarter in enumerate(quarters):
        evaluation = data.loc[data["quarter"].eq(quarter)]
        if index == 0:
            chosen = DEFAULT_PENALTIES
            weights = {target: 0.4 for target in TARGETS}
            status = "fallback_no_history"
        else:
            fit_quarters = quarters[:index]; assert_nested_order(fit_quarters, quarter)
            fit = data.loc[data["quarter"].isin(fit_quarters)]
            if index < 2:
                chosen = DEFAULT_PENALTIES
                status = "default_penalty_insufficient_inner_quarters"
            else:
                inner_scores = {candidate: _inner_penalty_score(fit, candidate) for candidate in penalties}
                chosen = max(inner_scores, key=inner_scores.get)
                status = "nested_selected"
            weights, selected, search = search_group_weights(fit, chosen, selected_history)
            search.insert(0, "evaluation_quarter", quarter); search_parts.append(search)
            selected_history.append(np.asarray([weights[target] for target in TARGETS]))
        selected_penalties.append(chosen)
        prediction = apply_group_weights(evaluation, weights, "constrained_prediction")
        prediction["selection_status"] = status; prediction_parts.append(prediction)
        weight_rows.append({
            "evaluation_quarter": quarter, "fit_quarters": repr(quarters[:index]),
            "selection_status": status, "penalty_key": chosen.key,
            "lambda_global": chosen.lambda_global, "lambda_spread": chosen.lambda_spread,
            "lambda_instability": chosen.lambda_instability,
            **{f"weight_g{group}": weights[target] for group, target in enumerate(TARGETS, 1)},
        })
    nested = pd.concat(prediction_parts, ignore_index=True)
    weights_frame = pd.DataFrame(weight_rows)
    search_frame = pd.concat(search_parts, ignore_index=True) if search_parts else pd.DataFrame()
    chosen_final = Counter(value.key for value in selected_penalties[1:]).most_common(1)[0][0]
    final_penalty = next(value for value in penalties if value.key == chosen_final)
    final_weights, final_row, final_search = search_group_weights(
        data, final_penalty, selected_history
    )
    final_search.insert(0, "evaluation_quarter", "final_all_oof"); search_frame = pd.concat(
        [search_frame, final_search], ignore_index=True
    )
    summary = {
        "final_penalties": final_penalty.__dict__, "final_weights": final_weights,
        "nested_score": score_prediction(nested, "constrained_prediction"),
        "weight_mean": {f"g{group}": float(weights_frame[f"weight_g{group}"].mean()) for group in (1,2,3)},
        "weight_std": {f"g{group}": float(weights_frame[f"weight_g{group}"].std()) for group in (1,2,3)},
    }
    return nested, weights_frame, search_frame, summary


def penalty_grid(config: dict) -> list[Penalties]:
    return [
        Penalties(float(a), float(b), float(c))
        for a, b, c in itertools.product(
            config["lambda_global"], config["lambda_spread"], config["lambda_instability"]
        )
    ]
