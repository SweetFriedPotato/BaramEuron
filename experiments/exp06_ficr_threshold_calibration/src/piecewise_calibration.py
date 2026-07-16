"""Bounded piecewise-affine calibration fitted only on earlier rolling OOF."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS
from experiments.exp05_cross_group_transfer.src.constrained_blend import _single_group_score
from experiments.exp05_cross_group_transfer.src.nested_rolling import ORDERED_QUARTERS, assert_nested_order
from experiments.exp05_cross_group_transfer.src.oof_contract import score_prediction
from .threshold_audit import reward_from_error


@dataclass(frozen=True)
class CalibrationPenalty:
    identity: float
    smoothness: float
    instability: float

    @property
    def key(self) -> str:
        return f"i{self.identity:.4f}_s{self.smoothness:.4f}_q{self.instability:.4f}"


@dataclass
class PiecewiseModel:
    scheme: str
    boundaries: dict[str, list[float]]
    parameters: pd.DataFrame
    penalty: CalibrationPenalty


def fit_band_boundaries(data: pd.DataFrame, scheme: str) -> dict[str, list[float]]:
    result = {}
    for target in TARGETS:
        part = data.loc[data["target"].eq(target)]
        prediction_cf = np.clip(
            part["global_blend_prediction"].to_numpy(dtype=float) / CAPACITY_KWH[target], 0.0, 1.2
        )
        if scheme == "two_band":
            boundaries = [0.50]
        elif scheme == "physical_three":
            boundaries = [0.30, 0.70]
        elif scheme == "quantile_three":
            boundaries = np.quantile(prediction_cf, [1/3, 2/3]).tolist() if len(part) else [0.30, 0.70]
        else:
            raise ValueError(f"unknown piecewise scheme: {scheme}")
        result[target] = [float(value) for value in boundaries]
    return result


def assign_band(prediction: np.ndarray, target: str, boundaries: dict[str, list[float]]) -> np.ndarray:
    prediction_cf = np.clip(np.asarray(prediction, dtype=float) / CAPACITY_KWH[target], 0.0, 1.2)
    return np.digitize(prediction_cf, boundaries[target], right=False)


def identity_parameters(boundaries: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"target": target, "bin": bin_index, "scale": 1.0, "offset_fraction": 0.0}
        for target in TARGETS for bin_index in range(len(boundaries[target]) + 1)
    ])


def calibration_regularization(
    parameters: pd.DataFrame,
    prior: pd.DataFrame | None = None,
) -> dict[str, float]:
    identity = float(np.mean(
        np.square(parameters["scale"].to_numpy(dtype=float) - 1.0)
        + np.square(parameters["offset_fraction"].to_numpy(dtype=float))
    ))
    smooth_values = []
    for _, part in parameters.sort_values(["target", "bin"]).groupby("target", sort=True):
        smooth_values.extend(np.square(np.diff(part["scale"])).tolist())
        smooth_values.extend(np.square(np.diff(part["offset_fraction"])).tolist())
    smoothness = float(np.mean(smooth_values)) if smooth_values else 0.0
    instability = 0.0
    if prior is not None and not prior.empty:
        aligned = parameters.merge(
            prior[["target", "bin", "scale", "offset_fraction"]],
            on=["target", "bin"], suffixes=("", "_prior"), how="inner",
        )
        if not aligned.empty:
            instability = float(np.mean(
                np.square(aligned["scale"] - aligned["scale_prior"])
                + np.square(aligned["offset_fraction"] - aligned["offset_fraction_prior"])
            ))
    return {"identity": identity, "smoothness": smoothness, "instability": instability}


def apply_piecewise(
    data: pd.DataFrame,
    model: PiecewiseModel,
    output_column: str = "piecewise_prediction",
) -> pd.DataFrame:
    out = data.copy(); out[output_column] = out["global_blend_prediction"]
    for target in TARGETS:
        indices = out.index[out["target"].eq(target)]
        if len(indices) == 0:
            continue
        base = out.loc[indices, "global_blend_prediction"].to_numpy(dtype=float)
        bins = assign_band(base, target, model.boundaries)
        params = model.parameters.loc[model.parameters["target"].eq(target)].set_index("bin")
        scale = np.asarray([params.loc[value, "scale"] for value in bins], dtype=float)
        offset = np.asarray([params.loc[value, "offset_fraction"] for value in bins], dtype=float)
        out.loc[indices, output_column] = scale * base + offset * CAPACITY_KWH[target]
    return out


def _group_objective(
    part: pd.DataFrame,
    target: str,
    boundaries: dict[str, list[float]],
    parameters: pd.DataFrame,
    penalty: CalibrationPenalty,
    prior: pd.DataFrame | None,
) -> tuple[float, float, dict[str, float]]:
    base = part["global_blend_prediction"].to_numpy(dtype=float)
    bins = assign_band(base, target, boundaries)
    target_parameters = parameters.loc[parameters["target"].eq(target)].set_index("bin")
    scale = target_parameters.loc[bins, "scale"].to_numpy(dtype=float)
    offset = target_parameters.loc[bins, "offset_fraction"].to_numpy(dtype=float)
    prediction = scale * base + offset * CAPACITY_KWH[target]
    score = _single_group_score(part, prediction)
    target_parameters = target_parameters.reset_index()
    target_prior = None if prior is None else prior.loc[prior["target"].eq(target)]
    regularization = calibration_regularization(target_parameters, target_prior)
    value = score - penalty.identity * regularization["identity"] \
        - penalty.smoothness * regularization["smoothness"] \
        - penalty.instability * regularization["instability"]
    return float(value), float(score), regularization


def fit_piecewise(
    data: pd.DataFrame,
    scheme: str,
    penalty: CalibrationPenalty,
    config: dict,
    prior_parameters: pd.DataFrame | None = None,
) -> tuple[PiecewiseModel, pd.DataFrame]:
    boundaries = fit_band_boundaries(data, scheme)
    parameters = identity_parameters(boundaries)
    rows = []
    coarse_scales = np.round(np.arange(
        config["scale_bounds"][0], config["scale_bounds"][1] + 1e-12, config["coarse_scale_step"]
    ), 6)
    coarse_offsets = np.round(np.arange(
        config["offset_fraction_bounds"][0], config["offset_fraction_bounds"][1] + 1e-12,
        config["coarse_offset_step"],
    ), 6)
    for target in TARGETS:
        part = data.loc[data["target"].eq(target)]
        if part.empty:
            continue
        capacity = float(CAPACITY_KWH[target])
        base_values = part["global_blend_prediction"].to_numpy(dtype=float)
        actual_values = part["y_true_kwh"].to_numpy(dtype=float)
        official_mask = actual_values >= 0.10 * capacity
        bin_values = assign_band(base_values, target, boundaries)
        for stage in ("coarse", "fine"):
            for _ in range(int(config.get("coordinate_passes", 2))):
                for bin_index in range(len(boundaries[target]) + 1):
                    current = parameters.loc[
                        parameters["target"].eq(target) & parameters["bin"].eq(bin_index)
                    ].iloc[0]
                    if stage == "coarse":
                        scales, offsets = coarse_scales, coarse_offsets
                    else:
                        scales = np.round(np.arange(
                            max(config["scale_bounds"][0], current.scale-config["fine_scale_radius"]),
                            min(config["scale_bounds"][1], current.scale+config["fine_scale_radius"])+1e-12,
                            config["fine_scale_step"],
                        ), 6)
                        offsets = np.round(np.arange(
                            max(config["offset_fraction_bounds"][0], current.offset_fraction-config["fine_offset_radius"]),
                            min(config["offset_fraction_bounds"][1], current.offset_fraction+config["fine_offset_radius"])+1e-12,
                            config["fine_offset_step"],
                        ), 6)
                    combinations = list(itertools.product(scales, offsets))
                    candidate_scales = np.asarray([value[0] for value in combinations], dtype=float)
                    candidate_offsets = np.asarray([value[1] for value in combinations], dtype=float)
                    target_params = parameters.loc[parameters["target"].eq(target)].set_index("bin")
                    current_scale = target_params.loc[bin_values, "scale"].to_numpy(dtype=float)
                    current_offset = target_params.loc[bin_values, "offset_fraction"].to_numpy(dtype=float)
                    current_prediction = current_scale * base_values + current_offset * capacity
                    coordinate_mask = official_mask & (bin_values == bin_index)
                    fixed_mask = official_mask & (bin_values != bin_index)
                    candidate_prediction = (
                        base_values[coordinate_mask, None] * candidate_scales[None, :]
                        + capacity * candidate_offsets[None, :]
                    )
                    candidate_error = np.abs(candidate_prediction - actual_values[coordinate_mask, None]) / capacity
                    fixed_error_sum = float(
                        np.abs(current_prediction[fixed_mask] - actual_values[fixed_mask]).sum() / capacity
                    )
                    nmae = (fixed_error_sum + candidate_error.sum(axis=0)) / max(int(official_mask.sum()), 1)
                    fixed_error = np.abs(current_prediction[fixed_mask] - actual_values[fixed_mask]) / capacity
                    fixed_earned = float(np.sum(actual_values[fixed_mask] * reward_from_error(fixed_error)))
                    candidate_earned = np.sum(
                        actual_values[coordinate_mask, None] * reward_from_error(candidate_error), axis=0
                    )
                    max_settlement = float(np.sum(actual_values[official_mask] * 4.0))
                    candidate_score = 0.5 * (1.0 - nmae) + 0.5 * (
                        fixed_earned + candidate_earned
                    ) / max_settlement
                    best = None
                    mask = parameters["target"].eq(target) & parameters["bin"].eq(bin_index)
                    for candidate_index, (scale, offset) in enumerate(combinations):
                        candidate = parameters.copy()
                        candidate.loc[mask, ["scale", "offset_fraction"]] = [float(scale), float(offset)]
                        target_candidate = candidate.loc[candidate["target"].eq(target)]
                        target_prior = None if prior_parameters is None else prior_parameters.loc[
                            prior_parameters["target"].eq(target)
                        ]
                        regularization = calibration_regularization(target_candidate, target_prior)
                        score = float(candidate_score[candidate_index])
                        objective = score - penalty.identity * regularization["identity"] \
                            - penalty.smoothness * regularization["smoothness"] \
                            - penalty.instability * regularization["instability"]
                        key = (objective, -abs(scale-1.0), -abs(offset))
                        if best is None or key > best[0]:
                            best = (key, float(scale), float(offset), score, regularization)
                    parameters.loc[mask, ["scale", "offset_fraction"]] = [best[1], best[2]]
            rows.append({
                "stage": stage, "scheme": scheme, "penalty_key": penalty.key,
                "target": target, "fit_rows": len(part),
                **{f"boundary_{index}": value for index, value in enumerate(boundaries[target])},
                **best[4], "fit_group_score": best[3],
            })
    model = PiecewiseModel(scheme, boundaries, parameters.reset_index(drop=True), penalty)
    return model, pd.DataFrame(rows)


def penalty_grid(config: dict) -> list[CalibrationPenalty]:
    return [CalibrationPenalty(*map(float, values)) for values in itertools.product(
        config["identity_penalties"], config["smoothness_penalties"], config["instability_penalties"]
    )]


def nested_piecewise_selection(
    data: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PiecewiseModel]:
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(data["quarter"])]
    candidates = [(scheme, penalty) for scheme in config["schemes"] for penalty in penalty_grid(config)]
    prediction_parts, score_rows, search_rows, selected_keys = [], [], [], []
    prior_parameters = None
    for outer_index, quarter in enumerate(quarters):
        evaluation = data.loc[data["quarter"].eq(quarter)].copy()
        if outer_index == 0:
            boundaries = fit_band_boundaries(evaluation, "physical_three")
            model = PiecewiseModel("physical_three", boundaries, identity_parameters(boundaries), candidates[0][1])
            status = "fallback_no_history"; inner_score = np.nan
        else:
            fit_quarters = quarters[:outer_index]; assert_nested_order(fit_quarters, quarter)
            fit = data.loc[data["quarter"].isin(fit_quarters)]
            if outer_index == 1:
                scheme, penalty = "physical_three", candidates[0][1]
                status = "default_insufficient_inner_quarters"; inner_score = np.nan
            else:
                inner_valid_quarter = fit_quarters[-1]
                inner_train = data.loc[data["quarter"].isin(fit_quarters[:-1])]
                inner_valid = data.loc[data["quarter"].eq(inner_valid_quarter)]
                best = None
                for scheme_candidate, penalty_candidate in candidates:
                    inner_model, _ = fit_piecewise(inner_train, scheme_candidate, penalty_candidate, config)
                    predicted = apply_piecewise(inner_valid, inner_model, "inner_prediction")
                    metric = score_prediction(predicted, "inner_prediction")["total_score"]
                    search_rows.append({
                        "evaluation_quarter": quarter,
                        "inner_validation_quarter": inner_valid_quarter,
                        "scheme": scheme_candidate, "penalty_key": penalty_candidate.key,
                        "inner_score": metric,
                    })
                    key = (metric, -abs(penalty_candidate.identity), scheme_candidate == "physical_three")
                    if best is None or key > best[0]:
                        best = (key, scheme_candidate, penalty_candidate, metric)
                _, scheme, penalty, inner_score = best; status = "nested_selected"
            model, fit_search = fit_piecewise(fit, scheme, penalty, config, prior_parameters)
            fit_search.insert(0, "evaluation_quarter", quarter)
            search_rows.extend(fit_search.to_dict("records"))
            prior_parameters = model.parameters.copy()
            selected_keys.append((scheme, penalty.key))
        predicted = apply_piecewise(evaluation, model, "piecewise_prediction")
        predicted["selection_status"] = status; prediction_parts.append(predicted)
        metric = score_prediction(predicted, "piecewise_prediction")
        changes = (
            predicted["piecewise_prediction"] - predicted["global_blend_prediction"]
        ).abs() / predicted["capacity_kwh"]
        score_rows.append({
            "evaluation_quarter": quarter, "selection_status": status,
            "fit_quarters": repr(quarters[:outer_index]), "scheme": model.scheme,
            "penalty_key": model.penalty.key, "inner_score": inner_score,
            **metric, "change_p95_fraction": float(changes.quantile(.95)),
            "parameters": model.parameters.to_json(orient="records"),
            "boundaries": repr(model.boundaries),
        })
    nested = pd.concat(prediction_parts, ignore_index=True)
    selected = Counter(selected_keys).most_common(1)[0][0] if selected_keys else ("physical_three", candidates[0][1].key)
    final_penalty = next(value for value in penalty_grid(config) if value.key == selected[1])
    final_model, final_search = fit_piecewise(data, selected[0], final_penalty, config, prior_parameters)
    final_search.insert(0, "evaluation_quarter", "final_all_oof")
    search_rows.extend(final_search.to_dict("records"))
    return nested, pd.DataFrame(score_rows), pd.DataFrame(search_rows), final_model
