"""Nested Ridge and small CatBoost residual correction models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baram.constants import CAPACITY_KWH, TARGETS

from .constrained_blend import _single_group_score
from .nested_rolling import ORDERED_QUARTERS, assert_nested_order
from .oof_contract import score_prediction


def bounded_correction(
    predicted_residual: np.ndarray,
    target: str,
    shrinkage: float,
    bound_fraction: float,
) -> np.ndarray:
    bound = float(bound_fraction) * float(CAPACITY_KWH[target])
    return float(shrinkage) * np.clip(np.asarray(predicted_residual, dtype=float), -bound, bound)


def corrected_group_score(
    validation: pd.DataFrame,
    predicted_residual: np.ndarray,
    prediction_column: str,
    shrinkage: float,
    bound_fraction: float,
) -> float:
    target = str(validation["target"].iloc[0])
    prediction = validation[prediction_column].to_numpy(dtype=float) + bounded_correction(
        predicted_residual, target, shrinkage, bound_fraction
    )
    return _single_group_score(validation, prediction)


@dataclass
class FinalStacker:
    kind: str
    models: dict[str, list[Any]]
    parameters: dict[str, dict]
    feature_columns: list[str]
    prediction_column: str


def _available_group_quarters(frame: pd.DataFrame, target: str) -> list[str]:
    values = set(frame.loc[frame["target"].eq(target), "quarter"])
    return [quarter for quarter in ORDERED_QUARTERS if quarter in values]


def nested_ridge_stacker(
    frame: pd.DataFrame,
    feature_columns: list[str],
    base_column: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, FinalStacker]:
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(frame["quarter"])]
    output_parts, metric_rows = [], []
    selected_parameters: dict[str, list[dict]] = {target: [] for target in TARGETS}
    for outer_index, quarter in enumerate(quarters):
        evaluation = frame.loc[frame["quarter"].eq(quarter)].copy()
        evaluation["ridge_prediction"] = evaluation[base_column]
        fit_quarters = quarters[:outer_index]
        if fit_quarters:
            assert_nested_order(fit_quarters, quarter)
        for target in TARGETS:
            eval_group = evaluation.loc[evaluation["target"].eq(target)]
            available = [value for value in fit_quarters if value in _available_group_quarters(frame, target)]
            if eval_group.empty or len(available) < 2:
                metric_rows.append({"evaluation_quarter": quarter, "target": target,
                                    "status": "fallback_insufficient_inner_quarters"})
                continue
            inner_valid_quarter = available[-1]
            inner_train_quarters = available[:-1]
            train = frame.loc[frame["target"].eq(target) & frame["quarter"].isin(inner_train_quarters)]
            validation = frame.loc[frame["target"].eq(target) & frame["quarter"].eq(inner_valid_quarter)]
            residual = train["y_true_kwh"].to_numpy(dtype=float) - train[base_column].to_numpy(dtype=float)
            best = None
            for alpha in config["alphas"]:
                model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
                model.fit(train[feature_columns], residual)
                predicted = model.predict(validation[feature_columns])
                for shrinkage in config["shrinkages"]:
                    score = corrected_group_score(
                        validation, predicted, base_column, float(shrinkage),
                        float(config["final_bounds"][target]),
                    )
                    row = {"alpha": float(alpha), "shrinkage": float(shrinkage), "inner_score": score}
                    if best is None or (row["inner_score"], -row["alpha"], -row["shrinkage"]) > (
                        best["inner_score"], -best["alpha"], -best["shrinkage"]
                    ):
                        best = row
            outer_train = frame.loc[frame["target"].eq(target) & frame["quarter"].isin(available)]
            outer_residual = outer_train["y_true_kwh"].to_numpy(dtype=float) - outer_train[base_column].to_numpy(dtype=float)
            model = make_pipeline(StandardScaler(), Ridge(alpha=best["alpha"]))
            model.fit(outer_train[feature_columns], outer_residual)
            predicted = model.predict(eval_group[feature_columns])
            correction = bounded_correction(
                predicted, target, best["shrinkage"], float(config["final_bounds"][target])
            )
            evaluation.loc[eval_group.index, "ridge_prediction"] = eval_group[base_column] + correction
            parameters = {
                "evaluation_quarter": quarter, "target": target, "status": "nested_selected",
                "inner_train_quarters": repr(inner_train_quarters),
                "inner_validation_quarter": inner_valid_quarter,
                "alpha": best["alpha"], "shrinkage": best["shrinkage"],
                "bound_fraction": float(config["final_bounds"][target]),
                "inner_score": best["inner_score"],
                "correction_p95_fraction": float(np.quantile(np.abs(correction), .95) / CAPACITY_KWH[target]),
            }
            metric_rows.append(parameters); selected_parameters[target].append(parameters)
        output_parts.append(evaluation)
    nested = pd.concat(output_parts, ignore_index=True)
    final_models, final_parameters = {}, {}
    for target in TARGETS:
        choices = selected_parameters[target]
        if choices:
            alpha = Counter(value["alpha"] for value in choices).most_common(1)[0][0]
            shrinkage = Counter(value["shrinkage"] for value in choices).most_common(1)[0][0]
        else:
            alpha, shrinkage = 10.0, 0.25
        train = frame.loc[frame["target"].eq(target)]
        residual = train["y_true_kwh"].to_numpy(dtype=float) - train[base_column].to_numpy(dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(train[feature_columns], residual)
        final_models[target] = [model]
        final_parameters[target] = {
            "alpha": float(alpha), "shrinkage": float(shrinkage),
            "bound_fraction": float(config["final_bounds"][target]),
        }
    return nested, pd.DataFrame(metric_rows), FinalStacker(
        "ridge", final_models, final_parameters, feature_columns, base_column
    )


def nested_catboost_stacker(
    frame: pd.DataFrame,
    feature_columns: list[str],
    base_column: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, FinalStacker]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("catboost is required for Stage C") from exc
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(frame["quarter"])]
    output_parts, metric_rows = [], []
    selected_parameters: dict[str, list[dict]] = {target: [] for target in TARGETS}
    for outer_index, quarter in enumerate(quarters):
        evaluation = frame.loc[frame["quarter"].eq(quarter)].copy()
        evaluation["catboost_prediction"] = evaluation[base_column]
        fit_quarters = quarters[:outer_index]
        if fit_quarters: assert_nested_order(fit_quarters, quarter)
        for target in TARGETS:
            eval_group = evaluation.loc[evaluation["target"].eq(target)]
            available = [value for value in fit_quarters if value in _available_group_quarters(frame, target)]
            if eval_group.empty or len(available) < 2:
                metric_rows.append({"evaluation_quarter": quarter, "target": target,
                                    "status": "fallback_insufficient_inner_quarters"})
                continue
            inner_valid_quarter = available[-1]; inner_train_quarters = available[:-1]
            train = frame.loc[frame["target"].eq(target) & frame["quarter"].isin(inner_train_quarters)]
            validation = frame.loc[frame["target"].eq(target) & frame["quarter"].eq(inner_valid_quarter)]
            residual = train["y_true_kwh"].to_numpy(dtype=float) - train[base_column].to_numpy(dtype=float)
            valid_residual = validation["y_true_kwh"].to_numpy(dtype=float) - validation[base_column].to_numpy(dtype=float)
            best = None
            for depth in config["depths"]:
                model = CatBoostRegressor(
                    depth=int(depth), iterations=int(config["iterations"]),
                    learning_rate=float(config["learning_rate"]), l2_leaf_reg=float(config["l2_leaf_reg"]),
                    loss_function=config["loss_function"], random_seed=42, verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    train[feature_columns], residual,
                    eval_set=(validation[feature_columns], valid_residual),
                    early_stopping_rounds=int(config["early_stopping_rounds"]), verbose=False,
                )
                predicted = model.predict(validation[feature_columns])
                iterations = max(20, int(model.get_best_iteration()) + 1)
                for shrinkage in config["shrinkages"]:
                    score = corrected_group_score(
                        validation, predicted, base_column, float(shrinkage),
                        float(config["final_bounds"][target]),
                    )
                    row = {"depth": int(depth), "iterations": iterations,
                           "shrinkage": float(shrinkage), "inner_score": score}
                    if best is None or row["inner_score"] > best["inner_score"] + 1e-15:
                        best = row
            outer_train = frame.loc[frame["target"].eq(target) & frame["quarter"].isin(available)]
            outer_residual = outer_train["y_true_kwh"].to_numpy(dtype=float) - outer_train[base_column].to_numpy(dtype=float)
            seed_models, seed_predictions = [], []
            for seed in config["seeds"]:
                model = CatBoostRegressor(
                    depth=best["depth"], iterations=best["iterations"],
                    learning_rate=float(config["learning_rate"]), l2_leaf_reg=float(config["l2_leaf_reg"]),
                    loss_function=config["loss_function"], random_seed=int(seed), verbose=False,
                    allow_writing_files=False,
                )
                model.fit(outer_train[feature_columns], outer_residual, verbose=False)
                seed_models.append(model); seed_predictions.append(model.predict(eval_group[feature_columns]))
            predicted = np.mean(seed_predictions, axis=0)
            correction = bounded_correction(
                predicted, target, best["shrinkage"], float(config["final_bounds"][target])
            )
            evaluation.loc[eval_group.index, "catboost_prediction"] = eval_group[base_column] + correction
            parameters = {
                "evaluation_quarter": quarter, "target": target, "status": "nested_selected",
                "inner_train_quarters": repr(inner_train_quarters),
                "inner_validation_quarter": inner_valid_quarter,
                **best, "bound_fraction": float(config["final_bounds"][target]),
                "correction_p95_fraction": float(np.quantile(np.abs(correction), .95) / CAPACITY_KWH[target]),
                "seed_prediction_std": float(np.mean(np.std(seed_predictions, axis=0))),
            }
            metric_rows.append(parameters); selected_parameters[target].append(parameters)
        output_parts.append(evaluation)
    nested = pd.concat(output_parts, ignore_index=True)
    final_models, final_parameters = {}, {}
    for target in TARGETS:
        choices = selected_parameters[target]
        if choices:
            depth = Counter(value["depth"] for value in choices).most_common(1)[0][0]
            iterations = int(np.median([value["iterations"] for value in choices]))
            shrinkage = Counter(value["shrinkage"] for value in choices).most_common(1)[0][0]
        else:
            depth, iterations, shrinkage = 4, 100, 0.25
        train = frame.loc[frame["target"].eq(target)]
        residual = train["y_true_kwh"].to_numpy(dtype=float) - train[base_column].to_numpy(dtype=float)
        models = []
        for seed in config["seeds"]:
            model = CatBoostRegressor(
                depth=int(depth), iterations=int(iterations), learning_rate=float(config["learning_rate"]),
                l2_leaf_reg=float(config["l2_leaf_reg"]), loss_function=config["loss_function"],
                random_seed=int(seed), verbose=False, allow_writing_files=False,
            )
            model.fit(train[feature_columns], residual, verbose=False); models.append(model)
        final_models[target] = models
        final_parameters[target] = {
            "depth": int(depth), "iterations": int(iterations), "shrinkage": float(shrinkage),
            "bound_fraction": float(config["final_bounds"][target]),
        }
    return nested, pd.DataFrame(metric_rows), FinalStacker(
        "catboost", final_models, final_parameters, feature_columns, base_column
    )


def apply_final_stacker(stacker: FinalStacker, frame: pd.DataFrame, output_column: str) -> pd.DataFrame:
    out = frame.copy(); out[output_column] = out[stacker.prediction_column]
    for target in TARGETS:
        indices = out.index[out["target"].eq(target)]
        if len(indices) == 0: continue
        predictions = [model.predict(out.loc[indices, stacker.feature_columns]) for model in stacker.models[target]]
        residual = np.mean(predictions, axis=0)
        parameters = stacker.parameters[target]
        correction = bounded_correction(
            residual, target, parameters["shrinkage"], parameters["bound_fraction"]
        )
        out.loc[indices, output_column] = out.loc[indices, stacker.prediction_column] + correction
    return out
