"""Load exp01/exp02 out-of-fold predictions through one strict contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from official.dacon_baram_metric.metric import CAPACITY_KWH, TARGET_COLS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP01_PREDICTIONS = PROJECT_ROOT / "experiments/exp01_catboost_physics/outputs/predictions"
EXP02_PREDICTIONS = PROJECT_ROOT / "experiments/exp02_daily_tcn_scada_aux/outputs/predictions"
TIME_COL = "forecast_kst_dtm"
KEY_COLS = ["fold", TIME_COL, "target", "group_id"]
GROUP_IDS = {target: index + 1 for index, target in enumerate(TARGET_COLS)}


def validate_prediction_contract(frame: pd.DataFrame, *, include_seed: bool = False) -> pd.DataFrame:
    required = {*KEY_COLS, "y_true_kwh", "y_pred_kwh", "model_id"}
    if include_seed:
        required.add("seed")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction contract columns missing: {missing}")
    out = frame.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    unknown = sorted(set(out["target"]) - set(TARGET_COLS))
    if unknown:
        raise ValueError(f"unknown prediction targets: {unknown}")
    expected_group = out["target"].map(GROUP_IDS)
    if not expected_group.eq(out["group_id"].astype(int)).all():
        raise ValueError("target/group_id mapping is inconsistent")
    numeric = out[["y_true_kwh", "y_pred_kwh"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("prediction contains NaN or infinity")
    duplicate_keys = [*KEY_COLS, "model_id"] + (["seed"] if include_seed else [])
    if out.duplicated(duplicate_keys).any():
        raise ValueError(f"duplicate prediction keys: {duplicate_keys}")
    if not out["fold"].isin(["fold_a", "fold_b"]).all():
        raise ValueError("unexpected validation fold")
    out["capacity_kwh"] = out["target"].map(CAPACITY_KWH).astype(float)
    return out.sort_values(duplicate_keys).reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required prediction artifact is missing: {path}")
    return pd.read_csv(path)


def load_exp01_model(model_id: str) -> pd.DataFrame:
    parts = []
    for fold in ("fold_a", "fold_b"):
        data = _read_csv(EXP01_PREDICTIONS / f"{fold}_predictions.csv")
        part = data.loc[data["experiment_id"].eq(model_id)].copy()
        if part.empty:
            raise ValueError(f"exp01 model not found: {model_id}/{fold}")
        part["model_id"] = model_id
        parts.append(part)
    return validate_prediction_contract(pd.concat(parts, ignore_index=True))


def load_neural_seeds(model_id: str) -> pd.DataFrame:
    data = _read_csv(EXP02_PREDICTIONS / "tcn_seed_predictions.csv")
    part = data.loc[data["experiment_id"].eq(model_id)].copy()
    if part.empty:
        raise ValueError(f"exp02 seed model not found: {model_id}")
    part["model_id"] = model_id
    return validate_prediction_contract(part, include_seed=True)


def aggregate_seeds(seed_predictions: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    validate_prediction_contract(seed_predictions, include_seed=True)
    keys = KEY_COLS
    truth_counts = seed_predictions.groupby(keys, sort=False)["y_true_kwh"].nunique(dropna=False)
    if not truth_counts.eq(1).all():
        raise ValueError("seed predictions disagree on validation truth")
    grouped = seed_predictions.groupby(keys, sort=False)["y_pred_kwh"]
    if method == "mean":
        prediction = grouped.mean()
    elif method == "median":
        prediction = grouped.median()
    elif method == "trimmed_mean":
        prediction = grouped.apply(
            lambda values: float(np.mean(np.sort(values.to_numpy(dtype=float))[1:-1]))
            if len(values) > 2
            else float(np.mean(values))
        )
    else:
        raise ValueError(f"unknown seed aggregation: {method}")
    truth = seed_predictions.groupby(keys, sort=False)["y_true_kwh"].first()
    out = pd.concat([truth, prediction.rename("y_pred_kwh")], axis=1).reset_index()
    out["model_id"] = f"{seed_predictions['model_id'].iloc[0]}_{method}"
    out["seed_aggregation"] = method
    return validate_prediction_contract(out)


def load_neural_ensemble(model_id: str) -> pd.DataFrame:
    data = _read_csv(EXP02_PREDICTIONS / "tcn_ensemble_predictions.csv")
    part = data.loc[data["experiment_id"].eq(model_id)].copy()
    if part.empty:
        raise ValueError(f"exp02 ensemble model not found: {model_id}")
    part["model_id"] = model_id
    return validate_prediction_contract(part)


def load_best_blend() -> pd.DataFrame:
    data = _read_csv(EXP02_PREDICTIONS / "best_blend_predictions.csv")
    data["model_id"] = "cat025_tcn075"
    return validate_prediction_contract(data)


def load_existing_predictions() -> pd.DataFrame:
    models = [load_exp01_model("rf_reference"), load_exp01_model("catboost_selected")]
    for model_id in ("mlp_pointwise", "tcn_plain", "tcn_aux_005", "tcn_aux_015"):
        models.append(load_neural_ensemble(model_id))
    models.append(load_best_blend())
    return pd.concat(models, ignore_index=True)


def prediction_inventory(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["model_id", "fold", "target", "group_id"], sort=True)
        .agg(
            rows=(TIME_COL, "size"),
            first_timestamp=(TIME_COL, "min"),
            last_timestamp=(TIME_COL, "max"),
            unique_timestamps=(TIME_COL, "nunique"),
            truth_missing=("y_true_kwh", lambda values: int(values.isna().sum())),
            prediction_missing=("y_pred_kwh", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )


def align_models(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_data = left[[*KEY_COLS, "y_true_kwh", "y_pred_kwh"]].rename(
        columns={"y_pred_kwh": "left_prediction"}
    )
    right_data = right[[*KEY_COLS, "y_true_kwh", "y_pred_kwh"]].rename(
        columns={"y_true_kwh": "right_truth", "y_pred_kwh": "right_prediction"}
    )
    merged = left_data.merge(right_data, on=KEY_COLS, how="inner", validate="one_to_one")
    if len(merged) != len(left_data) or len(merged) != len(right_data):
        raise ValueError("prediction key sets differ")
    # exp01 CSVs were written with three-decimal truth while exp02 retained full
    # label precision. Accept only that bounded serialization difference and use
    # the higher-precision exp02 truth for all downstream official scoring.
    if not np.allclose(merged["y_true_kwh"], merged["right_truth"], rtol=0.0, atol=0.002):
        raise ValueError("aligned predictions disagree on truth")
    merged["y_true_kwh"] = merged["right_truth"]
    return merged.drop(columns="right_truth")
