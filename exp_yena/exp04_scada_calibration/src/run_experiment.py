from __future__ import annotations

import argparse
import copy
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
import yaml

from baram.constants import TIME_COL
from baram.data import load_sample_submission
from baram.feature_builder import get_features_for_group, load_raw_feature_artifacts
from baram.preprocessing import fit_tree_preprocessor
from baram.submission import create_submission, postprocess
from baram.validation import split_labeled_table
from experiments.exp02_catboost_feature.src.run_experiment import (
    FOLD_SPECS,
    apply_feature_drop,
    calculate_group_metrics,
    calculate_oof_metrics,
    load_dropped_features,
)
from shared.constants import CAPACITY_KWH

from .feature_engineering import (
    add_direction_interactions,
    add_lead_time_interactions,
    build_direction_table,
    build_feature_pipeline,
    select_group_direction_features,
)
from .config import load_experiment_config
from .scada import (
    AuxiliaryScadaModel,
    EmpiricalPowerCurve,
    OffsetCalibrator,
    cross_fitted_offsets,
    cross_fitted_power_curve,
    cross_fitted_scada_predictions,
    load_hourly_scada,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp04: leak-safe SCADA calibration ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--no-finalize", action="store_true")
    return parser.parse_args()


def experiment_flags(config: dict) -> dict:
    defaults = {
        "lead_interactions": False,
        "direction_features": False,
        "scada_offset": False,
        "predicted_scada": False,
        "power_curve": False,
        "sample_weighting": False,
        "two_stage": False,
    }
    defaults.update(config.get("experiment", {}))
    if defaults["power_curve"] and not defaults["predicted_scada"]:
        raise ValueError("power_curve requires predicted_scada=true")
    if defaults["sample_weighting"] and defaults["two_stage"]:
        raise ValueError("sample_weighting and two_stage must be evaluated in separate runs")
    return defaults


def model_params(config: dict, feature_names: list[str], iterations: int | None = None) -> dict:
    params = dict(config["model"]["params"])
    if "n_estimators" in params:
        params["iterations"] = params.pop("n_estimators")
    if iterations is not None:
        params["iterations"] = int(iterations)
    params.setdefault("iterations", 2000)
    params.setdefault("random_seed", 42)
    params["verbose"] = False
    params.pop("monotone_constraints", None)
    return params


def fit_regressor(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame | None,
    val_y: pd.Series | None,
    params: dict,
    config: dict,
    weights: np.ndarray | None = None,
):
    train_pool = cb.Pool(train_x, train_y, weight=weights)
    try:
        gpu_params = dict(params, task_type="GPU")
        model = cb.CatBoostRegressor(**gpu_params)
        fit_args = {"verbose": False}
        if val_x is not None:
            fit_args.update(eval_set=cb.Pool(val_x, val_y), early_stopping_rounds=150)
        model.fit(train_pool, **fit_args)
        return model
    except Exception as error:
        if config.get("require_gpu", False):
            raise RuntimeError("CatBoost GPU training failed and CPU fallback is disabled") from error
        cpu_params = dict(params, task_type="CPU")
        model = cb.CatBoostRegressor(**cpu_params)
        fit_args = {"verbose": False}
        if val_x is not None:
            fit_args.update(eval_set=cb.Pool(val_x, val_y), early_stopping_rounds=150)
        model.fit(train_pool, **fit_args)
        return model


def fit_predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    other_x: pd.DataFrame,
    other_y: pd.Series | None,
    group_id: int,
    config: dict,
    flags: dict,
    iterations: int | None = None,
):
    params = model_params(config, train_x.columns.tolist(), iterations)
    threshold = CAPACITY_KWH[f"kpx_group_{group_id}"] * 0.10
    if flags["two_stage"]:
        classifier_params = dict(config.get("two_stage", {}).get("classifier_params", {}))
        classifier_params.setdefault("iterations", 500)
        classifier_params.setdefault("depth", 7)
        classifier_params.setdefault("learning_rate", 0.05)
        classifier_params.setdefault("loss_function", "Logloss")
        classifier_params.setdefault("verbose", False)
        classifier_params.setdefault("task_type", params.get("task_type", "GPU"))
        classifier = cb.CatBoostClassifier(**classifier_params)
        classifier.fit(train_x, (train_y >= threshold).astype(int), verbose=False)
        high = train_y >= threshold
        if not high.any():
            raise ValueError(f"No high-generation training rows for group {group_id}")
        high_val = None if other_y is None else other_y >= threshold
        val_x = None if other_y is None or not high_val.any() else other_x.loc[high_val]
        val_y = None if other_y is None or not high_val.any() else other_y.loc[high_val]
        regressor = fit_regressor(
            train_x.loc[high], train_y.loc[high], val_x, val_y, params, config
        )
        probability = classifier.predict_proba(other_x)[:, 1]
        regression_prediction = regressor.predict(other_x)
        probability_threshold = float(config.get("two_stage", {}).get("probability_threshold", 0.5))
        predictions = np.where(probability >= probability_threshold, regression_prediction, 0.0)
        return regressor, predictions

    weights = None
    if flags["sample_weighting"]:
        low_weight = float(config.get("sample_weighting", {}).get("low_generation_weight", 0.25))
        weights = np.where(train_y.to_numpy() >= threshold, 1.0, low_weight)
    model = fit_regressor(train_x, train_y, other_x if other_y is not None else None, other_y, params, config, weights)
    return model, model.predict(other_x)


def merge_direction_features(
    features: pd.DataFrame,
    direction_table: pd.DataFrame | None,
    group_id: int,
) -> pd.DataFrame:
    if direction_table is None:
        return features
    selected = select_group_direction_features(direction_table, group_id)
    return features.merge(selected, on=TIME_COL, how="left", validate="one_to_one")


def transform_pair(
    train_x: pd.DataFrame,
    other_x: pd.DataFrame,
    group_id: int,
    config: dict,
    flags: dict,
    dropped_features: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = build_feature_pipeline(config, group_id)
    train_processed = pipeline.fit_transform(train_x)
    other_processed = pipeline.transform(other_x)
    if flags["lead_interactions"]:
        train_processed = add_lead_time_interactions(train_processed)
        other_processed = add_lead_time_interactions(other_processed)
    if flags["direction_features"]:
        train_processed = add_direction_interactions(train_processed, group_id)
        other_processed = add_direction_interactions(other_processed, group_id)
    _, train_array, other_array, names = fit_tree_preprocessor(
        train_processed, other_processed, config=config
    )
    train_frame = pd.DataFrame(train_array, columns=names)
    other_frame = pd.DataFrame(other_array, columns=names)
    if config.get("use_feature_drop", False):
        train_frame, other_frame, dropped = apply_feature_drop(
            train_frame, other_frame, dropped_features
        )
        if dropped:
            print(f"Dropped {len(dropped)} features for group {group_id}")
    return train_frame, other_frame


def align_scada(hourly: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    aligned = hourly.set_index(TIME_COL).reindex(pd.DatetimeIndex(times)).reset_index(drop=True)
    if aligned["scada_ws_mean"].isna().all():
        raise ValueError("SCADA alignment produced no usable wind observations")
    return aligned


def add_scada_features(
    train_x: pd.DataFrame,
    other_x: pd.DataFrame,
    train_scada: pd.DataFrame,
    config: dict,
    flags: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_x.copy()
    other_out = other_x.copy()
    scada_config = config.get("scada", {})
    blocks = int(scada_config.get("cross_fit_blocks", 5))
    forecast_column = scada_config.get("forecast_wind_column", "gfs__ws100__mean")

    if flags["scada_offset"]:
        if forecast_column not in train_x:
            raise ValueError(f"SCADA offset forecast column is missing: {forecast_column}")
        smoothing = float(scada_config.get("offset_smoothing", 48.0))
        train_offset = cross_fitted_offsets(
            train_x, train_scada["scada_ws_mean"], forecast_column, smoothing, blocks
        )
        calibrator = OffsetCalibrator(smoothing=smoothing).fit(
            train_x, train_scada["scada_ws_mean"], forecast_column
        )
        other_offset = calibrator.transform(other_x, forecast_column)
        train_out = pd.concat([train_out, train_offset.add_prefix("scada__")], axis=1)
        other_out = pd.concat([other_out, other_offset.add_prefix("scada__")], axis=1)

    if flags["predicted_scada"]:
        auxiliary_params = dict(scada_config.get("auxiliary_model_params", {}))
        train_predictions = cross_fitted_scada_predictions(
            train_x, train_scada, auxiliary_params, blocks
        )
        auxiliary = AuxiliaryScadaModel(auxiliary_params).fit(train_x, train_scada)
        other_predictions = auxiliary.predict(other_x)
        train_out = pd.concat([train_out, train_predictions.add_prefix("scada__")], axis=1)
        other_out = pd.concat([other_out, other_predictions.add_prefix("scada__")], axis=1)

        if flags["power_curve"]:
            predicted_column = "predicted_scada_ws_mean"
            curve_config = config.get("power_curve", {})
            bin_width = float(curve_config.get("bin_width", 0.5))
            smoothing = float(curve_config.get("smoothing", 24.0))
            train_curve = cross_fitted_power_curve(
                train_predictions[predicted_column],
                train_scada["scada_power_kwh"],
                bin_width,
                smoothing,
                blocks,
            )
            curve = EmpiricalPowerCurve(bin_width=bin_width, smoothing=smoothing).fit(
                train_predictions[predicted_column], train_scada["scada_power_kwh"]
            )
            other_curve = curve.predict(other_predictions[predicted_column])
            train_out["scada__power_curve_prediction_kwh"] = train_curve.to_numpy()
            other_out["scada__power_curve_prediction_kwh"] = other_curve.to_numpy()

    return train_out, other_out


def load_targets(config: dict) -> pd.DataFrame:
    path = Path(config["data"]["train_dir"]) / "train_labels.csv"
    targets = pd.read_csv(path, encoding="utf-8-sig")
    targets = targets.rename(columns={"kst_dtm": TIME_COL})
    targets[TIME_COL] = pd.to_datetime(targets[TIME_COL])
    return targets


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.iterations is not None:
        config["model"]["params"]["iterations"] = args.iterations
    flags = experiment_flags(config)
    output_root = Path(args.output_root or config.get("output_root", "experiments/exp04_scada_calibration/outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    dropped_features = load_dropped_features(config) if config.get("use_feature_drop", False) else set()

    artifacts = load_raw_feature_artifacts(config)
    train_features, test_features = artifacts[0], artifacts[1]
    targets = load_targets(config)
    direction_train = build_direction_table("train", config) if flags["direction_features"] else None
    direction_test = build_direction_table("test", config) if flags["direction_features"] else None
    needs_scada = flags["scada_offset"] or flags["predicted_scada"] or flags["power_curve"]
    hourly_scada = load_hourly_scada(config) if needs_scada else {}

    oof_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    importance_frames: list[pd.DataFrame] = []
    best_iterations = {1: [], 2: [], 3: []}
    validation_runs = [
        (spec["fold"], group_id, spec["validation"])
        for spec in FOLD_SPECS for group_id in spec["groups"]
    ]

    for fold_name, group_id, validation in validation_runs:
        target = f"kpx_group_{group_id}"
        print(f"--- {fold_name} | group {group_id} | flags={flags} ---")
        group_features = merge_direction_features(
            get_features_for_group(train_features, group_id).copy(), direction_train, group_id
        )
        group_features[TIME_COL] = pd.to_datetime(group_features[TIME_COL])
        table = group_features.merge(targets[[TIME_COL, target]], on=TIME_COL, how="inner", validate="one_to_one")
        fold_config = copy.deepcopy(config)
        fold_config["validation"] = copy.deepcopy(validation)
        train_mask, val_mask = split_labeled_table(table, target, fold_config)
        train = table.loc[train_mask].reset_index(drop=True)
        val = table.loc[val_mask].reset_index(drop=True)
        if train.empty or val.empty:
            raise ValueError(f"Empty split for {fold_name} / group {group_id}")
        drop_columns = [TIME_COL] + [column for column in table if column.startswith("kpx_group_")]
        train_x, val_x = transform_pair(
            train.drop(columns=drop_columns, errors="ignore"),
            val.drop(columns=drop_columns, errors="ignore"),
            group_id, config, flags, dropped_features,
        )
        if needs_scada:
            train_scada = align_scada(hourly_scada[group_id], train[TIME_COL])
            train_x, val_x = add_scada_features(
                train_x, val_x, train_scada, config, flags
            )
        model, raw_predictions = fit_predict(
            train_x, train[target], val_x, val[target], group_id, config, flags
        )
        predictions = postprocess(raw_predictions, target, config.get("postprocess", {}))
        tree_count = int(model.tree_count_)
        best_iterations[group_id].append(tree_count)
        metrics = calculate_group_metrics(val[target], predictions, target)
        metric_rows.append({
            "fold": fold_name,
            "group_id": group_id,
            "target": target,
            "train_start": str(train[TIME_COL].min()),
            "train_end": str(train[TIME_COL].max()),
            "valid_start": str(val[TIME_COL].min()),
            "valid_end": str(val[TIME_COL].max()),
            "best_iteration": tree_count,
            **metrics,
        })
        capacity = CAPACITY_KWH[target]
        oof_frames.append(pd.DataFrame({
            TIME_COL: val[TIME_COL], "fold": fold_name, "group_id": group_id, "target": target,
            "y_true": val[target], "raw_prediction": raw_predictions, "prediction": predictions,
            "absolute_error": np.abs(predictions - val[target].to_numpy()),
            "error_rate": np.abs(predictions - val[target].to_numpy()) / capacity,
            "valid_for_metric": val[target].to_numpy() >= capacity * 0.10,
        }))
        importance_frames.append(pd.DataFrame({
            "feature": train_x.columns,
            "importance": model.get_feature_importance(),
            "fold": fold_name,
            "group_id": group_id,
        }))
        print(f"score={metrics['total_score']:.5f}, FICR={metrics['ficr']:.5f}, trees={tree_count}")

    oof = pd.concat(oof_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    overall, group_metrics = calculate_oof_metrics(oof)
    oof.to_csv(output_root / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(output_root / "fold_group_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics[["fold", "group_id", "best_iteration"]].to_csv(
        output_root / "best_iterations.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(importance_frames).to_csv(
        output_root / "feature_importances_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    (output_root / "val_results.yaml").write_text(
        yaml.safe_dump({**overall, "group_metrics": group_metrics, "fold_group_metrics": metric_rows}, sort_keys=False),
        encoding="utf-8",
    )
    print(yaml.safe_dump(overall, sort_keys=False))
    if args.no_finalize:
        return

    sample = load_sample_submission(config)
    final_predictions: dict[str, np.ndarray] = {}
    for group_id in (1, 2, 3):
        target = f"kpx_group_{group_id}"
        train_features_group = merge_direction_features(
            get_features_for_group(train_features, group_id).copy(), direction_train, group_id
        )
        test_features_group = merge_direction_features(
            get_features_for_group(test_features, group_id).copy(), direction_test, group_id
        )
        train_features_group[TIME_COL] = pd.to_datetime(train_features_group[TIME_COL])
        train_table = train_features_group.merge(
            targets[[TIME_COL, target]], on=TIME_COL, how="inner", validate="one_to_one"
        )
        train_table = train_table.loc[train_table[target].notna()].reset_index(drop=True)
        drop_columns = [TIME_COL] + [column for column in train_table if column.startswith("kpx_group_")]
        train_x, test_x = transform_pair(
            train_table.drop(columns=drop_columns, errors="ignore"),
            test_features_group.drop(columns=[TIME_COL], errors="ignore"),
            group_id, config, flags, dropped_features,
        )
        if needs_scada:
            train_scada = align_scada(hourly_scada[group_id], train_table[TIME_COL])
            train_x, test_x = add_scada_features(
                train_x, test_x, train_scada, config, flags
            )
        multiplier = float(config.get("final_iteration_multiplier", 1.0))
        iterations = max(1, int(round(np.median(best_iterations[group_id]) * multiplier)))
        model, raw_predictions = fit_predict(
            train_x, train_table[target], test_x, None, group_id, config, flags, iterations
        )
        final_predictions[target] = postprocess(
            raw_predictions, target, config.get("postprocess", {})
        )

    submission_dir = output_root / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    create_submission(sample, final_predictions, path=submission_dir / "submission_exp04.csv")
    print(f"Experiment complete: {output_root}")


if __name__ == "__main__":
    main()
