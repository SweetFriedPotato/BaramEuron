from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baram.constants import TIME_COL
from baram.data import load_sample_submission
from baram.feature_builder import get_features_for_group, load_raw_feature_artifacts
from baram.preprocessing import fit_tree_preprocessor
from baram.submission import create_submission, postprocess
from baram.validation import split_labeled_table
from shared.metrics import calculate_competition_metric

from .feature_blocks import FeatureBlockPipeline


FOLDS = [
    ("Fold A", 1),
    ("Fold A", 2),
    ("Fold B", 1),
    ("Fold B", 2),
    ("Fold B", 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp03: AutoGluon regression")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--presets")
    parser.add_argument("--no-finalize", action="store_true")
    return parser.parse_args()


def load_dropped_features(config: dict) -> set[str]:
    path = Path(
        config.get(
            "feature_drop_list",
            "experiments/exp03_autogluon/configs/dropped_features_list.txt",
        )
    )
    if not path.exists():
        print(f"No feature drop list found at {path}; using all features.")
        return set()
    features = {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    print(f"Loaded {len(features)} features to drop from: {path}")
    return features


def apply_feature_drop(
    train: pd.DataFrame,
    other: pd.DataFrame,
    dropped_features: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    columns = [name for name in train.columns if name in dropped_features and name in other.columns]
    train = train.drop(columns=columns)
    other = other.drop(columns=columns)
    if train.shape[1] == 0:
        raise ValueError("Feature dropping removed every available feature")
    return train, other, columns


def feature_block_config(config: dict) -> dict[str, bool]:
    features = config.get("features", {})
    return {
        "wind_physics": features.get("wind_physics", features.get("power_curve_features", False)),
        "thermodynamic": features.get("thermodynamic", False),
        "forecast_disagreement": features.get(
            "forecast_disagreement", features.get("weather_summary", False)
        ),
        "advanced_meteorology": features.get("advanced_meteorology", True),
    }


def transform_pair(
    train: pd.DataFrame,
    other: pd.DataFrame,
    group_id: int,
    config: dict,
    dropped_features: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = FeatureBlockPipeline(
        blocks=feature_block_config(config),
        group_id=group_id,
        wind_config=config.get("wind_physics", {}),
    )
    train_processed = pipeline.fit_transform(train)
    other_processed = pipeline.transform(other)
    _, train_array, other_array, feature_names = fit_tree_preprocessor(
        train_processed, other_processed, config=config
    )
    train_out = pd.DataFrame(train_array, columns=feature_names)
    other_out = pd.DataFrame(other_array, columns=feature_names)
    train_out, other_out, dropped = apply_feature_drop(
        train_out, other_out, dropped_features
    )
    if dropped:
        print(f"Dropped {len(dropped)} available features for group {group_id}.")
    return train_out, other_out


def predictor_fit_args(model_config: dict) -> dict:
    args = dict(model_config.get("fit_args", {}))
    args["presets"] = model_config.get("presets", "medium_quality")
    time_limit = model_config.get("time_limit")
    if time_limit is not None:
        args["time_limit"] = int(time_limit)
    num_gpus = model_config.get("num_gpus")
    if num_gpus is not None:
        args["num_gpus"] = int(num_gpus)
    return args


def fit_predictor(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    label: str,
    path: Path,
    model_config: dict,
    tuning_x: pd.DataFrame | None = None,
    tuning_y: pd.Series | None = None,
):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as error:
        raise RuntimeError(
            "AutoGluon is not installed. Install experiments/exp03_autogluon/requirements.txt"
        ) from error

    train_data = train_x.copy()
    train_data[label] = train_y.reset_index(drop=True)
    tuning_data = None
    if tuning_x is not None and tuning_y is not None:
        tuning_data = tuning_x.copy()
        tuning_data[label] = tuning_y.reset_index(drop=True)

    predictor = TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric=model_config.get("eval_metric", "mean_absolute_error"),
        path=str(path),
        verbosity=int(model_config.get("verbosity", 2)),
    )
    predictor.fit(
        train_data=train_data,
        tuning_data=tuning_data,
        **predictor_fit_args(model_config),
    )
    return predictor, tuning_data


def feature_importance(
    predictor,
    data: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
    importance = predictor.feature_importance(data=data, silent=True).reset_index()
    importance = importance.rename(columns={importance.columns[0]: "feature"})
    return importance[["feature", "importance"]]


def load_targets(config: dict) -> pd.DataFrame:
    path = Path(config["data"]["train_dir"]) / "train_labels.csv"
    if not path.exists():
        path = Path(config["data"]["root"]) / "train" / "train_labels.csv"
    targets = pd.read_csv(path, encoding="utf-8-sig")
    for candidate in ("kst_dtm", "datetime", "timestamp"):
        if candidate in targets.columns:
            targets = targets.rename(columns={candidate: TIME_COL})
            break
    targets[TIME_COL] = pd.to_datetime(targets[TIME_COL])
    return targets


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_config = dict(config.get("model", {}))
    if args.time_limit is not None:
        model_config["time_limit"] = args.time_limit
    if args.presets is not None:
        model_config["presets"] = args.presets

    output_root = Path(args.output_root or config.get("output_root", "experiments/exp03_autogluon/outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    model_root = output_root / "models"
    dropped_features = load_dropped_features(config)

    print("[1/4] Loading raw feature artifacts and labels...")
    artifacts = load_raw_feature_artifacts(config)
    if isinstance(artifacts, tuple):
        train_features, test_features = artifacts[0], artifacts[1]
    else:
        train_features, test_features = artifacts, None
    targets = load_targets(config)

    all_oof_preds: list[pd.DataFrame] = []
    all_oof_trues: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    print("[2/4] Starting fold validation...")
    for fold_name, group_id in FOLDS:
        print(f"--- Processing {fold_name} | Group {group_id} ---")
        target = f"kpx_group_{group_id}"
        group_features = get_features_for_group(train_features, group_id).copy()
        group_features[TIME_COL] = pd.to_datetime(group_features[TIME_COL])
        group_data = group_features.merge(targets[[TIME_COL, target]], on=TIME_COL, how="inner")

        fold_config = dict(config)
        fold_config["fold"] = fold_name
        train_mask, val_mask = split_labeled_table(group_data, target, fold_config)
        train_df = group_data.loc[train_mask].reset_index(drop=True)
        val_df = group_data.loc[val_mask].reset_index(drop=True)
        drop_columns = [TIME_COL] + [name for name in group_data if name.startswith("kpx_group_")]
        train_x = train_df.drop(columns=drop_columns, errors="ignore")
        val_x = val_df.drop(columns=drop_columns, errors="ignore")
        train_x, val_x = transform_pair(train_x, val_x, group_id, config, dropped_features)

        model_name = f"{fold_name.lower().replace(' ', '_')}_group_{group_id}"
        predictor, tuning_data = fit_predictor(
            train_x,
            train_df[target],
            target,
            model_root / model_name,
            model_config,
            val_x,
            val_df[target],
        )
        predictions = np.asarray(predictor.predict(val_x))
        all_oof_preds.append(pd.DataFrame({target: predictions}))
        all_oof_trues.append(pd.DataFrame({target: val_df[target].to_numpy()}))

        try:
            importance = feature_importance(
                predictor,
                tuning_data,
                int(model_config.get("feature_importance_rows", 5000)),
            )
            importance["fold"] = fold_name
            importance["group_id"] = group_id
            importances.append(importance)
        except Exception as error:
            print(f"Warning: feature importance failed for {model_name}: {error}")

    print("[3/4] Running competition metric evaluation...")
    oof_preds = pd.concat(all_oof_preds, axis=1).fillna(0)
    oof_trues = pd.concat(all_oof_trues, axis=1).fillna(0)
    metrics = calculate_competition_metric(oof_trues, oof_preds)
    print(yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False))
    (output_root / "val_results.txt").write_text(
        yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    if importances:
        importance_report = (
            pd.concat(importances, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        importance_report.to_csv(
            output_root / "feature_importances_report.csv", index=False, encoding="utf-8-sig"
        )

    if args.no_finalize:
        print("Option '--no-finalize' detected. Skipping final training and submission.")
        return
    if test_features is None:
        raise RuntimeError("Test feature artifacts are required for finalization")

    print("[4/4] Final training and inference...")
    sample_submission = load_sample_submission(config)
    final_predictions: dict[str, np.ndarray] = {}
    for group_id in (1, 2, 3):
        target = f"kpx_group_{group_id}"
        print(f"--- Final training | Group {group_id} ---")
        group_train = get_features_for_group(train_features, group_id).copy()
        group_test = get_features_for_group(test_features, group_id).copy()
        group_train[TIME_COL] = pd.to_datetime(group_train[TIME_COL])
        group_test[TIME_COL] = pd.to_datetime(group_test[TIME_COL])
        group_data = group_train.merge(targets[[TIME_COL, target]], on=TIME_COL, how="inner")
        group_data = group_data.loc[group_data[target].notna()].reset_index(drop=True)
        drop_columns = [TIME_COL] + [name for name in group_data if name.startswith("kpx_group_")]
        train_x = group_data.drop(columns=drop_columns, errors="ignore")
        test_x = group_test.drop(columns=[TIME_COL], errors="ignore")
        train_x, test_x = transform_pair(train_x, test_x, group_id, config, dropped_features)

        predictor, _ = fit_predictor(
            train_x,
            group_data[target],
            target,
            model_root / f"final_group_{group_id}",
            model_config,
        )
        raw_predictions = np.asarray(predictor.predict(test_x))
        final_predictions[target] = postprocess(
            raw_predictions, target, config.get("postprocess", {})
        )

    submission_dir = output_root / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / "submission_autogluon.csv"
    create_submission(sample_submission, final_predictions, path=submission_path)
    print(f"Experiment complete. Outputs and submission saved at: {output_root}")


if __name__ == "__main__":
    main()
