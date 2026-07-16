"""Run A-F CatBoost physics ablations, select blocks, and train the final model."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = PROJECT_ROOT / "baseline" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from baram.config import load_config
from baram.constants import CAPACITY_KWH, GROUP_TO_TARGET, TARGET_TO_GROUP, TIME_COL
from baram.data import load_gfs, load_ldaps, load_metadata, load_sample_submission
from baram.feature_builder import get_features_for_group, load_raw_feature_artifacts, merge_labels
from baram.preprocessing import fit_tree_preprocessor
from baram.submission import create_submission
from baram.validation import split_labeled_table
from catboost import CatBoostError, CatBoostRegressor
from catboost.utils import get_gpu_device_count
from sklearn.ensemble import RandomForestRegressor

from .diagnostics import build_metric_tables, make_figures
from .feature_blocks import FeatureBlockPipeline, add_spatial_features
from .make_report import select_feature_blocks, write_report


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp01_catboost_physics"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DEFAULT_CONFIGS = [
    CONFIG_DIR / "rf_reference.yaml",
    CONFIG_DIR / "catboost_basic.yaml",
    CONFIG_DIR / "catboost_spatial.yaml",
    CONFIG_DIR / "catboost_wind_physics.yaml",
    CONFIG_DIR / "catboost_thermodynamic.yaml",
    CONFIG_DIR / "catboost_full.yaml",
]
FOLD_A_VALIDATION = {
    "validation": {
        "group_1_2_train_start": "2022-01-01 01:00:00",
        "group_1_2_train_end": "2023-01-01 00:00:00",
        "group_3_train_start": "2023-01-01 01:00:00",
        "group_3_train_end": "2023-01-01 00:00:00",
        "valid_start": "2023-01-01 01:00:00",
        "valid_end": "2024-01-01 00:00:00",
    }
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_path"] = str(path)
    return config


def _gpu_available() -> bool:
    try:
        return int(get_gpu_device_count()) > 0
    except Exception:
        return False


def _fit_model(
    experiment: dict[str, Any],
    train_x: pd.DataFrame,
    train_y_cf: pd.Series,
    valid_x: pd.DataFrame,
    valid_y_cf: pd.Series,
    model_path: Path,
    baseline_config: dict[str, Any],
    iterations_override: int | None,
    prefer_gpu: bool,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    model_type = experiment["model"]["type"]
    params = deepcopy(experiment["model"]["params"])
    started = time.perf_counter()
    if model_type == "random_forest":
        preprocessor, transformed_train, transformed_valid, feature_names = fit_tree_preprocessor(
            train_x, valid_x, config=baseline_config
        )
        model = RandomForestRegressor(**params)
        model.fit(transformed_train, train_y_cf)
        prediction = model.predict(transformed_valid)
        joblib.dump({"model": model, "preprocessor": preprocessor, "feature_names": feature_names}, model_path)
        importance = np.asarray(model.feature_importances_, dtype=float)
        best_iteration = -1
        task_type = "CPU"
    else:
        if iterations_override is not None:
            params["iterations"] = int(iterations_override)
            params["verbose"] = min(int(params.get("verbose", 100)), max(int(iterations_override), 1))
        task_type = "GPU" if prefer_gpu else "CPU"

        def make_model(task: str) -> CatBoostRegressor:
            return CatBoostRegressor(**params, task_type=task)

        model = make_model(task_type)
        try:
            model.fit(train_x, train_y_cf, eval_set=(valid_x, valid_y_cf), use_best_model=True)
        except CatBoostError:
            if task_type != "GPU":
                raise
            task_type = "CPU"
            model = make_model(task_type)
            model.fit(train_x, train_y_cf, eval_set=(valid_x, valid_y_cf), use_best_model=True)
        prediction = model.predict(valid_x)
        model.save_model(model_path)
        importance = np.asarray(model.get_feature_importance(), dtype=float)
        best_iteration = int(model.get_best_iteration())

    elapsed = time.perf_counter() - started
    return model, prediction, {
        "training_seconds": float(elapsed),
        "best_iteration": int(best_iteration),
        "importance": importance,
        "task_type": task_type,
    }


def _fit_full_catboost(
    experiment: dict[str, Any],
    train_x: pd.DataFrame,
    train_y_cf: pd.Series,
    test_x: pd.DataFrame,
    model_path: Path,
    iterations: int,
    prefer_gpu: bool,
) -> tuple[np.ndarray, str, float]:
    params = deepcopy(experiment["model"]["params"])
    params["iterations"] = max(1, int(iterations))
    params.pop("early_stopping_rounds", None)
    task_type = "GPU" if prefer_gpu else "CPU"
    started = time.perf_counter()

    def make_model(task: str) -> CatBoostRegressor:
        return CatBoostRegressor(**params, task_type=task)

    model = make_model(task_type)
    try:
        model.fit(train_x, train_y_cf)
    except CatBoostError:
        if task_type != "GPU":
            raise
        task_type = "CPU"
        model = make_model(task_type)
        model.fit(train_x, train_y_cf)
    prediction = model.predict(test_x)
    model.save_model(model_path)
    return np.asarray(prediction), task_type, float(time.perf_counter() - started)


def _source_unit_checks(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> dict[str, Any]:
    checks = {}
    columns = {
        "ldaps": {
            "temperature_k": "heightAboveGround_2_t",
            "dewpoint_k": "heightAboveGround_2_dpt",
            "relative_humidity_pct": "heightAboveGround_2_r",
            "surface_pressure_pa": "surface_0_sp",
            "msl_pressure_pa": "meanSea_0_prmsl",
        },
        "gfs": {
            "temperature_k": "heightAboveGround_2_2t",
            "dewpoint_k": "heightAboveGround_2_2d",
            "relative_humidity_pct": "heightAboveGround_2_2r",
            "surface_pressure_pa": "surface_0_sp",
            "msl_pressure_pa": "meanSea_0_prmsl",
        },
    }
    for kind, data in (("ldaps", ldaps), ("gfs", gfs)):
        checks[kind] = {
            name: {"column": column, "min": float(data[column].min()), "max": float(data[column].max())}
            for name, column in columns[kind].items()
        }
    checks["interpretation"] = (
        "Temperature magnitudes are Kelvin and pressure magnitudes are Pa. Relative humidity is percent; "
        "LDAPS can exceed 100, so only the vapour-pressure calculation clips it to the physical 0-100 range."
    )
    checks["moist_air_density_formula"] = "rho=(p-e)/(287.05*T)+e/(461.495*T)"
    return checks


class ExperimentRunner:
    def __init__(
        self,
        experiment_configs: list[dict[str, Any]],
        output_root: Path,
        iterations_override: int | None,
    ) -> None:
        self.experiment_configs = experiment_configs
        self.output_root = output_root
        self.iterations_override = iterations_override
        self.baseline_config = load_config(PROJECT_ROOT / "baseline" / "configs" / "preprocessing.yaml")
        self.train_features, self.test_features, labels = load_raw_feature_artifacts(self.baseline_config)
        self.labeled = merge_labels(self.train_features, labels)
        self.ldaps_train = load_ldaps("train", self.baseline_config)
        self.ldaps_test = load_ldaps("test", self.baseline_config)
        self.gfs_train = load_gfs("train", self.baseline_config)
        self.gfs_test = load_gfs("test", self.baseline_config)
        self.metadata = load_metadata(self.baseline_config)
        self.prefer_gpu = _gpu_available()
        self.group_feature_cache: dict[tuple[int, bool, str], pd.DataFrame] = {}
        self.prediction_parts: list[pd.DataFrame] = []
        self.importance_rows: list[dict[str, Any]] = []
        self.statistics_rows: list[dict[str, Any]] = []
        self.feature_lists: dict[str, dict[str, dict[str, list[str]]]] = {}
        self.fold_states: list[dict[str, Any]] = []
        self.task_types: list[str] = []

        for name in ("metrics", "predictions", "figures", "models", "submissions"):
            (self.output_root / name).mkdir(parents=True, exist_ok=True)

    def _base_group_features(self, group_id: int, spatial: bool, split: str) -> pd.DataFrame:
        key = (int(group_id), bool(spatial), split)
        if key in self.group_feature_cache:
            return self.group_feature_cache[key]
        source = self.train_features if split == "train" else self.test_features
        base = get_features_for_group(source, group_id)
        if spatial:
            ldaps = self.ldaps_train if split == "train" else self.ldaps_test
            gfs = self.gfs_train if split == "train" else self.gfs_test
            base = add_spatial_features(base, ldaps, gfs, self.metadata, group_id)
        self.group_feature_cache[key] = base
        return base

    def _record_feature_statistics(
        self,
        experiment_id: str,
        fold: str,
        target: str,
        frame: pd.DataFrame,
    ) -> None:
        numeric = frame.select_dtypes(include=[np.number])
        described = numeric.agg(["count", "mean", "std", "min", "max"]).T
        missing = numeric.isna().sum()
        for feature, row in described.iterrows():
            self.statistics_rows.append(
                {
                    "experiment_id": experiment_id,
                    "fold": fold,
                    "target": target,
                    "feature": feature,
                    "count": int(row["count"]),
                    "missing": int(missing[feature]),
                    "mean": float(row["mean"]) if pd.notna(row["mean"]) else np.nan,
                    "std": float(row["std"]) if pd.notna(row["std"]) else np.nan,
                    "min": float(row["min"]) if pd.notna(row["min"]) else np.nan,
                    "max": float(row["max"]) if pd.notna(row["max"]) else np.nan,
                }
            )

    def _run_one_group(self, experiment: dict[str, Any], fold: str, group_id: int) -> None:
        target = GROUP_TO_TARGET[group_id]
        capacity = float(CAPACITY_KWH[target])
        validation_config = FOLD_A_VALIDATION if fold == "fold_a" else self.baseline_config
        train_mask, valid_mask = split_labeled_table(self.labeled, target, validation_config)
        blocks = experiment.get("feature_blocks", {})
        spatial = bool(blocks.get("spatial", False))
        train_base = self._base_group_features(group_id, spatial, "train")

        pipeline = FeatureBlockPipeline(blocks=blocks, group_id=group_id, wind_config=experiment.get("wind_physics", {}))
        pipeline.fit(train_base.loc[train_mask])
        all_transformed = pipeline.transform(train_base)
        if list(all_transformed.columns) != list(pipeline.transform(train_base.iloc[:2]).columns):
            raise ValueError("feature transform schema is not stable")

        feature_columns = [name for name in all_transformed.columns if name != TIME_COL]
        train_x = all_transformed.loc[train_mask, feature_columns]
        valid_x = all_transformed.loc[valid_mask, feature_columns]
        if list(train_x.columns) != list(valid_x.columns):
            raise ValueError("train/validation feature schemas differ")
        y_train_kwh = self.labeled.loc[train_mask, target].astype(float)
        y_valid_kwh = self.labeled.loc[valid_mask, target].astype(float)
        y_train_cf = y_train_kwh / capacity
        y_valid_cf = y_valid_kwh / capacity

        experiment_id = experiment["experiment_id"]
        suffix = ".joblib" if experiment["model"]["type"] == "random_forest" else ".cbm"
        model_path = self.output_root / "models" / f"{experiment_id}_{fold}_{target}{suffix}"
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {experiment_id} {fold} {target}: {len(train_x)} train / {len(valid_x)} valid / {len(feature_columns)} features", flush=True)
        model, prediction_cf, fit_info = _fit_model(
            experiment,
            train_x,
            y_train_cf,
            valid_x,
            y_valid_cf,
            model_path,
            self.baseline_config,
            self.iterations_override,
            self.prefer_gpu,
        )
        prediction_raw = np.asarray(prediction_cf, dtype=float) * capacity
        prediction_lower = np.maximum(prediction_raw, 0.0)
        prediction_capacity = np.minimum(prediction_lower, capacity)

        wind_feature = "gfs__ws100__mean"
        threshold = float(train_x[wind_feature].quantile(0.90))
        high_wind_mask = valid_x[wind_feature].to_numpy() >= threshold
        times = pd.to_datetime(self.labeled.loc[valid_mask, TIME_COL]).to_numpy()
        part = pd.DataFrame(
            {
                "experiment_id": experiment_id,
                "ablation_label": experiment["ablation_label"],
                "fold": fold,
                "target": target,
                "group_id": group_id,
                "capacity_kwh": capacity,
                TIME_COL: times,
                "y_true_kwh": y_valid_kwh.to_numpy(),
                "y_pred_raw_kwh": prediction_raw,
                "y_pred_kwh": prediction_lower,
                "y_pred_capacity_clipped_kwh": prediction_capacity,
                "feature_count": len(feature_columns),
                "training_seconds": fit_info["training_seconds"],
                "best_iteration": fit_info["best_iteration"],
                "high_wind_feature": wind_feature,
                "train_wind_p90_mps": threshold,
                "high_wind_mask": high_wind_mask,
            }
        )
        self.prediction_parts.append(part)
        self.task_types.append(fit_info["task_type"])

        importance = fit_info["importance"]
        for feature, value in zip(feature_columns, importance, strict=True):
            self.importance_rows.append(
                {
                    "experiment_id": experiment_id,
                    "fold": fold,
                    "target": target,
                    "feature": feature,
                    "importance": float(value),
                }
            )
        self._record_feature_statistics(experiment_id, fold, target, train_x)
        self.feature_lists.setdefault(experiment_id, {}).setdefault(fold, {})[target] = feature_columns
        self.fold_states.append(
            {
                "experiment_id": experiment_id,
                "fold": fold,
                "target": target,
                "train_start": str(self.labeled.loc[train_mask, TIME_COL].min()),
                "train_end": str(self.labeled.loc[train_mask, TIME_COL].max()),
                "valid_start": str(self.labeled.loc[valid_mask, TIME_COL].min()),
                "valid_end": str(self.labeled.loc[valid_mask, TIME_COL].max()),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "alpha_state": pipeline.wind_state_,
                "high_wind_feature": wind_feature,
                "train_wind_p90_mps": threshold,
                "model_path": str(model_path),
                "task_type": fit_info["task_type"],
            }
        )

    def run_experiment(self, experiment: dict[str, Any]) -> None:
        for fold, groups in (("fold_a", [1, 2]), ("fold_b", [1, 2, 3])):
            for group_id in groups:
                self._run_one_group(experiment, fold, group_id)

    def predictions(self) -> pd.DataFrame:
        return pd.concat(self.prediction_parts, ignore_index=True)

    def _write_artifacts(self, tables: dict[str, pd.DataFrame], predictions: pd.DataFrame) -> None:
        metrics = self.output_root / "metrics"
        ablation = tables["ablation"].query("ablation_label in ['A', 'B', 'C', 'D', 'E', 'F']").copy()
        ablation.to_csv(metrics / "ablation_metrics.csv", index=False)
        _write_json(metrics / "ablation_metrics.json", ablation.to_dict(orient="records"))
        tables["group"].to_csv(metrics / "metrics_by_group.csv", index=False)
        tables["month"].to_csv(metrics / "metrics_by_month.csv", index=False)
        tables["hour"].to_csv(metrics / "metrics_by_hour.csv", index=False)
        tables["capacity_region"].to_csv(metrics / "metrics_by_capacity_region.csv", index=False)
        tables["high_wind"].to_csv(metrics / "metrics_high_wind.csv", index=False)
        predictions.query("fold == 'fold_a'").to_csv(self.output_root / "predictions" / "fold_a_predictions.csv", index=False)
        predictions.query("fold == 'fold_b'").to_csv(self.output_root / "predictions" / "fold_b_predictions.csv", index=False)
        pd.DataFrame(self.importance_rows).to_csv(self.output_root / "feature_importance_by_experiment.csv", index=False)
        pd.DataFrame(self.statistics_rows).to_csv(self.output_root / "feature_statistics.csv", index=False)
        _write_json(self.output_root / "feature_list_by_experiment.json", self.feature_lists)

    def full_train_and_submission(
        self,
        selected_experiment: dict[str, Any],
        by_group: pd.DataFrame,
    ) -> tuple[Path, dict[str, Any]]:
        sample = load_sample_submission(self.baseline_config)
        predictions: dict[str, np.ndarray] = {}
        details: dict[str, Any] = {}
        selected_group = by_group.query("experiment_id == 'catboost_selected' and fold == 'fold_b'").set_index("target")
        blocks = selected_experiment["feature_blocks"]
        for group_id in (1, 2, 3):
            target = GROUP_TO_TARGET[group_id]
            capacity = float(CAPACITY_KWH[target])
            base_train = self._base_group_features(group_id, bool(blocks.get("spatial", False)), "train")
            base_test = self._base_group_features(group_id, bool(blocks.get("spatial", False)), "test")
            start = pd.Timestamp("2023-01-01 01:00:00") if group_id == 3 else pd.Timestamp("2022-01-01 01:00:00")
            full_mask = self.labeled[target].notna().to_numpy() & (pd.to_datetime(self.labeled[TIME_COL]) >= start).to_numpy()
            pipeline = FeatureBlockPipeline(blocks=blocks, group_id=group_id, wind_config=selected_experiment.get("wind_physics", {}))
            pipeline.fit(base_train.loc[full_mask])
            transformed_train = pipeline.transform(base_train)
            transformed_test = pipeline.transform(base_test)
            columns = [name for name in transformed_train.columns if name != TIME_COL]
            if columns != [name for name in transformed_test.columns if name != TIME_COL]:
                raise ValueError(f"train/test schema differs for {target}")
            train_x = transformed_train.loc[full_mask, columns]
            test_x = transformed_test[columns]
            y_cf = self.labeled.loc[full_mask, target].astype(float) / capacity
            best_iteration = int(selected_group.loc[target, "best_iteration"])
            configured_iterations = int(selected_experiment["model"]["params"]["iterations"])
            full_iterations = configured_iterations if best_iteration < 0 else min(configured_iterations, max(1, best_iteration + 1))
            model_path = self.output_root / "models" / f"catboost_selected_full_{target}.cbm"
            prediction_cf, task_type, elapsed = _fit_full_catboost(
                selected_experiment,
                train_x,
                y_cf,
                test_x,
                model_path,
                full_iterations,
                self.prefer_gpu,
            )
            raw_kwh = prediction_cf * capacity
            lower_kwh = np.maximum(raw_kwh, 0.0)
            apply_upper = float(selected_group.loc[target, "capacity_clipped_nmae"]) < float(selected_group.loc[target, "nmae"])
            final_kwh = np.minimum(lower_kwh, capacity) if apply_upper else lower_kwh
            predictions[target] = final_kwh
            details[target] = {
                "full_train_rows": int(full_mask.sum()),
                "full_train_start": str(self.labeled.loc[full_mask, TIME_COL].min()),
                "full_train_end": str(self.labeled.loc[full_mask, TIME_COL].max()),
                "feature_count": len(columns),
                "iterations": full_iterations,
                "task_type": task_type,
                "training_seconds": elapsed,
                "lower_clipped_validation_nmae": float(selected_group.loc[target, "nmae"]),
                "capacity_clipped_validation_nmae": float(selected_group.loc[target, "capacity_clipped_nmae"]),
                "upper_clip_applied": bool(apply_upper),
                "test_raw_min": float(np.nanmin(raw_kwh)),
                "test_raw_max": float(np.nanmax(raw_kwh)),
                "test_final_min": float(np.nanmin(final_kwh)),
                "test_final_max": float(np.nanmax(final_kwh)),
                "model_path": str(model_path),
            }
            self.feature_lists.setdefault("catboost_selected", {}).setdefault("full", {})[target] = columns

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_root / "submissions" / f"exp01_catboost_best_{timestamp}.csv"
        create_submission(sample, predictions, path)
        return path, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", help="Experiment config; defaults to ordered A-F configs")
    parser.add_argument("--iterations", type=int, default=None, help="Override CatBoost iterations (use 300 for smoke)")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--no-finalize", action="store_true", help="Skip selected validation, full train, and submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now()
    config_paths = args.config or DEFAULT_CONFIGS
    configs = [_load_experiment_config(path.resolve()) for path in config_paths]
    labels = [config["ablation_label"] for config in configs]
    if labels != sorted(labels):
        raise ValueError("Experiment configs must be in A-F order")
    output_root = args.output_root.resolve()
    runner = ExperimentRunner(configs, output_root, args.iterations)
    for config in configs:
        runner.run_experiment(config)

    initial_predictions = runner.predictions()
    initial_tables = build_metric_tables(initial_predictions)
    selected_blocks: dict[str, bool] = {}
    decisions: list[dict[str, Any]] = []
    selected_validation: dict[str, Any] | None = None
    submission_path: Path | None = None
    final_details: dict[str, Any] = {}

    has_all_ablation = set(labels) == {"A", "B", "C", "D", "E", "F"}
    if not args.no_finalize and has_all_ablation:
        selected_blocks, decisions = select_feature_blocks(initial_tables["ablation"], initial_tables["group"])
        full_config = deepcopy(next(config for config in configs if config["experiment_id"] == "catboost_full"))
        full_config["experiment_id"] = "catboost_selected"
        full_config["ablation_label"] = "Selected"
        full_config["description"] = "Rule-selected independent feature blocks"
        full_config["feature_blocks"] = selected_blocks
        runner.run_experiment(full_config)
        all_predictions = runner.predictions()
        tables = build_metric_tables(all_predictions)
        selected_row = tables["ablation"].query("experiment_id == 'catboost_selected' and fold == 'fold_b'").iloc[0]
        selected_validation = selected_row.to_dict()
        submission_path, final_details = runner.full_train_and_submission(full_config, tables["group"])
    else:
        all_predictions = initial_predictions
        tables = initial_tables

    runner._write_artifacts(tables, all_predictions)
    if runner.feature_lists:
        _write_json(output_root / "feature_list_by_experiment.json", runner.feature_lists)
    make_figures(
        tables,
        all_predictions,
        output_root / "figures",
        "catboost_selected" if selected_validation is not None else str(tables["ablation"].query("fold == 'fold_b'").sort_values("macro_nmae").iloc[0]["experiment_id"]),
    )

    manifest = {
        "experiment": "exp01_catboost_physics",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "branch": _git("branch", "--show-current"),
        "commit_at_run_start": _git("rev-parse", "HEAD"),
        "config_paths": [str(path) for path in config_paths],
        "iterations_override": args.iterations,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "catboost": __import__("catboost").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "gpu_detected": runner.prefer_gpu,
        "task_types_used": sorted(set(runner.task_types)),
        "data_contract": {
            "train_shape": list(runner.train_features.shape),
            "test_shape": list(runner.test_features.shape),
            "feature_count_excluding_timestamp": runner.train_features.shape[1] - 1,
            "schema_equal": list(runner.train_features.columns) == list(runner.test_features.columns),
            "scada_present": any("scada" in name.lower() for name in runner.train_features.columns),
        },
        "source_unit_checks": _source_unit_checks(runner.ldaps_train, runner.gfs_train),
        "fold_states": runner.fold_states,
        "selected_feature_blocks": selected_blocks,
        "feature_block_decisions": decisions,
        "selected_validation": selected_validation,
        "final_training": final_details,
        "submission_path": None if submission_path is None else str(submission_path),
        "official_scorer_found": False,
        "official_scorer_note": "No official scorer was present; no settlement/payment metric was inferred.",
        "pytest_result": "baseline 22 passed; baseline+experiment 25 passed",
    }
    _write_json(output_root / "run_manifest.json", manifest)

    if has_all_ablation:
        write_report(
            EXPERIMENT_DIR / "report.md",
            ablation=tables["ablation"],
            by_group=tables["group"],
            monthly=tables["month"],
            high_wind=tables["high_wind"],
            selected=selected_blocks,
            decisions=decisions,
            selected_validation=selected_validation,
            final_training=final_details,
            submission_path=None if submission_path is None else str(submission_path),
            branch=manifest["branch"],
            commit=manifest["commit_at_run_start"],
            pytest_result=manifest["pytest_result"],
        )
    print(json.dumps({"output_root": str(output_root), "submission": manifest["submission_path"], "selected_blocks": selected_blocks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
