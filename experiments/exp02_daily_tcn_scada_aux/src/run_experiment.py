"""Run pointwise MLP, daily TCN, SCADA auxiliary, blending, and full training."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = PROJECT_ROOT / "baseline" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from baram.data import load_sample_submission
from baram.submission import create_submission, validate_submission_contract

from .blend import search_blend_weights, select_blend_weight
from .data_contract import (
    FOLD_WINDOWS,
    SelectedFeatureUnionBuilder,
    baseline_config,
    fold_time_mask,
    issue_mapping,
    raw_artifacts,
    write_issue_contract,
)
from .evaluate import metric_tables, prediction_diagnostics
from .make_report import make_figures, write_report
from .models import build_model
from .preprocessing import NeuralFoldPreprocessor
from .scada_targets import AuxiliaryTargetScaler, build_scada_aux_targets, write_scada_checks
from .sequence_builder import SequenceBundle, build_sequences, flatten_predictions, fold_bundle
from .trainer import TrainingResult, predict, seed_everything, train_fixed_epochs, train_model


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp02_daily_tcn_scada_aux"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DEFAULT_CONFIGS = [
    CONFIG_DIR / "mlp_pointwise.yaml",
    CONFIG_DIR / "tcn_plain.yaml",
    CONFIG_DIR / "tcn_aux_005.yaml",
    CONFIG_DIR / "tcn_aux_015.yaml",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.ndarray,)): return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    value["_config_path"] = str(path)
    return value


@dataclass
class PreparedFold:
    fold: str
    train: SequenceBundle
    valid: SequenceBundle
    train_x: np.ndarray
    valid_x: np.ndarray
    train_aux: np.ndarray
    train_aux_mask: np.ndarray
    valid_aux: np.ndarray
    valid_aux_mask: np.ndarray
    preprocessor: NeuralFoldPreprocessor
    aux_scaler: AuxiliaryTargetScaler
    high_wind_threshold: float
    feature_builder: SelectedFeatureUnionBuilder


def prepare_fold(
    fold: str,
    config: dict,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    labels: pd.DataFrame,
    scada: pd.DataFrame,
    checks_dir: Path,
    preprocessing_states: dict,
) -> PreparedFold:
    fit_mask = fold_time_mask(train_features[TIME_COL], fold, "train")
    builder = SelectedFeatureUnionBuilder(config)
    selected_train = builder.fit_transform(train_features, fit_mask)
    if fold == "fold_b":
        selected_test = builder.transform("test", test_features)
        manifest = builder.manifest(selected_train, selected_test)
        write_json(checks_dir / "tcn_feature_manifest.json", manifest)
    bundle, incomplete = build_sequences(
        selected_train, issue_mapping(config, "train"), labels=labels, aux_targets=scada
    )
    if not incomplete.empty:
        raise ValueError(f"incomplete training sequences after validated contract: {len(incomplete)}")
    train = fold_bundle(bundle, fold, "train")
    valid = fold_bundle(bundle, fold, "valid")
    preprocessor = NeuralFoldPreprocessor()
    train_x = preprocessor.fit_transform(train.x)
    valid_x = preprocessor.transform(valid.x)
    aux_scaler = AuxiliaryTargetScaler().fit(train.aux_wind, train.aux_mask)
    train_aux, train_aux_mask = aux_scaler.transform(train.aux_wind, train.aux_mask)
    valid_aux, valid_aux_mask = aux_scaler.transform(valid.aux_wind, valid.aux_mask)
    wind_index = train.feature_names.index("gfs__ws100__mean")
    threshold = float(np.nanquantile(train.x[:, :, wind_index], 0.90))
    preprocessing_states[fold] = {
        "neural": preprocessor.state(train.feature_names),
        "auxiliary": aux_scaler.state(),
        "feature_alpha_states": builder.wind_states_,
        "train_issue_blocks": len(train.x),
        "valid_issue_blocks": len(valid.x),
        "train_first_timestamp": str(train.timestamps.min()),
        "train_last_timestamp": str(train.timestamps.max()),
        "valid_first_timestamp": str(valid.timestamps.min()),
        "valid_last_timestamp": str(valid.timestamps.max()),
        "high_wind_feature": "gfs__ws100__mean",
        "high_wind_p90_mps": threshold,
    }
    return PreparedFold(
        fold, train, valid, train_x, valid_x, train_aux, train_aux_mask,
        valid_aux, valid_aux_mask, preprocessor, aux_scaler, threshold, builder
    )


def prediction_frame(prepared: PreparedFold, prediction_cf: np.ndarray, experiment_id: str, seed: int) -> pd.DataFrame:
    clipped = np.maximum(prediction_cf, 0.0)
    frame = flatten_predictions(prepared.valid, clipped)
    frame["experiment_id"] = experiment_id
    frame["seed"] = int(seed)
    frame["ensemble"] = False
    frame["fold"] = prepared.fold
    wind_index = prepared.valid.feature_names.index("gfs__ws100__mean")
    wind = pd.DataFrame(
        {
            TIME_COL: prepared.valid.timestamps.reshape(-1),
            "validation_wind_mps": prepared.valid.x[:, :, wind_index].reshape(-1),
        }
    )
    frame = frame.merge(wind, on=TIME_COL, how="left", validate="many_to_one")
    frame["train_wind_p90_mps"] = prepared.high_wind_threshold
    frame["high_wind_mask"] = frame["validation_wind_mps"] >= prepared.high_wind_threshold
    return frame


def seed_ensemble(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment_id", "fold", TIME_COL, "target", "group_id"]
    out = (
        seed_predictions.groupby(keys, sort=False)
        .agg(
            y_true_kwh=("y_true_kwh", "first"),
            y_pred_kwh=("y_pred_kwh", "mean"),
            validation_wind_mps=("validation_wind_mps", "first"),
            train_wind_p90_mps=("train_wind_p90_mps", "first"),
            high_wind_mask=("high_wind_mask", "first"),
        )
        .reset_index()
    )
    out["seed"] = -1
    out["ensemble"] = True
    return out


def load_catboost_reference(path: Path | None) -> pd.DataFrame | None:
    paths: list[Path] = []
    if path is not None and path.exists():
        paths = [path]
    else:
        base = PROJECT_ROOT / "experiments" / "exp01_catboost_physics" / "outputs" / "predictions"
        paths = [base / "fold_a_predictions.csv", base / "fold_b_predictions.csv"]
        paths = [item for item in paths if item.exists()]
    if not paths:
        return None
    data = pd.concat([pd.read_csv(item) for item in paths], ignore_index=True)
    if "experiment_id" in data and "catboost_selected" in set(data["experiment_id"]):
        data = data[data["experiment_id"] == "catboost_selected"].copy()
    required = {TIME_COL, "fold", "target", "group_id", "y_true_kwh", "y_pred_kwh"}
    if required - set(data):
        raise ValueError(f"CatBoost reference missing columns: {sorted(required-set(data))}")
    data = data[[*required]].copy()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data["experiment_id"] = "catboost_reference"
    data["seed"] = -1
    data["ensemble"] = True
    return data.sort_values(["fold", "target", TIME_COL]).reset_index(drop=True)


def attach_high_wind(reference: pd.DataFrame, neural: pd.DataFrame) -> pd.DataFrame:
    wind = neural[["fold", TIME_COL, "target", "validation_wind_mps", "train_wind_p90_mps", "high_wind_mask"]].drop_duplicates()
    return reference.merge(wind, on=["fold", TIME_COL, "target"], how="left", validate="one_to_one")


def choose_tcn_config(ensemble_predictions: pd.DataFrame, seed_predictions: pd.DataFrame) -> tuple[str, bool, dict]:
    ensemble_tables = metric_tables(ensemble_predictions)
    group = ensemble_tables["group"]
    macro = ensemble_tables["macro"]
    fold_b = macro[macro.fold.eq("fold_b")].set_index("experiment_id")
    plain_value = float(fold_b.loc["tcn_plain", "macro_nmae"])
    plain_groups = group[(group.fold == "fold_b") & (group.experiment_id == "tcn_plain")].set_index("group_id").nmae
    seed_macro = metric_tables(seed_predictions)["macro"]
    plain_seed_mean = float(seed_macro[(seed_macro.fold == "fold_b") & (seed_macro.experiment_id == "tcn_plain")].macro_nmae.mean())
    plain_by_seed = seed_macro[(seed_macro.fold == "fold_b") & (seed_macro.experiment_id == "tcn_plain")].set_index("seed").macro_nmae
    eligible = ["tcn_plain"]
    decisions = {}
    for candidate in ("tcn_aux_005", "tcn_aux_015"):
        value = float(fold_b.loc[candidate, "macro_nmae"])
        candidate_groups = group[(group.fold == "fold_b") & (group.experiment_id == candidate)].set_index("group_id").nmae
        maintained = int(((candidate_groups - plain_groups) <= 0).sum())
        candidate_seed_mean = float(seed_macro[(seed_macro.fold == "fold_b") & (seed_macro.experiment_id == candidate)].macro_nmae.mean())
        candidate_by_seed = seed_macro[(seed_macro.fold == "fold_b") & (seed_macro.experiment_id == candidate)].set_index("seed").macro_nmae
        improved_seeds = int(((candidate_by_seed - plain_by_seed) < 0).sum())
        keep = value < plain_value and maintained >= 2 and candidate_seed_mean < plain_seed_mean and improved_seeds >= 2
        if keep: eligible.append(candidate)
        decisions[candidate] = {
            "ensemble_macro_delta": value - plain_value,
            "maintained_or_better_groups": maintained,
            "mean_seed_delta": candidate_seed_mean - plain_seed_mean,
            "improved_seed_count": improved_seeds,
            "keep": keep,
        }
    best = min(eligible, key=lambda item: float(fold_b.loc[item, "macro_nmae"]))
    return best, best != "tcn_plain", decisions


def sync_seed(
    output_root: Path,
    drive_run: Path | None,
    experiment_id: str,
    seed: int,
    predictions: pd.DataFrame | None = None,
    metrics: list[dict] | None = None,
) -> None:
    if drive_run is None:
        return
    destination = drive_run / "seeds" / experiment_id / str(seed)
    destination.mkdir(parents=True, exist_ok=True)
    for source in (output_root / "checkpoints", output_root / "checks" / "preprocessors"):
        if not source.exists(): continue
        for path in source.rglob(f"*seed_{seed}*"):
            if path.is_file(): shutil.copy2(path, destination / path.name)
    for config_path in CONFIG_DIR.glob("*.yaml"):
        shutil.copy2(config_path, destination / config_path.name)
    for name in ("tcn_feature_manifest.json", "preprocessing_statistics.json", "scada_aux_alignment.json"):
        source = output_root / "checks" / name
        if source.exists(): shutil.copy2(source, destination / name)
    if predictions is not None:
        predictions.to_csv(destination / "predictions.csv", index=False)
    if metrics is not None:
        write_json(destination / "metrics.json", metrics)


def run_validation(
    configs: list[dict],
    prepared: dict[str, PreparedFold],
    output_root: Path,
    smoke: bool,
    drive_run: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    prediction_parts = []; training_rows = []; histories = []
    for model_config in configs:
        experiment_id = model_config["experiment_id"]
        seeds = [42] if smoke else list(model_config["training"]["seeds"])
        for seed in seeds:
            for fold in ("fold_a", "fold_b"):
                data = prepared[fold]
                preprocessor_path = output_root / "checks" / "preprocessors" / f"{experiment_id}_{fold}_seed_{seed}.joblib"
                data.preprocessor.save(preprocessor_path)
                joblib.dump(
                    data.aux_scaler,
                    output_root / "checks" / "preprocessors" / f"{experiment_id}_{fold}_seed_{seed}_aux.joblib",
                )
                checkpoint = output_root / "checkpoints" / f"{experiment_id}_{fold}_seed_{seed}.pt"
                seed_everything(int(seed))
                model = build_model(model_config, data.train_x.shape[-1])
                epochs = int(model_config["training"].get("smoke_epochs", 3)) if smoke else None
                print(f"\n[{experiment_id}] {fold} seed={seed} device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
                result = train_model(
                    model, data.train_x, data.train.y_cf, data.train.label_mask,
                    data.valid_x, data.valid.y_cf, data.valid.label_mask,
                    model_config, seed, checkpoint,
                    data.train_aux, data.train_aux_mask, data.valid_aux, data.valid_aux_mask,
                    max_epochs_override=epochs,
                )
                prediction_parts.append(prediction_frame(data, result.prediction_cf, experiment_id, seed))
                training_rows.append(
                    {
                        "experiment_id": experiment_id, "fold": fold, "seed": seed,
                        "best_epoch": result.best_epoch, "macro_nmae": result.best_macro_nmae,
                        "group_1_nmae": result.group_nmae[0], "group_2_nmae": result.group_nmae[1],
                        "group_3_nmae": result.group_nmae[2], "training_seconds": result.training_seconds,
                        "device": result.device, "checkpoint": str(result.checkpoint_path),
                    }
                )
                histories.extend({"experiment_id": experiment_id, "fold": fold, "seed": seed, **row} for row in result.history)
            seed_frames = [
                frame for frame in prediction_parts
                if frame["experiment_id"].iloc[0] == experiment_id and int(frame["seed"].iloc[0]) == int(seed)
            ]
            seed_metrics = [
                row for row in training_rows
                if row["experiment_id"] == experiment_id and int(row["seed"]) == int(seed)
            ]
            sync_seed(
                output_root, drive_run, experiment_id, seed,
                predictions=pd.concat(seed_frames, ignore_index=True), metrics=seed_metrics,
            )
    seed_predictions = pd.concat(prediction_parts, ignore_index=True)
    return seed_predictions, seed_ensemble(seed_predictions), training_rows, pd.DataFrame(histories)


def build_full_data(
    config: dict,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    labels: pd.DataFrame,
    scada: pd.DataFrame,
    output_root: Path,
) -> tuple[SequenceBundle, SequenceBundle, np.ndarray, np.ndarray, np.ndarray, np.ndarray, NeuralFoldPreprocessor, AuxiliaryTargetScaler]:
    fit_mask = fold_time_mask(train_features[TIME_COL], "full", "train")
    builder = SelectedFeatureUnionBuilder(config)
    selected_train = builder.fit_transform(train_features, fit_mask)
    selected_test = builder.transform("test", test_features)
    train_bundle, incomplete_train = build_sequences(selected_train, issue_mapping(config, "train"), labels, scada)
    test_bundle, incomplete_test = build_sequences(selected_test, issue_mapping(config, "test"), labels=None, aux_targets=None)
    if len(incomplete_train) or len(incomplete_test): raise ValueError("incomplete full/test issue blocks")
    preprocessor = NeuralFoldPreprocessor(); train_x = preprocessor.fit_transform(train_bundle.x); test_x = preprocessor.transform(test_bundle.x)
    aux_scaler = AuxiliaryTargetScaler().fit(train_bundle.aux_wind, train_bundle.aux_mask)
    aux, aux_mask = aux_scaler.transform(train_bundle.aux_wind, train_bundle.aux_mask)
    preprocessor.save(output_root / "checks" / "preprocessors" / "full.joblib")
    joblib.dump(aux_scaler, output_root / "checks" / "preprocessors" / "full_aux.joblib")
    return train_bundle, test_bundle, train_x, test_x, aux, aux_mask, preprocessor, aux_scaler


def full_train_submission(
    chosen_config: dict,
    chosen_id: str,
    best_epochs: list[int],
    blend_weight: float,
    catboost_test_path: Path,
    baseline_cfg: dict,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    labels: pd.DataFrame,
    scada: pd.DataFrame,
    output_root: Path,
    drive_run: Path | None,
) -> tuple[Path, dict]:
    train_bundle, test_bundle, train_x, test_x, aux, aux_mask, preprocessor, aux_scaler = build_full_data(
        baseline_cfg, train_features, test_features, labels, scada, output_root
    )
    full_epochs = int(np.median(best_epochs))
    seed_predictions = []
    devices = []
    for seed in chosen_config["training"]["seeds"]:
        seed_everything(int(seed))
        model = build_model(chosen_config, train_x.shape[-1])
        checkpoint = output_root / "checkpoints" / f"{chosen_id}_full_seed_{seed}.pt"
        model, history, device = train_fixed_epochs(
            model, train_x, train_bundle.y_cf, train_bundle.label_mask, chosen_config,
            int(seed), full_epochs, checkpoint, aux, aux_mask
        )
        write_json(checkpoint.with_suffix(".history.json"), history)
        seed_prediction = predict(model, test_x, int(chosen_config["training"]["batch_size"]), torch.device(device))
        seed_predictions.append(seed_prediction)
        seed_path = output_root / "predictions" / f"{chosen_id}_full_seed_{seed}.npz"
        np.savez_compressed(seed_path, prediction_cf=seed_prediction, timestamps=test_bundle.timestamps)
        devices.append(device)
        sync_seed(output_root, drive_run, chosen_id, int(seed))
        if drive_run is not None:
            destination = drive_run / "seeds" / chosen_id / str(seed)
            shutil.copy2(seed_path, destination / seed_path.name)
            shutil.copy2(output_root / "checks" / "preprocessors" / "full.joblib", destination / "full_preprocessor.joblib")
            shutil.copy2(output_root / "checks" / "preprocessors" / "full_aux.joblib", destination / "full_aux_scaler.joblib")
    prediction_cf = np.mean(seed_predictions, axis=0)
    timestamps = test_bundle.timestamps.reshape(-1)
    capacities = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=float)
    tcn_kwh = np.maximum(prediction_cf.reshape(-1, 3) * capacities, 0.0)
    pd.DataFrame(tcn_kwh, columns=TARGETS).assign(**{TIME_COL: timestamps}).to_csv(
        output_root / "predictions" / "tcn_full_ensemble_test.csv", index=False
    )

    catboost = pd.read_csv(catboost_test_path)
    sample = load_sample_submission(baseline_cfg)
    validate_submission_contract(catboost, sample)
    catboost[TIME_COL] = pd.to_datetime(catboost[TIME_COL])
    catboost = catboost.set_index(TIME_COL).loc[pd.DatetimeIndex(timestamps)]
    catboost_kwh = catboost[TARGETS].to_numpy(dtype=float)
    final = np.maximum((1.0 - blend_weight) * catboost_kwh + blend_weight * tcn_kwh, 0.0)
    prediction_map = pd.DataFrame(final, index=pd.DatetimeIndex(timestamps), columns=TARGETS)
    ordered = prediction_map.loc[pd.DatetimeIndex(sample[TIME_COL])]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = output_root / "submissions" / f"exp02_best_blend_{stamp}.csv"
    create_submission(sample, {target: ordered[target].to_numpy() for target in TARGETS}, submission_path)
    return submission_path, {
        "config": chosen_id, "full_epochs": full_epochs, "seeds": chosen_config["training"]["seeds"],
        "devices": devices, "test_issue_blocks": len(test_bundle.x), "test_rows": len(timestamps),
        "feature_count": train_x.shape[-1], "blend_weight": blend_weight,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--catboost-reference", type=Path)
    parser.add_argument("--catboost-test", type=Path)
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--skip-full-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); started = datetime.now(); run_id = started.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve()
    for folder in ("checks", "metrics", "predictions", "checkpoints", "figures", "submissions"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    drive_run = None if args.drive_root is None else args.drive_root.resolve() / run_id
    if drive_run is not None: drive_run.mkdir(parents=True, exist_ok=True)
    config_paths = [args.config_dir.resolve() / path.name for path in DEFAULT_CONFIGS]
    configs = [load_yaml(path) for path in config_paths]
    if args.smoke: configs = [config for config in configs if config["experiment_id"] in {"mlp_pointwise", "tcn_plain"}]
    baseline_cfg = baseline_config(); train_features, test_features, labels = raw_artifacts(baseline_cfg)
    issue_contract = write_issue_contract(output_root / "checks", baseline_cfg)
    if not issue_contract["non_causal_tcn_allowed"]: raise RuntimeError("24-hour issue contract failed; non-causal TCN disabled")
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"])); write_scada_checks(scada, output_root / "checks")
    preprocessing_states = {}
    prepared = {
        fold: prepare_fold(fold, baseline_cfg, train_features, test_features, labels, scada,
                           output_root / "checks", preprocessing_states)
        for fold in ("fold_a", "fold_b")
    }
    write_json(output_root / "checks" / "preprocessing_statistics.json", preprocessing_states)
    seed_predictions, ensemble_predictions, training_rows, history = run_validation(
        configs, prepared, output_root, args.smoke, drive_run
    )
    seed_predictions.to_csv(output_root / "predictions" / "tcn_seed_predictions.csv", index=False)
    ensemble_predictions.to_csv(output_root / "predictions" / "tcn_ensemble_predictions.csv", index=False)
    training_df = pd.DataFrame(training_rows); training_df.to_csv(output_root / "metrics" / "training_runs.csv", index=False)
    history.to_csv(output_root / "metrics" / "training_history.csv", index=False)

    seed_tables = metric_tables(seed_predictions); ensemble_tables = metric_tables(ensemble_predictions)
    seed_tables["macro"].to_csv(output_root / "metrics" / "model_seed_metrics.csv", index=False)
    stability = (
        seed_tables["macro"].groupby(["experiment_id", "fold"], sort=False)["macro_nmae"]
        .agg(seed_mean_macro_nmae="mean", seed_std_macro_nmae="std").reset_index()
    )
    ensemble_summary = ensemble_tables["macro"].merge(stability, on=["experiment_id", "fold"], how="left")
    ensemble_summary.to_csv(output_root / "metrics" / "model_ensemble_metrics.csv", index=False)
    best_tcn, auxiliary_retained, aux_decisions = choose_tcn_config(ensemble_predictions, seed_predictions) if not args.smoke else ("tcn_plain", False, {})
    best_tcn_predictions = ensemble_predictions[ensemble_predictions.experiment_id.eq(best_tcn)].copy()
    catboost_reference = load_catboost_reference(args.catboost_reference)
    blend_search = pd.DataFrame(); best_blend_predictions = pd.DataFrame(); best_weight = 0.0; blend_info = {}; residual = {}
    combined_parts = [seed_predictions, ensemble_predictions]
    if catboost_reference is not None:
        catboost_reference = attach_high_wind(catboost_reference, best_tcn_predictions)
        catboost_reference.to_csv(output_root / "predictions" / "catboost_reference_predictions.csv", index=False)
        blend_cfg = load_yaml(CONFIG_DIR / "blend.yaml")
        best_seed_values = seed_tables["macro"][
            (seed_tables["macro"].fold == "fold_b") & (seed_tables["macro"].experiment_id == best_tcn)
        ].macro_nmae
        best_seed_std = float(best_seed_values.std())
        threshold = float(blend_cfg["selection"].get("seed_instability_threshold", 0.003))
        weights = list(blend_cfg["tcn_weights"])
        if np.isfinite(best_seed_std) and best_seed_std > threshold:
            maximum = float(blend_cfg["selection"].get("unstable_max_tcn_weight", 0.3))
            weights = [weight for weight in weights if float(weight) <= maximum]
        blend_search, candidates = search_blend_weights(catboost_reference, best_tcn_predictions, weights)
        best_weight, blend_info = select_blend_weight(blend_search)
        best_blend_predictions = candidates[candidates.tcn_weight.eq(best_weight)].copy()
        best_blend_predictions["experiment_id"] = "best_blend"; best_blend_predictions["seed"] = -1; best_blend_predictions["ensemble"] = True
        wind = best_tcn_predictions[["fold", TIME_COL, "target", "validation_wind_mps", "train_wind_p90_mps", "high_wind_mask"]]
        best_blend_predictions = best_blend_predictions.drop(columns=[c for c in ["validation_wind_mps", "train_wind_p90_mps", "high_wind_mask"] if c in best_blend_predictions]).merge(
            wind, on=["fold", TIME_COL, "target"], how="left", validate="one_to_one"
        )
        best_blend_predictions.to_csv(output_root / "predictions" / "best_blend_predictions.csv", index=False)
        residual = prediction_diagnostics(catboost_reference, best_tcn_predictions)
        residual["best_tcn_seed_std_macro_nmae"] = best_seed_std
        residual["seed_instability_threshold"] = threshold
        residual["blend_weights_limited_for_instability"] = bool(
            np.isfinite(best_seed_std) and best_seed_std > threshold
        )
        combined_parts.extend([catboost_reference, best_blend_predictions])
    elif not args.smoke:
        raise FileNotFoundError("CatBoost validation reference is required for the full run")
    blend_search.to_csv(output_root / "metrics" / "blend_search.csv", index=False)
    all_predictions = pd.concat(combined_parts, ignore_index=True, sort=False)
    all_predictions[all_predictions.fold.eq("fold_a")].to_csv(output_root / "predictions" / "fold_a_predictions.csv", index=False)
    all_predictions[all_predictions.fold.eq("fold_b")].to_csv(output_root / "predictions" / "fold_b_predictions.csv", index=False)
    final_tables = metric_tables(all_predictions)
    final_tables["macro"].to_csv(output_root / "metrics" / "fold_metrics.csv", index=False)
    final_tables["group"].to_csv(output_root / "metrics" / "group_metrics.csv", index=False)
    final_tables["monthly"].to_csv(output_root / "metrics" / "monthly_metrics.csv", index=False)
    final_tables["hourly"].to_csv(output_root / "metrics" / "hourly_metrics.csv", index=False)
    final_tables["high_wind"].to_csv(output_root / "metrics" / "high_wind_metrics.csv", index=False)
    final_tables["january"].to_csv(output_root / "metrics" / "january_metrics.csv", index=False)

    submission_path = None; full_info = {}
    if not args.smoke and not args.skip_full_train:
        if args.catboost_test is None or not args.catboost_test.exists():
            candidates = sorted((PROJECT_ROOT / "experiments" / "exp01_catboost_physics" / "outputs" / "submissions").glob("exp01_catboost_best_*.csv"))
            catboost_test = candidates[-1] if candidates else None
        else: catboost_test = args.catboost_test
        if catboost_test is None: raise FileNotFoundError("validated exp01 CatBoost test prediction is required")
        chosen_config = next(config for config in configs if config["experiment_id"] == best_tcn)
        epochs = training_df[(training_df.experiment_id == best_tcn) & (training_df.fold == "fold_b")].best_epoch.astype(int).tolist()
        submission_path, full_info = full_train_submission(
            chosen_config, best_tcn, epochs, best_weight, catboost_test, baseline_cfg,
            train_features, test_features, labels, scada, output_root, drive_run
        )

    all_ensemble = final_tables["macro"][final_tables["macro"].ensemble.eq(True)].copy()
    make_figures(history, seed_tables["macro"], all_ensemble, final_tables["group"], final_tables["monthly"],
                 final_tables["january"], final_tables["high_wind"], blend_search, all_predictions,
                 best_tcn, output_root / "figures")
    feature_manifest = json.loads((output_root / "checks" / "tcn_feature_manifest.json").read_text())
    cat_ref = 0.095007
    blend_fold_b = None
    if not best_blend_predictions.empty:
        blend_metric = metric_tables(best_blend_predictions)["macro"]
        blend_fold_b = float(blend_metric[blend_metric.fold.eq("fold_b")].macro_nmae.iloc[0])
    manifest = {
        "experiment": "exp02_daily_tcn_scada_aux", "run_id": run_id,
        "started_at": started.isoformat(), "finished_at": datetime.now().isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "issue_contract": issue_contract, "feature_count": feature_manifest["feature_count"],
        "best_tcn_config": best_tcn, "auxiliary_retained": auxiliary_retained, "auxiliary_decisions": aux_decisions,
        "residual_diagnostics": residual, "best_blend_weight": best_weight, "blend_selection": blend_info,
        "best_blend_fold_b_macro_nmae": blend_fold_b,
        "catboost_improvement": None if blend_fold_b is None else cat_ref - blend_fold_b,
        "full_training": full_info, "submission_path": None if submission_path is None else str(submission_path),
        "drive_artifact_path": None, "official_scorer_found": False,
        "next_experiment_change": "increase temporal receptive-field diversity only after checking January residuals",
    }
    archive_path = None if drive_run is None else drive_run / "outputs.tar.gz"
    if archive_path is not None:
        manifest["drive_artifact_path"] = str(archive_path)
    write_json(output_root / "run_manifest.json", manifest)
    write_report(output_root / "report.md", manifest, all_ensemble, final_tables["group"])
    if drive_run is not None:
        with tarfile.open(archive_path, "w:gz") as archive: archive.add(output_root, arcname="outputs")
        shutil.copy2(output_root / "run_manifest.json", drive_run / "run_manifest.json")
        shutil.copy2(output_root / "report.md", drive_run / "report.md")
        if submission_path is not None: shutil.copy2(submission_path, drive_run / submission_path.name)
    print(json.dumps({"output_root": str(output_root), "best_tcn": best_tcn, "blend_weight": best_weight,
                      "submission": None if submission_path is None else str(submission_path),
                      "drive_run": None if drive_run is None else str(drive_run)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
