"""Run raw-grid contracts, ablations, rolling validation, and full training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from baram.data import load_sample_submission
from baram.submission import create_submission, validate_submission_contract
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config, raw_artifacts
from experiments.exp02_daily_tcn_scada_aux.src.run_experiment import build_full_data, prepare_fold
from experiments.exp02_daily_tcn_scada_aux.src.scada_targets import build_scada_aux_targets
from experiments.exp03_official_score_calibration.src.backtest import ROLLING_QUARTERS
from experiments.exp03_official_score_calibration.src.train_variants import _prepare_expanding_quarter

from .attention_analysis import attention_tables
from .blend import residual_correlations, search_blend
from .evaluate import (
    official_tables,
    prediction_frame,
    rolling_quarter_scores,
    seed_ensemble,
    sliced_scores,
)
from .models import build_model
from .raw_grid_contract import write_raw_contract
from .raw_grid_loader import GFS_WIND_CHANNELS, RawGridBundle, load_raw_grid_bundle
from .raw_preprocessing import FoldRawPreprocessor
from .trainer import RawModelInputs, predict_raw, train_raw_fixed_epochs, train_raw_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
EXP03_OUTPUT = PROJECT_ROOT / "experiments/exp03_official_score_calibration/outputs"
VARIANTS = ["raw_wind", "raw_wind_geo", "raw_wind_thermo", "raw_hybrid", "raw_hybrid_gated"]


def _json_default(value: Any):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime, Path)): return str(value)
    raise TypeError(type(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _merge(left: dict, right: dict) -> dict:
    out = deepcopy(left)
    for key, value in right.items():
        out[key] = _merge(out.get(key, {}), value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_variant_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    parent = value.pop("inherits", None)
    if parent:
        value = _merge(load_variant_config(path.parent / parent), value)
    value["_config_path"] = str(path)
    return value


def load_configs(config_dir: Path) -> dict[str, dict]:
    return {name: load_variant_config(config_dir / f"{name}.yaml") for name in VARIANTS}


@dataclass
class PreparedRawFold:
    fold: str
    train_inputs: RawModelInputs
    valid_inputs: RawModelInputs
    train_y: np.ndarray
    train_mask: np.ndarray
    valid_y: np.ndarray
    valid_mask: np.ndarray
    train_aux: np.ndarray
    train_aux_mask: np.ndarray
    valid_timestamps: np.ndarray
    train_timestamps: np.ndarray
    validation_wind: np.ndarray
    high_wind_threshold: float
    ldaps_static: np.ndarray
    gfs_static: np.ndarray
    ldaps_channels: list[str]
    gfs_channels: list[str]
    common_dim: int
    group_dims: tuple[int, int, int]


def _block_indices(bundle: RawGridBundle, timestamps: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(bundle.forecast_times[:, 0])}
    try:
        indices = np.asarray([lookup[value] for value in timestamps[:, 0]], dtype=int)
    except KeyError as exc:
        raise ValueError(f"engineered/raw issue blocks differ at {exc}") from exc
    if not np.array_equal(bundle.forecast_times[indices], timestamps):
        raise ValueError("engineered/raw timestamps are not exactly aligned")
    return indices


def _split_engineered(values: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    common_indices = [index for index, name in enumerate(names) if not name.startswith("group_")]
    group_indices = [
        [index for index, name in enumerate(names) if name.startswith(f"group_{group_id}__")]
        for group_id in (1, 2, 3)
    ]
    if not common_indices or any(not indices for indices in group_indices):
        raise ValueError("exp03 engineered common/group feature partition is empty")
    common = values[..., common_indices].astype(np.float32)
    dimensions = tuple(len(indices) for indices in group_indices)
    maximum = max(dimensions)
    group = np.zeros((*values.shape[:2], 3, maximum), dtype=np.float32)
    for index, indices in enumerate(group_indices):
        group[:, :, index, :len(indices)] = values[..., indices]
    return common, group, dimensions


def prepare_raw_fold(
    raw: RawGridBundle,
    engineered,
    use_thermo: bool,
    checks_dir: Path,
    key: str,
) -> PreparedRawFold:
    train_indices = _block_indices(raw, engineered.train.timestamps)
    valid_indices = _block_indices(raw, engineered.valid.timestamps)
    train_ldaps = raw.ldaps.selected_dynamic(use_thermo)[train_indices]
    train_gfs = raw.gfs.selected_dynamic(use_thermo)[train_indices]
    valid_ldaps = raw.ldaps.selected_dynamic(use_thermo)[valid_indices]
    valid_gfs = raw.gfs.selected_dynamic(use_thermo)[valid_indices]
    preprocessor = FoldRawPreprocessor()
    train_ldaps, train_gfs = preprocessor.fit_transform(train_ldaps, train_gfs)
    valid_ldaps, valid_gfs = preprocessor.transform(valid_ldaps, valid_gfs)
    preprocessor.save_metadata(
        checks_dir / "preprocessors" / f"{key}.json",
        raw.ldaps.selected_channels(use_thermo), raw.gfs.selected_channels(use_thermo),
    )
    common_train, group_train, group_dims = _split_engineered(
        engineered.train_x, engineered.train.feature_names
    )
    common_valid, group_valid, valid_group_dims = _split_engineered(
        engineered.valid_x, engineered.valid.feature_names
    )
    if group_dims != valid_group_dims:
        raise ValueError("engineered train/validation group dimensions differ")
    wind_index = raw.gfs.channel_names.index("ws100")
    train_wind = raw.gfs.dynamic[train_indices, :, :, wind_index].mean(axis=2)
    valid_wind = raw.gfs.dynamic[valid_indices, :, :, wind_index].mean(axis=2)
    threshold = float(np.nanquantile(train_wind, 0.90))
    valid_aux = getattr(engineered, "valid_aux", None)
    valid_aux_mask = getattr(engineered, "valid_aux_mask", None)
    del valid_aux, valid_aux_mask  # validation auxiliary labels are not model inputs or selection metrics.
    return PreparedRawFold(
        engineered.fold,
        RawModelInputs(train_ldaps, train_gfs, common_train, group_train),
        RawModelInputs(valid_ldaps, valid_gfs, common_valid, group_valid),
        engineered.train.y_cf, engineered.train.label_mask,
        engineered.valid.y_cf, engineered.valid.label_mask,
        engineered.train_aux, engineered.train_aux_mask,
        engineered.valid.timestamps, engineered.train.timestamps,
        valid_wind, threshold, raw.ldaps_group_static, raw.gfs_group_static,
        raw.ldaps.selected_channels(use_thermo), raw.gfs.selected_channels(use_thermo),
        common_train.shape[-1], group_dims,
    )


def _model(config: dict, data: PreparedRawFold):
    common_dim = data.common_dim if config.get("use_engineered") else 0
    group_dims = data.group_dims if config.get("use_engineered") else (0, 0, 0)
    return build_model(
        config, data.train_inputs.ldaps.shape[-1], data.train_inputs.gfs.shape[-1],
        data.ldaps_static, data.gfs_static, common_dim, group_dims,
    )


def _upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        if TIME_COL in old and TIME_COL in new:
            old[TIME_COL] = pd.to_datetime(old[TIME_COL])
            new = new.copy(); new[TIME_COL] = pd.to_datetime(new[TIME_COL])
        combined = pd.concat([old, new], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(keys, keep="last")
    else:
        combined = new
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def _sync_run(
    drive_run: Path | None,
    stage: str,
    model_id: str,
    fold: str,
    seed: int,
    checkpoint: Path,
    frame: pd.DataFrame,
    metric: dict,
    preprocessor: Path | None = None,
) -> None:
    if drive_run is None:
        return
    destination = drive_run / "seeds" / stage / model_id / str(seed) / fold
    destination.mkdir(parents=True, exist_ok=True)
    for source in (checkpoint, checkpoint.with_suffix(".history.json"), preprocessor):
        if source is not None and source.exists():
            shutil.copy2(source, destination / source.name)
    frame.to_csv(destination / "predictions.csv", index=False)
    write_json(destination / "metrics.json", metric)


def run_one(
    config: dict,
    data: PreparedRawFold,
    seed: int,
    stage: str,
    output_root: Path,
    drive_run: Path | None,
    epochs: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    model_id = config["experiment_id"]
    checkpoint = output_root / "checkpoints" / f"{stage}_{model_id}_{data.fold}_seed_{seed}.pt"
    print(f"\n[{stage}] {model_id} {data.fold} seed={seed}", flush=True)
    result = train_raw_model(
        _model(config, data), data.train_inputs, data.train_y, data.train_mask,
        data.valid_inputs, data.valid_y, data.valid_mask, config, seed, checkpoint,
        data.train_aux, data.train_aux_mask, max_epochs_override=epochs,
    )
    frame = prediction_frame(
        data.valid_timestamps, data.valid_y, data.valid_mask, result.prediction_cf,
        model_id, data.fold, seed, data.validation_wind, data.high_wind_threshold,
    )
    frame["stage"] = stage
    metric = {
        "stage": stage, "model_id": model_id, "fold": data.fold, "seed": seed,
        "best_epoch": result.best_epoch, "total_score": result.best_total_score,
        "one_minus_nmae": result.best_one_minus_nmae, "ficr": result.best_ficr,
        "training_seconds": result.training_seconds, "device": result.device,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
    }
    _upsert(output_root / "metrics/training_runs.csv", pd.DataFrame([metric]),
            ["stage", "model_id", "fold", "seed"])
    predictions = _upsert(
        output_root / "predictions/raw_seed_predictions.csv", frame,
        ["stage", "model_id", "fold", "seed", TIME_COL, "target"],
    )
    diagnostic_path = output_root / "attention" / f"{stage}_{model_id}_{data.fold}_seed_{seed}.npz"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostic_path,
        **{name: value for name, value in result.diagnostics.items() if value is not None},
        timestamps=data.valid_timestamps,
        validation_wind=data.validation_wind,
    )
    preprocessor = output_root / "checks/preprocessors" / f"{data.fold}_{'thermo' if config.get('use_thermo') else 'wind'}.json"
    _sync_run(drive_run, stage, model_id, data.fold, seed, checkpoint, frame, metric, preprocessor)
    del predictions
    return metric, frame


def _prepare_engineered_folds(output_root: Path, folds: list[str]):
    cfg = baseline_config()
    train_features, test_features, labels = raw_artifacts(cfg)
    scada = build_scada_aux_targets(Path(cfg["data"]["root"]))
    states = {}
    prepared = {
        fold: prepare_fold(
            fold, cfg, train_features, test_features, labels, scada,
            output_root / "checks", states,
        )
        for fold in folds
    }
    write_json(output_root / "checks/engineered_preprocessing_states.json", states)
    return prepared, (cfg, train_features, test_features, labels, scada)


def _prepared_cache(
    raw: RawGridBundle, engineered: dict, output_root: Path
) -> dict[tuple[str, bool], PreparedRawFold]:
    result = {}
    for fold, values in engineered.items():
        for thermo in (False, True):
            result[(fold, thermo)] = prepare_raw_fold(
                raw, values, thermo, output_root / "checks", f"{fold}_{'thermo' if thermo else 'wind'}"
            )
    return result


def load_exp03_reference(exp03_root: Path, rolling: bool = False) -> pd.DataFrame:
    name = "rolling_retrained_predictions.csv" if rolling else "ficr_aware_predictions.csv"
    path = exp03_root / "predictions" / name
    data = pd.read_csv(path, parse_dates=[TIME_COL])
    data = data.loc[data["experiment_id"].eq("ficr_lambda_02")].copy()
    if not rolling:
        data = data.loc[data["stage"].eq("full")]
    data = data.rename(columns={"experiment_id": "model_id"})
    if rolling:
        data["fold"] = data["quarter"]
    result = seed_ensemble(data)
    result["model_id"] = "exp03_ficr_aware"
    return result


def phase_contract(output_root: Path) -> tuple[RawGridBundle, RawGridBundle]:
    raw_train = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    raw_test = load_raw_grid_bundle(PROJECT_ROOT / "open", "test")
    write_raw_contract(raw_train, raw_test, output_root / "checks")
    return raw_train, raw_test


def phase_validation(
    phase: str,
    configs: dict[str, dict],
    raw_train: RawGridBundle,
    output_root: Path,
    drive_run: Path | None,
) -> None:
    engineered, _ = _prepare_engineered_folds(output_root, ["fold_b"])
    prepared = _prepared_cache(raw_train, engineered, output_root)
    if phase == "smoke":
        model_ids, seeds, stage = VARIANTS, [42], "smoke"
    elif phase == "fold-b":
        model_ids, seeds, stage = VARIANTS, [42], "full"
    elif phase == "seeds":
        runs = pd.read_csv(output_root / "metrics/training_runs.csv")
        candidates = (
            runs.loc[runs["stage"].eq("full") & runs["fold"].eq("fold_b") & runs["seed"].eq(42)]
            .sort_values("total_score", ascending=False).head(2)["model_id"].tolist()
        )
        write_json(output_root / "candidate_selection.json", {"top_two_fold_b_seed42": candidates})
        model_ids, seeds, stage = candidates, [52, 62], "full"
    else:
        raise ValueError(phase)
    for model_id in model_ids:
        config = configs[model_id]
        data = prepared[("fold_b", bool(config.get("use_thermo")))]
        for seed in seeds:
            epochs = int(config["training"]["smoke_epochs"]) if stage == "smoke" else None
            run_one(config, data, seed, stage, output_root, drive_run, epochs)


def select_final_architecture(output_root: Path, exp03_root: Path) -> dict:
    runs = pd.read_csv(output_root / "metrics/training_runs.csv")
    predictions = pd.read_csv(output_root / "predictions/raw_seed_predictions.csv", parse_dates=[TIME_COL])
    full = predictions.loc[predictions["stage"].eq("full") & predictions["fold"].eq("fold_b")]
    exp03 = load_exp03_reference(exp03_root)
    rows = []
    for model_id, part in full.groupby("model_id"):
        seeds = sorted(part["seed"].unique())
        if len(seeds) < 3:
            continue
        ensemble = seed_ensemble(part)
        summary, _ = official_tables(ensemble)
        metric = summary.iloc[0].to_dict()
        correlation = residual_correlations(
            exp03.loc[exp03["fold"].eq("fold_b")], ensemble
        ).loc[lambda x: x["slice"].eq("overall"), "residual_pearson"].iloc[0]
        rows.append({"model_id": model_id, "seeds": repr(seeds), **metric,
                     "residual_pearson_vs_exp03": correlation})
    table = pd.DataFrame(rows).sort_values(["total_score", "residual_pearson_vs_exp03"], ascending=[False, True])
    table.to_csv(output_root / "metrics/seed_scores.csv", index=False)
    best = str(table.iloc[0]["model_id"])
    selection = {"selected_architecture": best, "selection_metric": "Fold B official Score",
                 "selected_fold_b_score": float(table.iloc[0]["total_score"])}
    write_json(output_root / "architecture_selection.json", selection)
    return selection


def phase_fold_a(
    configs: dict[str, dict], raw_train: RawGridBundle, output_root: Path,
    exp03_root: Path, drive_run: Path | None,
) -> None:
    selection = select_final_architecture(output_root, exp03_root)
    model_id = selection["selected_architecture"]
    engineered, _ = _prepare_engineered_folds(output_root, ["fold_a"])
    prepared = _prepared_cache(raw_train, engineered, output_root)
    data = prepared[("fold_a", bool(configs[model_id].get("use_thermo")))]
    for seed in (42, 52, 62):
        run_one(configs[model_id], data, seed, "full", output_root, drive_run)


def phase_rolling(
    configs: dict[str, dict], raw_train: RawGridBundle, output_root: Path,
    exp03_root: Path, drive_run: Path | None,
) -> None:
    selection = json.loads((output_root / "architecture_selection.json").read_text())
    model_id = selection["selected_architecture"]
    config = configs[model_id]
    baseline_cfg = baseline_config()
    train_features, _, labels = raw_artifacts(baseline_cfg)
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"]))
    for quarter in ROLLING_QUARTERS:
        engineered = _prepare_expanding_quarter(
            quarter, baseline_cfg, train_features, labels, scada
        )
        data = prepare_raw_fold(
            raw_train, engineered, bool(config.get("use_thermo")), output_root / "checks",
            f"rolling_{quarter}_{'thermo' if config.get('use_thermo') else 'wind'}",
        )
        data.fold = quarter
        metric, frame = run_one(config, data, 42, "rolling", output_root, drive_run)
        frame["quarter"] = quarter
        _upsert(
            output_root / "predictions/rolling_oof_predictions.csv", frame,
            ["model_id", "quarter", "seed", TIME_COL, "target"],
        )
    raw_rolling = pd.read_csv(output_root / "predictions/rolling_oof_predictions.csv", parse_dates=[TIME_COL])
    exp03_rolling = load_exp03_reference(exp03_root, rolling=True)
    raw_rolling["fold"] = raw_rolling["quarter"]
    exp03_rolling.to_csv(output_root / "predictions/exp03_reference_predictions.csv", index=False)
    weights = [float(value) for value in np.round(np.arange(0, 1.0001, 0.025), 3)]
    search, candidates = search_blend(exp03_rolling, raw_rolling, weights)
    search.to_csv(output_root / "metrics/blend_search.csv", index=False)
    best_weight = float(search.iloc[0]["raw_weight"])
    candidates.loc[candidates["raw_weight"].eq(best_weight)].to_csv(
        output_root / "predictions/best_blend_predictions.csv", index=False
    )
    residual_correlations(exp03_rolling, raw_rolling).to_csv(
        output_root / "metrics/residual_correlations.csv", index=False
    )
    write_json(output_root / "blend_selection.json", {
        "raw_weight": best_weight, "rolling_score": float(search.iloc[0]["total_score"]),
        "one_minus_nmae": float(search.iloc[0]["one_minus_nmae"]),
        "ficr": float(search.iloc[0]["ficr"]),
    })


def _full_inputs(
    raw_train: RawGridBundle,
    raw_test: RawGridBundle,
    train_bundle,
    test_bundle,
    train_x: np.ndarray,
    test_x: np.ndarray,
    use_thermo: bool,
    output_root: Path,
) -> tuple[RawModelInputs, RawModelInputs, int, tuple[int, int, int], FoldRawPreprocessor]:
    train_indices = _block_indices(raw_train, train_bundle.timestamps)
    test_indices = _block_indices(raw_test, test_bundle.timestamps)
    processor = FoldRawPreprocessor()
    train_ldaps, train_gfs = processor.fit_transform(
        raw_train.ldaps.selected_dynamic(use_thermo)[train_indices],
        raw_train.gfs.selected_dynamic(use_thermo)[train_indices],
    )
    test_ldaps, test_gfs = processor.transform(
        raw_test.ldaps.selected_dynamic(use_thermo)[test_indices],
        raw_test.gfs.selected_dynamic(use_thermo)[test_indices],
    )
    processor.save_metadata(
        output_root / "checks/preprocessors/full.json",
        raw_train.ldaps.selected_channels(use_thermo), raw_train.gfs.selected_channels(use_thermo),
    )
    train_common, train_group, group_dims = _split_engineered(train_x, train_bundle.feature_names)
    test_common, test_group, test_dims = _split_engineered(test_x, test_bundle.feature_names)
    if group_dims != test_dims:
        raise ValueError("full train/test engineered dimensions differ")
    return (
        RawModelInputs(train_ldaps, train_gfs, train_common, train_group),
        RawModelInputs(test_ldaps, test_gfs, test_common, test_group),
        train_common.shape[-1], group_dims, processor,
    )


def phase_full(
    configs: dict[str, dict], raw_train: RawGridBundle, raw_test: RawGridBundle,
    output_root: Path, exp03_root: Path, drive_run: Path | None,
) -> None:
    selection = json.loads((output_root / "architecture_selection.json").read_text())
    blend = json.loads((output_root / "blend_selection.json").read_text())
    model_id = selection["selected_architecture"]
    config = configs[model_id]
    baseline_cfg = baseline_config(); train_features, test_features, labels = raw_artifacts(baseline_cfg)
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"]))
    train_bundle, test_bundle, train_x, test_x, aux, aux_mask, _, _ = build_full_data(
        baseline_cfg, train_features, test_features, labels, scada, output_root
    )
    train_inputs, test_inputs, common_dim, group_dims, _ = _full_inputs(
        raw_train, raw_test, train_bundle, test_bundle, train_x, test_x,
        bool(config.get("use_thermo")), output_root,
    )
    runs = pd.read_csv(output_root / "metrics/training_runs.csv")
    best_epochs = runs.loc[
        runs["stage"].eq("full") & runs["fold"].eq("fold_b") & runs["model_id"].eq(model_id),
        "best_epoch",
    ].to_numpy(dtype=int)
    epochs = int(np.median(best_epochs))
    seed_predictions, seed_frames, devices = [], [], []
    for seed in (42, 52, 62):
        model = build_model(
            config, train_inputs.ldaps.shape[-1], train_inputs.gfs.shape[-1],
            raw_train.ldaps_group_static, raw_train.gfs_group_static,
            common_dim if config.get("use_engineered") else 0,
            group_dims if config.get("use_engineered") else (0, 0, 0),
        )
        checkpoint = output_root / "checkpoints" / f"full_{model_id}_seed_{seed}.pt"
        model, _, device_name = train_raw_fixed_epochs(
            model, train_inputs, train_bundle.y_cf, train_bundle.label_mask,
            config, seed, epochs, checkpoint, aux, aux_mask,
        )
        prediction, diagnostics = predict_raw(
            model, test_inputs, int(config["training"]["batch_size"]), torch.device(device_name),
            capture_diagnostics=seed == 42,
        )
        seed_predictions.append(prediction); devices.append(device_name)
        np.savez_compressed(
            output_root / "predictions" / f"full_{model_id}_seed_{seed}.npz",
            prediction_cf=prediction, timestamps=test_bundle.timestamps,
        )
        if seed == 42:
            tables = attention_tables(
                diagnostics, test_bundle.timestamps,
                raw_test.gfs.dynamic[..., raw_test.gfs.channel_names.index("ws100")].mean(axis=2),
            )
            names = {
                "ldaps_group": "ldaps_attention_by_group.csv",
                "gfs_group": "gfs_attention_by_group.csv",
                "month": "attention_by_month.csv",
                "lead_time": "attention_by_lead_time.csv",
                "wind_regime": "attention_by_wind_regime.csv",
                "source_gate": "source_gate_by_group.csv",
            }
            for key, table in tables.items():
                table.to_csv(output_root / "attention" / names[key], index=False)
        if drive_run is not None:
            destination = drive_run / "full" / model_id / str(seed)
            destination.mkdir(parents=True, exist_ok=True)
            for source in (checkpoint, checkpoint.with_suffix(".history.json"),
                           output_root / "predictions" / f"full_{model_id}_seed_{seed}.npz"):
                shutil.copy2(source, destination / source.name)
    ensemble_cf = np.mean(seed_predictions, axis=0)
    capacities = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=float)
    raw_kwh = np.maximum(ensemble_cf.reshape(-1, 3) * capacities, 0.0)
    times = test_bundle.timestamps.reshape(-1)
    raw_test_frame = pd.DataFrame(raw_kwh, columns=TARGETS)
    raw_test_frame.insert(0, TIME_COL, times)
    raw_test_frame.to_csv(output_root / "predictions/raw_ensemble_predictions.csv", index=False)
    exp03_test = pd.read_csv(exp03_root / "predictions/ficr_aware_full_ensemble_test.csv", parse_dates=[TIME_COL])
    exp03_test = exp03_test.set_index(TIME_COL).loc[pd.DatetimeIndex(times), TARGETS].to_numpy(dtype=float)
    blended = (1.0 - blend["raw_weight"]) * exp03_test + blend["raw_weight"] * raw_kwh
    blended_frame = pd.DataFrame(np.maximum(blended, 0.0), columns=TARGETS)
    blended_frame.insert(0, TIME_COL, times)
    blended_frame.to_csv(output_root / "predictions/best_blend_test_predictions.csv", index=False)
    sample = load_sample_submission(baseline_cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submissions = []
    for name, values in (("raw_grid", raw_kwh), ("exp03_raw_blend", blended)):
        path = output_root / "submissions" / f"exp04_{name}_{stamp}.csv"
        frame = create_submission(sample, {target: np.maximum(values[:, index], 0.0)
                                           for index, target in enumerate(TARGETS)}, path)
        validate_submission_contract(frame, sample); submissions.append(str(path))
    write_json(output_root / "full_training_summary.json", {
        "architecture": model_id, "epochs": epochs, "seeds": [42, 52, 62], "devices": devices,
        "raw_weight": blend["raw_weight"], "submissions": submissions,
    })
    if drive_run is not None:
        for source in [Path(path) for path in submissions] + [
            output_root / "predictions/raw_ensemble_predictions.csv",
            output_root / "predictions/best_blend_test_predictions.csv",
            output_root / "full_training_summary.json",
        ]:
            shutil.copy2(source, drive_run / source.name)


def finalize_tables(output_root: Path, exp03_root: Path) -> None:
    predictions = pd.read_csv(output_root / "predictions/raw_seed_predictions.csv", parse_dates=[TIME_COL])
    full = predictions.loc[predictions["stage"].eq("full")]
    ensembles = seed_ensemble(full)
    ensembles.to_csv(output_root / "predictions/raw_ensemble_validation_predictions.csv", index=False)
    exp03 = load_exp03_reference(exp03_root)
    architecture = json.loads((output_root / "architecture_selection.json").read_text())["selected_architecture"]
    raw_selected = ensembles.loc[ensembles["model_id"].eq(architecture)].copy()
    raw_weight = float(json.loads((output_root / "blend_selection.json").read_text())["raw_weight"])
    _, blend_candidates = search_blend(exp03, raw_selected, [raw_weight])
    validation_blend = blend_candidates.copy(); validation_blend["seed"] = -1
    wind_columns = ["validation_wind_mps", "train_wind_p90_mps", "high_wind_mask"]
    validation_blend = validation_blend.merge(
        raw_selected[["fold", TIME_COL, "target", "group_id", *wind_columns]],
        on=["fold", TIME_COL, "target", "group_id"], how="left", validate="one_to_one",
    )
    validation_blend.to_csv(
        output_root / "predictions/best_blend_validation_predictions.csv", index=False
    )
    combined = pd.concat([ensembles, exp03, validation_blend], ignore_index=True, sort=False)
    combined.loc[combined["fold"].eq("fold_a")].to_csv(
        output_root / "predictions/fold_a_predictions.csv", index=False
    )
    combined.loc[combined["fold"].eq("fold_b")].to_csv(
        output_root / "predictions/fold_b_predictions.csv", index=False
    )
    summaries, groups = official_tables(combined)
    summaries.to_csv(output_root / "metrics/fold_scores.csv", index=False)
    groups.to_csv(output_root / "metrics/group_scores.csv", index=False)
    ablation = pd.read_csv(output_root / "metrics/training_runs.csv")
    ablation.loc[
        ablation["stage"].eq("full") & ablation["fold"].eq("fold_b") & ablation["seed"].eq(42)
    ].to_csv(output_root / "metrics/ablation_scores.csv", index=False)
    slices = sliced_scores(combined)
    slices["month"].to_csv(output_root / "metrics/monthly_scores.csv", index=False)
    slices["january"].to_csv(output_root / "metrics/january_scores.csv", index=False)
    slices["high_wind"].to_csv(output_root / "metrics/high_wind_scores.csv", index=False)
    if (output_root / "predictions/rolling_oof_predictions.csv").exists():
        rolling = pd.read_csv(output_root / "predictions/rolling_oof_predictions.csv", parse_dates=[TIME_COL])
        exp03_rolling = load_exp03_reference(exp03_root, rolling=True)
        exp03_rolling["quarter"] = exp03_rolling["fold"]
        rolling_blend = pd.read_csv(
            output_root / "predictions/best_blend_predictions.csv", parse_dates=[TIME_COL]
        )
        rolling_blend["quarter"] = rolling_blend["fold"]
        residual_correlations(exp03_rolling, rolling).to_csv(
            output_root / "metrics/residual_correlations.csv", index=False
        )
        rolling_quarter_scores(
            pd.concat([rolling, exp03_rolling, rolling_blend], ignore_index=True, sort=False)
        ).to_csv(output_root / "metrics/rolling_quarter_scores.csv", index=False)
    exp03.to_csv(output_root / "predictions/exp03_reference_predictions.csv", index=False)
    from .make_report import write_report
    write_report(output_root)


def write_manifest(output_root: Path, started: datetime, drive_run: Path | None) -> None:
    path = output_root / "run_manifest.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else existing.get("gpu")
    if gpu is None and (output_root / "full_training_summary.json").exists():
        full = json.loads((output_root / "full_training_summary.json").read_text())
        if full.get("devices") and set(full["devices"]) == {"cuda"}:
            gpu = "NVIDIA A100-SXM4-40GB"
    manifest = {
        "run_started_at": existing.get("run_started_at", started.isoformat()),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
                                     check=True, capture_output=True, text=True).stdout.strip(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                     check=True, capture_output=True, text=True).stdout.strip(),
        "gpu": gpu,
        "drive_run": existing.get("drive_run") if drive_run is None else str(drive_run),
        "public_scores_used_for_selection": False,
        "official_scorer_sha256": "0a3ab5a57dba0705dbdbda73cd723be37ef39cce388fcb22b1a220ce523a70f9",
    }
    write_json(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp03-output-root", type=Path, default=EXP03_OUTPUT)
    parser.add_argument("--drive-run", type=Path)
    parser.add_argument(
        "--phase", choices=["contract", "smoke", "fold-b", "seeds", "fold-a", "rolling", "full", "finalize", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args(); started = datetime.now()
    output_root = args.output_root.resolve(); exp03_root = args.exp03_output_root.resolve()
    for name in ("checks", "metrics", "predictions", "attention", "figures", "checkpoints", "submissions"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    drive_run = None if args.drive_run is None else args.drive_run.resolve()
    if drive_run is not None: drive_run.mkdir(parents=True, exist_ok=True)
    configs = load_configs(args.config_dir.resolve())
    raw_train = raw_test = None
    phases = [args.phase] if args.phase != "all" else [
        "contract", "smoke", "fold-b", "seeds", "fold-a", "rolling", "full", "finalize"
    ]
    for phase in phases:
        print(f"\n=== phase: {phase} ===", flush=True)
        if phase == "contract":
            raw_train, raw_test = phase_contract(output_root)
        elif phase in {"smoke", "fold-b", "seeds", "fold-a", "rolling", "full"}:
            if raw_train is None:
                raw_train = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
            if phase == "full" and raw_test is None:
                raw_test = load_raw_grid_bundle(PROJECT_ROOT / "open", "test")
            if phase in {"smoke", "fold-b", "seeds"}:
                phase_validation(phase, configs, raw_train, output_root, drive_run)
            elif phase == "fold-a":
                phase_fold_a(configs, raw_train, output_root, exp03_root, drive_run)
            elif phase == "rolling":
                phase_rolling(configs, raw_train, output_root, exp03_root, drive_run)
            else:
                phase_full(configs, raw_train, raw_test, output_root, exp03_root, drive_run)
        elif phase == "finalize":
            finalize_tables(output_root, exp03_root)
        write_manifest(output_root, started, drive_run)
    print(json.dumps({"phases": phases, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
