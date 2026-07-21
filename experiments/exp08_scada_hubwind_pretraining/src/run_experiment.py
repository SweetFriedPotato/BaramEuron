"""Run Exp08 contracts, A100 rolling stages, selection, and gated finalization."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baram.constants import TARGETS, TIME_COL
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config, raw_artifacts
from experiments.exp02_daily_tcn_scada_aux.src.scada_targets import build_scada_aux_targets
from experiments.exp03_official_score_calibration.src.backtest import ROLLING_QUARTERS, expanding_quarter_window
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp03_official_score_calibration.src.train_variants import _prepare_expanding_quarter
from experiments.exp04_raw_grid_spatiotemporal.src.blend import residual_correlations
from experiments.exp04_raw_grid_spatiotemporal.src.evaluate import prediction_frame, sliced_scores
from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_contract import write_raw_contract
from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_loader import RawGridBundle, load_raw_grid_bundle
from experiments.exp04_raw_grid_spatiotemporal.src.run_experiment import (
    PreparedRawFold,
    load_exp03_reference,
    prepare_raw_fold,
)

from .blend import search_convex_blend
from .evaluate import acceptance, reproduce_exp04_reference, stage1_metric_tables, summarize_power_candidate
from .make_report import render_figures, write_manifest, write_report
from .scada_hourly_targets import (
    HubWindTargetScaler,
    build_hourly_scada_targets,
    target_arrays,
    write_hourly_checks,
)
from .stage1_crossfit import (
    expanding_crossfit_windows,
    forecast_hubwind_fallback,
    write_crossfit_contract,
)
from .stage1_model import build_stage1_model
from .stage2_dataset import FoldHubFeatureImputer, build_stage2_hub_features, write_stage2_feature_schema
from .stage2_model import build_stage2_model, feature_indices_for_variant
from .trainer import train_stage1, train_stage2
from .transfer import load_matching_encoder_weights, load_stage1_from_exp04


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp08_scada_hubwind_pretraining"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
EXP03_OUTPUT = PROJECT_ROOT / "experiments/exp03_official_score_calibration/outputs"
EXP04_OUTPUT = PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs"

STAGE1_VARIANTS = {
    "s1_a_median": ("stage1_hubwind_mean.yaml", 1, False),
    "s1_b_mean": ("stage1_hubwind_mean.yaml", 2, False),
    "s1_c_distribution": ("stage1_hubwind_distribution.yaml", 4, False),
    "s1_d_aux_init": ("stage1_hubwind_distribution.yaml", 4, True),
}
STAGE2_VARIANTS = {
    "s2_b_pretrained": ("stage2_pretrained_encoder.yaml", "pretrained_encoder"),
    "s2_c_explicit": ("stage2_explicit_hubwind.yaml", "explicit_hubwind"),
    "s2_d_distribution": ("stage2_explicit_hubwind.yaml", "distribution_hubwind"),
}


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, (np.integer,)): return int(item)
        if isinstance(item, (np.floating,)): return float(item)
        if isinstance(item, (np.bool_,)): return bool(item)
        if isinstance(item, (np.ndarray,)): return item.tolist()
        if isinstance(item, (Path, pd.Timestamp)): return str(item)
        raise TypeError(type(item))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def load_config(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def directories(output_root: Path) -> None:
    for name in ("checks", "metrics", "predictions", "checkpoints", "figures", "submissions"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def _upsert(path: Path, frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        for data in (old, frame):
            if TIME_COL in data:
                data[TIME_COL] = pd.to_datetime(data[TIME_COL])
        frame = pd.concat([old, frame], ignore_index=True, sort=False).drop_duplicates(keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)
    return frame


def _sync(paths: list[Path], drive_run: Path | None, relative: Path) -> None:
    if drive_run is None:
        return
    destination = drive_run / relative; destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, destination / path.name)


def _restore(paths: list[Path], drive_run: Path | None, relative: Path) -> bool:
    if all(path.exists() for path in paths):
        return True
    if drive_run is None:
        return False
    source = drive_run / relative
    for path in paths:
        remote = source / path.name
        if remote.exists() and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(remote, path)
    return all(path.exists() for path in paths)


@dataclass
class QuarterData:
    raw: PreparedRawFold
    train_hub_raw: np.ndarray
    train_hub_mask: np.ndarray
    valid_hub_raw: np.ndarray
    valid_hub_mask: np.ndarray
    train_hub_scaled: np.ndarray
    train_hub_scaled_mask: np.ndarray
    valid_hub_scaled: np.ndarray
    valid_hub_scaled_mask: np.ndarray
    scaler: HubWindTargetScaler


class DataContext:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.baseline_cfg = baseline_config()
        self.train_features, self.test_features, self.labels = raw_artifacts(self.baseline_cfg)
        self.raw_train = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
        self.exp02_scada = build_scada_aux_targets(Path(self.baseline_cfg["data"]["root"]))
        self.label_times = pd.to_datetime(self.labels[TIME_COL])
        self._cache: dict[str, QuarterData] = {}

    def quarter(self, quarter: str) -> QuarterData:
        if quarter in self._cache:
            return self._cache[quarter]
        window = expanding_quarter_window(quarter)
        hourly, _ = build_hourly_scada_targets(
            PROJECT_ROOT / "open", fit_end=window["train_end"], label_timestamps=self.label_times,
        )
        engineered = _prepare_expanding_quarter(
            quarter, self.baseline_cfg, self.train_features, self.labels, self.exp02_scada,
        )
        prepared = prepare_raw_fold(
            self.raw_train, engineered, True, self.output_root / "checks",
            f"exp08_{quarter}_thermo",
        )
        prepared.fold = quarter
        train_raw, train_mask = target_arrays(hourly, prepared.train_timestamps)
        valid_raw, valid_mask = target_arrays(hourly, prepared.valid_timestamps)
        scaler = HubWindTargetScaler().fit(train_raw, train_mask)
        train_scaled, clean_train = scaler.transform(train_raw, train_mask)
        valid_scaled, clean_valid = scaler.transform(valid_raw, valid_mask)
        scaler.save(self.output_root / "checks/preprocessors" / f"hub_target_{quarter}.json")
        result = QuarterData(prepared, train_raw, train_mask, valid_raw, valid_mask,
                             train_scaled, clean_train, valid_scaled, clean_valid, scaler)
        self._cache[quarter] = result
        return result


def _exp04_checkpoint(exp04_root: Path, quarter: str) -> Path:
    path = exp04_root / "checkpoints" / f"rolling_raw_hybrid_gated_{quarter}_seed_42.pt"
    if not path.exists():
        raise FileNotFoundError(f"Exp04 rolling checkpoint required: {path}")
    return path


def _stage1_model(config: dict, data: PreparedRawFold):
    return build_stage1_model(
        config, data.train_inputs.ldaps.shape[-1], data.train_inputs.gfs.shape[-1],
        data.ldaps_static, data.gfs_static, data.common_dim, data.group_dims,
    )


def _stage2_model(config: dict, data: PreparedRawFold):
    return build_stage2_model(
        config, data.train_inputs.ldaps.shape[-1], data.train_inputs.gfs.shape[-1],
        data.ldaps_static, data.gfs_static, data.common_dim, data.group_dims,
    )


def _stage1_frame(npz_path: Path, model_id: str, seed: int, quarter: str) -> pd.DataFrame:
    data = np.load(npz_path)
    prediction, target, mask, timestamps = data["prediction"], data["target"], data["mask"], data["timestamps"]
    parts = []
    for group in range(3):
        frame = pd.DataFrame({TIME_COL: timestamps.reshape(-1), "group_id": group + 1})
        frame["lead_hour"] = np.tile(np.arange(1, 25), prediction.shape[0])
        for index, name in enumerate(("median", "mean", "std", "iqr")):
            frame[f"predicted_hub_ws_{name}"] = prediction[..., group, index].reshape(-1)
            frame[f"scada_hub_ws_{name}"] = target[..., group, index].reshape(-1)
            frame[f"target_mask_{name}"] = mask[..., group, index].reshape(-1)
        frame["model_id"], frame["seed"], frame["quarter"] = model_id, int(seed), quarter
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def phase_contracts(output_root: Path) -> dict:
    directories(output_root)
    reference = reproduce_exp04_reference(
        EXP04_OUTPUT / "predictions/best_blend_predictions.csv",
        output_root / "checks/reference_reproduction.json",
    )
    labels = pd.read_csv(PROJECT_ROOT / "open/train/train_labels.csv", encoding="utf-8-sig", usecols=["kst_dtm"])
    hourly, cleaner = build_hourly_scada_targets(
        PROJECT_ROOT / "open", label_timestamps=pd.to_datetime(labels["kst_dtm"]),
    )
    write_hourly_checks(hourly, cleaner, PROJECT_ROOT / "open", output_root / "checks")
    raw_train = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    raw_test = load_raw_grid_bundle(PROJECT_ROOT / "open", "test")
    write_raw_contract(raw_train, raw_test, output_root / "checks")
    records = {quarter: expanding_crossfit_windows(
        np.concatenate([raw_train.forecast_times[
            pd.DatetimeIndex(raw_train.forecast_times[:, 0]) <= expanding_quarter_window(quarter)["valid_end"]
        ]], axis=0), quarter,
    ) for quarter in ROLLING_QUARTERS}
    write_crossfit_contract(records, output_root / "checks/stage1_crossfit_contract.json")
    write_stage2_feature_schema(
        output_root / "checks/stage2_feature_schema.json", "distribution_hubwind", feature_indices_for_variant("distribution_hubwind")
    )
    leakage = {
        "scada_is_training_target_only": True,
        "stage1_inputs": ["LDAPS raw grid", "GFS raw grid", "static geometry", "time/lead", "Exp01 engineered weather"],
        "test_scada_access": False,
        "target_or_target_lag_input": False,
        "forecast_disagreement_input": False,
        "crossfit_in_sample_predictions": False,
        "normalization_scope": "fold_train_only",
        "public_used_for_selection": False,
        "exp07_finetuned_checkpoint_used": False,
    }
    write_json(output_root / "checks/leakage_audit.json", leakage)
    return {"reference": reference, "leakage": leakage}


def run_stage1_quarter(
    context: DataContext, model_id: str, seed: int, quarter: str,
    output_root: Path, exp04_root: Path, drive_run: Path | None,
    smoke_epochs: int | None = None,
) -> tuple[Path, Path]:
    config_name, target_count, auxiliary_init = STAGE1_VARIANTS[model_id]
    config = deepcopy(load_config(config_name)); config["stage1"]["target_count"] = target_count
    data = context.quarter(quarter)
    checkpoint = output_root / f"checkpoints/stage1/{model_id}/{seed}/{quarter}.pt"
    prediction_path = output_root / f"predictions/stage1/{model_id}/{seed}/{quarter}.npz"
    relative = Path("stage1") / model_id / str(seed) / quarter
    if _restore([checkpoint, prediction_path], drive_run, relative):
        return checkpoint, prediction_path
    model = _stage1_model(config, data.raw)
    init = load_stage1_from_exp04(
        model, _exp04_checkpoint(exp04_root, quarter), auxiliary_init=auxiliary_init,
    )
    write_json(output_root / f"checks/transfer/stage1_{model_id}_{quarter}.json", init)
    if smoke_epochs is not None:
        train_indices = np.arange(max(0, len(data.raw.train_inputs) - 16), len(data.raw.train_inputs))
        valid_indices = np.arange(min(8, len(data.raw.valid_inputs)))
        train_inputs, valid_inputs = data.raw.train_inputs.subset(train_indices), data.raw.valid_inputs.subset(valid_indices)
        train_target, train_mask = data.train_hub_scaled[train_indices], data.train_hub_scaled_mask[train_indices]
        valid_target, valid_mask = data.valid_hub_raw[valid_indices], data.valid_hub_mask[valid_indices]
        valid_times = data.raw.valid_timestamps[valid_indices]
    else:
        train_inputs, valid_inputs = data.raw.train_inputs, data.raw.valid_inputs
        train_target, train_mask = data.train_hub_scaled, data.train_hub_scaled_mask
        valid_target, valid_mask, valid_times = data.valid_hub_raw, data.valid_hub_mask, data.raw.valid_timestamps
    result = train_stage1(
        model, train_inputs, train_target, train_mask, valid_inputs, valid_target, valid_mask,
        data.scaler, config, seed, checkpoint, max_epochs_override=smoke_epochs,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, prediction=result.prediction_mps, target=valid_target,
                        mask=valid_mask, timestamps=valid_times, best_epoch=result.best_epoch)
    _sync([checkpoint, checkpoint.with_suffix(".history.json"), prediction_path], drive_run, relative)
    return checkpoint, prediction_path


def _stage1_variants_for_seed(output_root: Path, seed: int, requested: str | None) -> list[str]:
    if requested:
        return [requested]
    if seed == 42:
        return list(STAGE1_VARIANTS)
    selection = json.loads((output_root / "stage1_selection.json").read_text(encoding="utf-8"))
    return list(selection["top_two"])


def phase_stage1(
    output_root: Path, seed: int, model_id: str | None,
    exp04_root: Path, drive_run: Path | None,
) -> dict:
    context = DataContext(output_root)
    variants = _stage1_variants_for_seed(output_root, seed, model_id)
    all_frames = []
    rows = []
    for variant in variants:
        active = STAGE1_VARIANTS[variant][1]
        variant_frames = []
        quarter_mae = []
        for quarter in ROLLING_QUARTERS:
            _, npz_path = run_stage1_quarter(context, variant, seed, quarter, output_root, exp04_root, drive_run)
            frame = _stage1_frame(npz_path, variant, seed, quarter); variant_frames.append(frame); all_frames.append(frame)
            data = np.load(npz_path)
            tables = stage1_metric_tables(data["prediction"], data["target"], data["mask"], data["timestamps"])
            median = tables["group"].loc[tables["group"]["target"].eq("hub_ws_median")]
            quarter_mae.append(float(median["mae"].mean()))
        points = pd.concat(variant_frames, ignore_index=True)
        valid = points["target_mask_median"].astype(bool)
        error = (points.loc[valid, "predicted_hub_ws_median"] - points.loc[valid, "scada_hub_ws_median"]).abs()
        correlation = points.loc[valid, ["predicted_hub_ws_median", "scada_hub_ws_median"]].corr().iloc[0, 1]
        rows.append({"model_id": variant, "seed": seed, "target_count": active,
                     "group_balanced_median_mae": float(error.groupby(points.loc[valid, "group_id"]).mean().mean()),
                     "median_pearson": float(correlation), "quarter_mae_std": float(np.std(quarter_mae)),
                     "stage2_score": np.nan})
    combined = _upsert(output_root / "predictions/stage1_oof_hubwind.csv", pd.concat(all_frames, ignore_index=True),
                       ["model_id", "seed", "quarter", TIME_COL, "group_id"])
    ablation = _upsert(output_root / "metrics/stage1_ablation.csv", pd.DataFrame(rows), ["model_id", "seed"])
    # Detailed tables for the current best physical model/seed.
    eligible = ablation.loc[
        ablation["seed"].eq(seed) & ablation["target_count"].eq(4)
    ].sort_values(["group_balanced_median_mae", "quarter_mae_std"])
    if seed == 42:
        top_two = eligible.head(2)["model_id"].tolist()
        write_json(output_root / "stage1_selection.json", {
            "selection_basis": "Stage-1 median MAE/correlation pending Stage-2 official-score tie-break",
            "selected_model": top_two[0], "top_two": top_two,
        })
    selected = json.loads((output_root / "stage1_selection.json").read_text())["selected_model"]
    selected_points = combined.loc[combined["model_id"].eq(selected) & combined["seed"].eq(seed)]
    metric_rows = []
    for group_id, part in selected_points.groupby("group_id"):
        for target_name in ("median", "mean", "std", "iqr"):
            valid = part[f"target_mask_{target_name}"].astype(bool)
            if not valid.any(): continue
            true, pred = part.loc[valid, f"scada_hub_ws_{target_name}"], part.loc[valid, f"predicted_hub_ws_{target_name}"]
            metric_rows.append({"model_id": selected, "seed": seed, "group_id": group_id,
                                "target": f"hub_ws_{target_name}", "samples": int(valid.sum()),
                                "mae": float((pred-true).abs().mean()), "rmse": float(np.sqrt(((pred-true)**2).mean())),
                                "pearson": float(pred.corr(true)), "spearman": float(pred.rank().corr(true.rank()))})
    _upsert(output_root / "metrics/stage1_group_metrics.csv", pd.DataFrame(metric_rows),
            ["model_id", "seed", "group_id", "target"])
    quarter_rows, wind_rows, lead_rows = [], [], []
    for quarter, part in selected_points.groupby("quarter", sort=True):
        valid = part["target_mask_median"].astype(bool)
        group_mae = (part.loc[valid, "predicted_hub_ws_median"] - part.loc[valid, "scada_hub_ws_median"]).abs().groupby(part.loc[valid, "group_id"]).mean()
        quarter_rows.append({"model_id": selected, "seed": seed, "quarter": quarter,
                             "group_balanced_mae": float(group_mae.mean()),
                             "pearson": float(part.loc[valid, "predicted_hub_ws_median"].corr(part.loc[valid, "scada_hub_ws_median"]))})
    median_points = selected_points.loc[selected_points["target_mask_median"].astype(bool)].copy()
    median_points["wind_regime"] = pd.cut(median_points["scada_hub_ws_median"], [-np.inf, 4.0, 10.0, np.inf], labels=["low", "mid", "high"])
    for (group_id, regime), part in median_points.groupby(["group_id", "wind_regime"], observed=True, sort=True):
        wind_rows.append({"model_id": selected, "seed": seed, "group_id": group_id, "wind_regime": regime,
                          "samples": len(part), "mae": float((part["predicted_hub_ws_median"]-part["scada_hub_ws_median"]).abs().mean())})
    for (group_id, lead), part in median_points.groupby(["group_id", "lead_hour"], sort=True):
        lead_rows.append({"model_id": selected, "seed": seed, "group_id": group_id, "lead_hour": lead,
                          "samples": len(part), "mae": float((part["predicted_hub_ws_median"]-part["scada_hub_ws_median"]).abs().mean())})
    _upsert(output_root / "metrics/stage1_quarter_metrics.csv", pd.DataFrame(quarter_rows), ["model_id", "seed", "quarter"])
    _upsert(output_root / "metrics/stage1_wind_regime_metrics.csv", pd.DataFrame(wind_rows), ["model_id", "seed", "group_id", "wind_regime"])
    _upsert(output_root / "metrics/stage1_lead_time_metrics.csv", pd.DataFrame(lead_rows), ["model_id", "seed", "group_id", "lead_hour"])
    return {"variants": variants, "selection": json.loads((output_root / "stage1_selection.json").read_text())}


def _gfs_ws100(raw: RawGridBundle, timestamps: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(raw.forecast_times[:, 0])}
    indices = np.asarray([lookup[value] for value in timestamps[:, 0]], dtype=int)
    channel = raw.gfs.channel_names.index("ws100")
    return raw.gfs.dynamic[indices, :, :, channel]


def _load_stage1_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path); return data["prediction"], data["timestamps"]


def _map_prediction(destination: np.ndarray, fallback: np.ndarray, destination_times: np.ndarray,
                    prediction: np.ndarray, source_times: np.ndarray) -> None:
    lookup = {value: index for index, value in enumerate(destination_times[:, 0])}
    for source_index, first in enumerate(source_times[:, 0]):
        destination_index = lookup.get(first)
        if destination_index is not None:
            if not np.array_equal(destination_times[destination_index], source_times[source_index]):
                raise ValueError("Stage-1 OOF issue timestamps differ from Stage-2")
            destination[destination_index] = prediction[source_index]; fallback[destination_index] = 0.0


def crossfit_stage2_features(
    context: DataContext, data: QuarterData, outer: str, stage1_model: str, seed: int,
    output_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    train_pred = forecast_hubwind_fallback(_gfs_ws100(context.raw_train, data.raw.train_timestamps))
    valid_pred = forecast_hubwind_fallback(_gfs_ws100(context.raw_train, data.raw.valid_timestamps))
    train_fallback = np.ones(train_pred.shape[:-1], dtype=np.float32)
    valid_fallback = np.ones(valid_pred.shape[:-1], dtype=np.float32)
    available_seeds = [value for value in (42, 52, 62) if (output_root / f"predictions/stage1/{stage1_model}/{value}/{outer}.npz").exists()]
    train_seed_predictions, valid_seed_predictions = [], []
    for candidate_seed in available_seeds or [seed]:
        candidate_train = train_pred.copy(); candidate_valid = valid_pred.copy()
        candidate_train_fallback = train_fallback.copy(); candidate_valid_fallback = valid_fallback.copy()
        for previous in ROLLING_QUARTERS:
            if pd.Period(previous, freq="Q") >= pd.Period(outer, freq="Q"): break
            path = output_root / f"predictions/stage1/{stage1_model}/{candidate_seed}/{previous}.npz"
            if path.exists():
                prediction, times = _load_stage1_prediction(path)
                _map_prediction(candidate_train, candidate_train_fallback, data.raw.train_timestamps, prediction, times)
        current = output_root / f"predictions/stage1/{stage1_model}/{candidate_seed}/{outer}.npz"
        if current.exists():
            prediction, times = _load_stage1_prediction(current)
            _map_prediction(candidate_valid, candidate_valid_fallback, data.raw.valid_timestamps, prediction, times)
        train_seed_predictions.append(candidate_train); valid_seed_predictions.append(candidate_valid)
        train_fallback = np.minimum(train_fallback, candidate_train_fallback)
        valid_fallback = np.minimum(valid_fallback, candidate_valid_fallback)
    train_stack, valid_stack = np.stack(train_seed_predictions), np.stack(valid_seed_predictions)
    train_mean, valid_mean = train_stack.mean(axis=0), valid_stack.mean(axis=0)
    train_std = train_stack[..., 0].std(axis=0); valid_std = valid_stack[..., 0].std(axis=0)
    train_forecast = np.nanmean(_gfs_ws100(context.raw_train, data.raw.train_timestamps), axis=2)
    valid_forecast = np.nanmean(_gfs_ws100(context.raw_train, data.raw.valid_timestamps), axis=2)
    train = build_stage2_hub_features(train_mean, train_forecast, seed_std=train_std, fallback_indicator=train_fallback)
    valid = build_stage2_hub_features(valid_mean, valid_forecast, seed_std=valid_std, fallback_indicator=valid_fallback)
    imputer = FoldHubFeatureImputer().fit(train)
    imputer.save(output_root / f"checks/preprocessors/hub_features_{outer}.json")
    return imputer.transform(train), imputer.transform(valid)


def _load_compatible(model: torch.nn.Module, checkpoint: Path) -> dict:
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    state = model.state_dict(); loaded = []
    for name, value in source.items():
        if name in state and state[name].shape == value.shape:
            state[name] = value; loaded.append(name)
    model.load_state_dict(state); return {"loaded": loaded, "source": str(checkpoint)}


def run_stage2_quarter(
    context: DataContext, model_id: str, seed: int, quarter: str, stage1_model: str,
    output_root: Path, exp04_root: Path, drive_run: Path | None,
    smoke_epochs: int | None = None,
) -> tuple[Path, Path]:
    config_name, variant = STAGE2_VARIANTS[model_id]
    config = deepcopy(load_config(config_name)); config["stage2"]["variant"] = variant
    data = context.quarter(quarter)
    checkpoint = output_root / f"checkpoints/stage2/{model_id}/{seed}/{quarter}.pt"
    prediction_path = output_root / f"predictions/stage2/{model_id}/{seed}/{quarter}.npz"
    relative = Path("stage2") / model_id / str(seed) / quarter
    if _restore([checkpoint, prediction_path], drive_run, relative): return checkpoint, prediction_path
    train_hub, valid_hub = crossfit_stage2_features(context, data, quarter, stage1_model, seed, output_root)
    model = _stage2_model(config, data.raw)
    exp04_init = _load_compatible(model, _exp04_checkpoint(exp04_root, quarter))
    stage1_checkpoint = output_root / f"checkpoints/stage1/{stage1_model}/{seed}/{quarter}.pt"
    if not stage1_checkpoint.exists(): stage1_checkpoint = output_root / f"checkpoints/stage1/{stage1_model}/42/{quarter}.pt"
    stage1_init = load_matching_encoder_weights(model, stage1_checkpoint)
    write_json(output_root / f"checks/transfer/stage2_{model_id}_{quarter}.json", {"exp04": exp04_init, "stage1": stage1_init})
    if smoke_epochs is not None:
        ti = np.arange(max(0, len(data.raw.train_inputs)-16), len(data.raw.train_inputs)); vi = np.arange(min(8, len(data.raw.valid_inputs)))
        train_inputs, valid_inputs = data.raw.train_inputs.subset(ti), data.raw.valid_inputs.subset(vi)
        train_hub, valid_hub = train_hub[ti], valid_hub[vi]
        train_y, train_mask = data.raw.train_y[ti], data.raw.train_mask[ti]
        valid_y, valid_mask, valid_times = data.raw.valid_y[vi], data.raw.valid_mask[vi], data.raw.valid_timestamps[vi]
        retention, retention_mask = data.train_hub_scaled[ti], data.train_hub_scaled_mask[ti]
    else:
        train_inputs, valid_inputs = data.raw.train_inputs, data.raw.valid_inputs
        train_y, train_mask, valid_y, valid_mask, valid_times = data.raw.train_y, data.raw.train_mask, data.raw.valid_y, data.raw.valid_mask, data.raw.valid_timestamps
        retention, retention_mask = data.train_hub_scaled, data.train_hub_scaled_mask
    result = train_stage2(model, train_inputs, train_hub, train_y, train_mask,
                          valid_inputs, valid_hub, valid_y, valid_mask, config, seed, checkpoint,
                          retention_target=retention, retention_mask=retention_mask,
                          max_epochs_override=smoke_epochs)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, prediction=result.prediction_cf, target=valid_y, mask=valid_mask,
                        timestamps=valid_times, validation_wind=data.raw.validation_wind[:len(valid_y)],
                        high_wind_threshold=data.raw.high_wind_threshold, best_epoch=result.best_epoch)
    _sync([checkpoint, checkpoint.with_suffix(".history.json"), prediction_path], drive_run, relative)
    return checkpoint, prediction_path


def phase_stage2(
    output_root: Path, seed: int, model_id: str | None, exp04_root: Path, drive_run: Path | None,
) -> dict:
    selection = json.loads((output_root / "stage1_selection.json").read_text())
    stage1_model = selection["selected_model"]
    if model_id:
        variants = [model_id]
    elif seed == 42:
        variants = list(STAGE2_VARIANTS)
    else:
        variants = list(json.loads((output_root / "stage2_selection.json").read_text())["top_two"])
    context = DataContext(output_root); frames = []
    for variant in variants:
        for quarter in ROLLING_QUARTERS:
            _, path = run_stage2_quarter(context, variant, seed, quarter, stage1_model, output_root, exp04_root, drive_run)
            data = np.load(path)
            frame = prediction_frame(data["timestamps"], data["target"], data["mask"], data["prediction"],
                                     variant, quarter, seed, data["validation_wind"], float(data["high_wind_threshold"]))
            frame["quarter"] = quarter; frames.append(frame)
    combined = _upsert(output_root / "predictions/stage2_oof_predictions.csv", pd.concat(frames, ignore_index=True),
                       ["model_id", "seed", "quarter", TIME_COL, "target"])
    rows = []
    for (variant, candidate_seed), part in combined.groupby(["model_id", "seed"]):
        summary, _ = score_available_groups(part)
        rows.append({"model_id": variant, "seed": candidate_seed, **summary})
    scores = pd.DataFrame(rows).sort_values("total_score", ascending=False)
    scores.to_csv(output_root / "metrics/stage2_candidate_scores.csv", index=False)
    if seed == 42:
        top_two = scores.loc[scores["seed"].eq(42)].head(2)["model_id"].tolist()
        write_json(output_root / "stage2_selection.json", {"selected_model": top_two[0], "top_two": top_two})
    return {"variants": variants, "scores": scores.to_dict("records")}


def phase_smoke(output_root: Path, exp04_root: Path, drive_run: Path | None) -> dict:
    smoke_root = output_root / "smoke"; directories(smoke_root)
    smoke_drive = None if drive_run is None else drive_run / "smoke"
    context = DataContext(smoke_root); quarter = ROLLING_QUARTERS[0]
    _, stage1_path = run_stage1_quarter(context, "s1_c_distribution", 42, quarter, smoke_root, exp04_root, smoke_drive, smoke_epochs=3)
    write_json(smoke_root / "stage1_selection.json", {"selected_model": "s1_c_distribution", "top_two": ["s1_c_distribution"]})
    _, stage2_path = run_stage2_quarter(context, "s2_d_distribution", 42, quarter, "s1_c_distribution", smoke_root, exp04_root, smoke_drive, smoke_epochs=3)
    payload = {"seed": 42, "stage1_epochs": 3, "stage2_epochs": 3,
               "stage1_prediction": str(stage1_path), "stage2_prediction": str(stage2_path),
               "forward_backward_verified": True}
    write_json(output_root / "checks/smoke.json", payload); return payload


def _seed_ensemble(frame: pd.DataFrame, model_id: str) -> pd.DataFrame:
    part = frame.loc[frame["model_id"].eq(model_id)]
    keys = ["fold", "quarter", TIME_COL, "target", "group_id"]
    aggregation = {"y_true_kwh": "first", "y_pred_kwh": "mean"}
    for column in ("validation_wind_mps", "train_wind_p90_mps", "high_wind_mask"):
        if column in part: aggregation[column] = "first"
    out = part.groupby(keys, sort=False).agg(aggregation).reset_index(); out["model_id"] = model_id; out["seed"] = -1
    return out


def phase_finalize(output_root: Path) -> dict:
    stage2 = pd.read_csv(output_root / "predictions/stage2_oof_predictions.csv", parse_dates=[TIME_COL])
    reference = pd.read_csv(EXP04_OUTPUT / "predictions/best_blend_predictions.csv", parse_dates=[TIME_COL])
    reference["quarter"] = reference["fold"]
    selections = json.loads((output_root / "stage2_selection.json").read_text())
    candidates = []
    for model_id in selections["top_two"]:
        candidates.append(_seed_ensemble(stage2, model_id))
    candidate_scores = []
    for candidate in candidates:
        summary, groups = score_available_groups(candidate)
        candidate_scores.append({"model_id": candidate["model_id"].iloc[0], **summary})
    pd.DataFrame(candidate_scores).to_csv(output_root / "metrics/final_candidate_scores.csv", index=False)
    best_model = max(candidate_scores, key=lambda row: row["total_score"])["model_id"]
    exp08 = next(frame for frame in candidates if frame["model_id"].iloc[0] == best_model)
    exp03 = load_exp03_reference(EXP03_OUTPUT, rolling=True)
    raw = pd.read_csv(EXP04_OUTPUT / "predictions/rolling_oof_predictions.csv", parse_dates=[TIME_COL])
    for frame in (exp03, raw):
        if "fold" not in frame: frame["fold"] = frame["quarter"]
    wind_columns = ["validation_wind_mps", "train_wind_p90_mps", "high_wind_mask"]
    wind = raw[["fold", TIME_COL, "target", "group_id", *wind_columns]].drop_duplicates(["fold", TIME_COL, "target", "group_id"])
    if "high_wind_mask" not in reference:
        reference = reference.merge(wind, on=["fold", TIME_COL, "target", "group_id"], how="left", validate="one_to_one")
    search, blend = search_convex_blend({"exp03": exp03, "exp04_raw": raw, "exp08": exp08})
    blend = blend.merge(wind, on=["fold", TIME_COL, "target", "group_id"], how="left", validate="one_to_one")
    search.to_csv(output_root / "metrics/blend_search.csv", index=False)
    blend["quarter"] = blend["fold"]; blend.to_csv(output_root / "predictions/final_blend_predictions.csv", index=False)
    candidate = summarize_power_candidate(blend, reference)
    champion = summarize_power_candidate(reference)
    seed_scores = []
    for seed, part in stage2.loc[stage2["model_id"].eq(best_model)].groupby("seed"):
        seed_scores.append(score_available_groups(part)[0]["total_score"])
    decision = acceptance(candidate, champion, seed_scores)
    quarter_rows = []
    for model_name, frame in (("exp04", reference), (best_model, exp08), ("final_blend", blend)):
        for quarter, part in frame.groupby("quarter"):
            quarter_rows.append({"model_id": model_name, "quarter": quarter, **score_available_groups(part)[0]})
    pd.DataFrame(quarter_rows).to_csv(output_root / "metrics/nested_quarter_scores.csv", index=False)
    _, group_scores = score_available_groups(blend); group_scores.insert(0, "model_id", "final_blend")
    group_scores.to_csv(output_root / "metrics/group_scores.csv", index=False)
    residual = residual_correlations(reference, exp08); residual.insert(0, "model_id", best_model)
    residual.to_csv(output_root / "metrics/residual_correlations.csv", index=False)
    slices = sliced_scores(pd.concat([reference.assign(model_id="exp04"), exp08, blend.assign(model_id="final_blend")], ignore_index=True))
    slices["january"].to_csv(output_root / "metrics/january_scores.csv", index=False)
    slices["high_wind"].to_csv(output_root / "metrics/high_wind_scores.csv", index=False)
    write_json(output_root / "final_selection.json", {
        "best_exp08_model": best_model,
        "blend": {column: float(search.iloc[0][column]) for column in search if column.startswith("weight_")},
        "candidate_score": candidate["total_score"], "reference_score": champion["total_score"],
        "acceptance": decision, "submissions": [],
        "full_training_allowed": bool(decision["accepted"]),
        "next_direction": "Run gated full training only if acceptance passed; otherwise retain Exp04 champion.",
    })
    render_figures(output_root); return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["contracts", "smoke", "stage1", "stage2", "finalize", "report", "all"], required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp04-root", type=Path, default=EXP04_OUTPUT)
    parser.add_argument("--drive-run", type=Path)
    parser.add_argument("--seed", type=int, default=42, choices=[42, 52, 62])
    parser.add_argument("--model")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); directories(args.output_root)
    if args.drive_run is not None: args.drive_run.mkdir(parents=True, exist_ok=True)
    result = {}
    if args.phase in {"contracts", "all"}: result["contracts"] = phase_contracts(args.output_root)
    if args.phase in {"smoke", "all"}: result["smoke"] = phase_smoke(args.output_root, args.exp04_root, args.drive_run)
    if args.phase in {"stage1", "all"}: result["stage1"] = phase_stage1(args.output_root, args.seed, args.model, args.exp04_root, args.drive_run)
    if args.phase in {"stage2", "all"}: result["stage2"] = phase_stage2(args.output_root, args.seed, args.model, args.exp04_root, args.drive_run)
    if args.phase in {"finalize", "all"}: result["finalize"] = phase_finalize(args.output_root)
    if args.phase in {"report", "all"}:
        render_figures(args.output_root)
        write_manifest(args.output_root, args.run_id, 123, None if args.drive_run is None else str(args.drive_run))
        write_report(args.output_root, EXPERIMENT_DIR / "report.md"); write_report(args.output_root)
    write_json(args.output_root / f"phase_{args.phase}.json", result)
    if args.drive_run is not None:
        _sync([args.output_root / f"phase_{args.phase}.json"], args.drive_run, Path("phases"))


if __name__ == "__main__":
    main()
