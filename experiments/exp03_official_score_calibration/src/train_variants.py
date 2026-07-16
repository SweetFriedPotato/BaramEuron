"""A100 trainer selecting checkpoints by official Score instead of macro nMAE."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset

from experiments.exp02_daily_tcn_scada_aux.src.trainer import predict, seed_everything
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import (
    SelectedFeatureUnionBuilder,
    baseline_config,
    issue_mapping,
    raw_artifacts,
)
from experiments.exp02_daily_tcn_scada_aux.src.models import build_model
from experiments.exp02_daily_tcn_scada_aux.src.preprocessing import NeuralFoldPreprocessor
from experiments.exp02_daily_tcn_scada_aux.src.run_experiment import (
    build_full_data,
    prepare_fold,
    prediction_frame,
    seed_ensemble,
)
from experiments.exp02_daily_tcn_scada_aux.src.scada_targets import build_scada_aux_targets
from experiments.exp02_daily_tcn_scada_aux.src.scada_targets import AuxiliaryTargetScaler
from experiments.exp02_daily_tcn_scada_aux.src.sequence_builder import build_sequences

from .ficr_surrogate import total_official_loss
from .backtest import ROLLING_QUARTERS, expanding_quarter_window


def official_validation_score(
    prediction_cf: np.ndarray, target_cf: np.ndarray, label_mask: np.ndarray
) -> tuple[float, float, float, list[dict]]:
    prediction = np.maximum(np.asarray(prediction_cf, dtype=float), 0.0)
    target = np.asarray(target_cf, dtype=float)
    mask = np.asarray(label_mask, dtype=bool) & (target >= 0.10)
    groups = []
    for group in range(target.shape[-1]):
        valid = mask[..., group]
        if not valid.any():
            continue
        error = np.abs(prediction[..., group][valid] - target[..., group][valid])
        actual = target[..., group][valid]
        unit_price = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
        nmae = float(error.mean())
        ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
        groups.append({"group_id": group + 1, "nmae": nmae, "ficr": ficr,
                       "evaluated_samples": int(valid.sum())})
    if not groups:
        raise ValueError("validation contains no officially evaluated labels")
    one_minus_nmae = float(1.0 - np.mean([group["nmae"] for group in groups]))
    ficr = float(np.mean([group["ficr"] for group in groups]))
    return 0.5 * one_minus_nmae + 0.5 * ficr, one_minus_nmae, ficr, groups


def is_better_official_score(candidate: float, best: float, tolerance: float = 1e-12) -> bool:
    return candidate > best + tolerance


def temporal_sample_weights(
    timestamps: np.ndarray,
    *,
    winter_weight: float = 1.0,
    year_weights: dict[int, float] | None = None,
) -> np.ndarray:
    flat = np.asarray(timestamps).astype("datetime64[h]")
    months = (flat.astype("datetime64[M]").astype(int) % 12) + 1
    years = flat.astype("datetime64[Y]").astype(int) + 1970
    weights = np.ones(flat.shape, dtype=np.float32)
    if winter_weight != 1.0:
        weights[np.isin(months, [12, 1, 2])] *= float(winter_weight)
    if year_weights:
        for year, weight in year_weights.items():
            weights[years == int(year)] *= float(weight)
    return np.repeat(weights[..., None], 3, axis=-1)


class OfficialDataset(Dataset):
    def __init__(self, x, y, mask, aux, aux_mask, sample_weight):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        mask_array = np.asarray(mask, dtype=bool)
        y_array = np.asarray(y, dtype=np.float32).copy(); y_array[~mask_array] = 0.0
        self.y = torch.from_numpy(y_array); self.mask = torch.from_numpy(mask_array)
        if aux is None:
            self.aux = torch.zeros_like(self.y); self.aux_mask = torch.zeros_like(self.mask)
        else:
            aux_mask_array = np.asarray(aux_mask, dtype=bool)
            aux_array = np.asarray(aux, dtype=np.float32).copy(); aux_array[~aux_mask_array] = 0.0
            self.aux = torch.from_numpy(aux_array); self.aux_mask = torch.from_numpy(aux_mask_array)
        self.sample_weight = torch.as_tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.mask[index], self.aux[index],
                self.aux_mask[index], self.sample_weight[index])


@dataclass
class OfficialTrainingResult:
    prediction_cf: np.ndarray
    history: list[dict]
    best_epoch: int
    best_total_score: float
    best_one_minus_nmae: float
    best_ficr: float
    group_metrics: list[dict]
    checkpoint_path: Path
    training_seconds: float
    device: str


def train_official_model(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    train_timestamps: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
    seed: int,
    checkpoint_path: Path,
    train_aux: np.ndarray | None = None,
    train_aux_mask: np.ndarray | None = None,
    max_epochs_override: int | None = None,
) -> OfficialTrainingResult:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = config["training"]
    batch_size = int(training.get("batch_size", 32))
    max_epochs = int(max_epochs_override or training.get("max_epochs", 100))
    patience = int(training.get("patience", 12))
    sample_weight = temporal_sample_weights(
        train_timestamps,
        winter_weight=float(config.get("winter_weight", 1.0)),
        year_weights={int(k): float(v) for k, v in config.get("year_weights", {}).items()},
    )
    dataset = OfficialDataset(train_x, train_y, train_mask, train_aux, train_aux_mask, sample_weight)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
                                  weight_decay=float(training.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history, best_state, best_groups = [], None, []
    best_score, best_nmae, best_ficr, best_epoch, stale = -np.inf, np.nan, np.nan, 0, 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train(); totals, mains, ficrs, auxiliaries = [], [], [], []
        for bx, by, bm, ba, bam, bw in loader:
            bx, by, bm, ba, bam, bw = [value.to(device) for value in (bx, by, bm, ba, bam, bw)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power, auxiliary = model(bx)
                total, main, ficr_loss, aux_loss = total_official_loss(
                    power, by, bm, auxiliary, ba, bam,
                    aux_weight=float(config.get("aux_weight", 0.05)),
                    lambda_ficr=float(config.get("lambda_ficr", 0.0)),
                    temperature=float(config.get("temperature", 0.005)),
                    sample_weight=bw,
                )
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer); scaler.update()
            totals.append(float(total.detach().cpu())); mains.append(float(main.detach().cpu()))
            ficrs.append(float(ficr_loss.detach().cpu())); auxiliaries.append(float(aux_loss.detach().cpu()))
        validation = predict(model, valid_x, batch_size, device)
        score, one_minus_nmae, ficr, groups = official_validation_score(validation, valid_y, valid_mask)
        scheduler.step(score)
        row = {"epoch": epoch, "train_total_loss": np.mean(totals), "train_power_loss": np.mean(mains),
               "train_ficr_loss": np.mean(ficrs), "train_aux_loss": np.mean(auxiliaries),
               "valid_total_score": score, "valid_one_minus_nmae": one_minus_nmae,
               "valid_ficr": ficr, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append({key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in row.items()})
        print(f"epoch={epoch:03d} loss={row['train_total_loss']:.6f} score={score:.6f} "
              f"1-nmae={one_minus_nmae:.6f} ficr={ficr:.6f}", flush=True)
        if is_better_official_score(score, best_score):
            best_score, best_nmae, best_ficr, best_epoch = score, one_minus_nmae, ficr, epoch
            best_groups, best_state, stale = groups, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("official trainer did not produce a checkpoint")
    model.load_state_dict(best_state); checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": config, "seed": seed, "best_epoch": best_epoch,
                "best_total_score": best_score, "best_one_minus_nmae": best_nmae, "best_ficr": best_ficr,
                "feature_dim": int(train_x.shape[-1])}, checkpoint_path)
    checkpoint_path.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final = predict(model, valid_x, batch_size, device)
    return OfficialTrainingResult(final, history, best_epoch, best_score, best_nmae, best_ficr,
                                  best_groups, checkpoint_path, time.perf_counter() - started, str(device))


def train_official_fixed_epochs(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray,
    config: dict,
    seed: int,
    epochs: int,
    checkpoint_path: Path,
    aux: np.ndarray | None = None,
    aux_mask: np.ndarray | None = None,
) -> tuple[torch.nn.Module, list[dict], str]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); training = config["training"]
    sample_weight = temporal_sample_weights(
        timestamps, winter_weight=float(config.get("winter_weight", 1.0)),
        year_weights={int(k): float(v) for k, v in config.get("year_weights", {}).items()},
    )
    dataset = OfficialDataset(x, y, mask, aux, aux_mask, sample_weight)
    loader = DataLoader(dataset, batch_size=int(training.get("batch_size", 32)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
                                  weight_decay=float(training.get("weight_decay", 1e-4)))
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled); history = []
    for epoch in range(1, int(epochs) + 1):
        model.train(); totals, mains, ficrs = [], [], []
        for bx, by, bm, ba, bam, bw in loader:
            bx, by, bm, ba, bam, bw = [value.to(device) for value in (bx, by, bm, ba, bam, bw)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power, auxiliary = model(bx)
                total, main, ficr_loss, _ = total_official_loss(
                    power, by, bm, auxiliary, ba, bam,
                    aux_weight=float(config.get("aux_weight", 0.05)),
                    lambda_ficr=float(config.get("lambda_ficr", 0.0)),
                    temperature=float(config.get("temperature", 0.005)), sample_weight=bw,
                )
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer); scaler.update()
            totals.append(float(total.detach().cpu())); mains.append(float(main.detach().cpu()))
            ficrs.append(float(ficr_loss.detach().cpu()))
        row = {"epoch": epoch, "train_total_loss": float(np.mean(totals)),
               "train_power_loss": float(np.mean(mains)), "train_ficr_loss": float(np.mean(ficrs))}
        history.append(row); print(f"full epoch={epoch:03d} loss={row['train_total_loss']:.6f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed, "epochs": epochs,
                "feature_dim": int(x.shape[-1])}, checkpoint_path)
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history, str(device)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp03_official_score_calibration"


def _load_base_config() -> dict:
    path = PROJECT_ROOT / "experiments/exp02_daily_tcn_scada_aux/configs/tcn_aux_005.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def variant_config(variant_id: str, lambda_ficr: float = 0.0) -> dict:
    config = copy.deepcopy(_load_base_config())
    config["experiment_id"] = variant_id
    config["lambda_ficr"] = float(lambda_ficr)
    config["temperature"] = 0.005
    config["checkpoint_metric"] = "official_total_score"
    if variant_id.endswith("winter"):
        config["winter_weight"] = 1.15
    if variant_id.endswith("recency"):
        config["year_weights"] = {2022: 0.70, 2023: 0.85, 2024: 1.0}
    return config


def _score_prediction_frame(frame: pd.DataFrame) -> dict:
    frame = frame.reset_index(drop=True)
    capacity = frame["target"].map(
        {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
    ).to_numpy()
    error = np.abs(frame["y_pred_kwh"].to_numpy() - frame["y_true_kwh"].to_numpy()) / capacity
    actual_cf = frame["y_true_kwh"].to_numpy() / capacity
    rows = []
    for target, indices in frame.groupby("target").groups.items():
        idx = np.asarray(list(indices)); valid = actual_cf[idx] >= 0.10
        if not valid.any():
            continue
        values = error[idx][valid]; actual = frame.iloc[idx]["y_true_kwh"].to_numpy()[valid]
        reward = np.select([values <= 0.06, values <= 0.08], [4.0, 3.0], default=0.0)
        rows.append((values.mean(), np.sum(actual * reward) / np.sum(actual * 4.0)))
    one_minus_nmae = 1.0 - float(np.mean([row[0] for row in rows])); ficr = float(np.mean([row[1] for row in rows]))
    return {"total_score": 0.5 * (one_minus_nmae + ficr), "one_minus_nmae": one_minus_nmae,
            "ficr": ficr, "groups_available": len(rows)}


def _sync_result(drive_run: Path | None, variant_id: str, seed: int, fold: str,
                 checkpoint: Path, predictions: pd.DataFrame, metrics: dict) -> None:
    if drive_run is None:
        return
    destination = drive_run / "seeds" / variant_id / str(seed) / fold
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination / checkpoint.name)
    history = checkpoint.with_suffix(".history.json")
    if history.exists():
        shutil.copy2(history, destination / history.name)
    predictions.to_csv(destination / "predictions.csv", index=False)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def run_training_variants(
    output_root: Path,
    drive_run: Path | None = None,
    smoke_only: bool = False,
    weighting_only: bool = False,
) -> dict:
    for name in ("checks", "metrics", "predictions", "checkpoints"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    baseline_cfg = baseline_config()
    train_features, test_features, labels = raw_artifacts(baseline_cfg)
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"]))
    states: dict = {}
    prepared = {
        fold: prepare_fold(fold, baseline_cfg, train_features, test_features, labels, scada,
                           output_root / "checks", states)
        for fold in ("fold_a", "fold_b")
    }
    smoke_configs = [variant_config("official_mask", 0.0)] + [
        variant_config(f"ficr_lambda_{str(value).replace('.', '')}", value)
        for value in (0.05, 0.10, 0.20, 0.30)
    ]
    score_path = output_root / "metrics/training_variant_scores.csv"
    prediction_path = output_root / "predictions/ficr_aware_predictions.csv"
    if weighting_only:
        if not score_path.exists() or not prediction_path.exists():
            raise FileNotFoundError("weighting-only requires completed smoke/full metrics and predictions")
        existing_scores = pd.read_csv(score_path)
        existing_predictions = pd.read_csv(prediction_path)
        rows = existing_scores.to_dict("records")
        predictions = [existing_predictions]
    else:
        rows, predictions = [], []

    def run_one(config: dict, fold: str, seed: int, epochs: int | None, stage: str) -> None:
        data = prepared[fold]; seed_everything(seed)
        model = build_model(config, data.train_x.shape[-1])
        checkpoint = output_root / "checkpoints" / f"{config['experiment_id']}_{fold}_seed_{seed}.pt"
        print(f"\n[{stage}] {config['experiment_id']} {fold} seed={seed}", flush=True)
        result = train_official_model(
            model, data.train_x, data.train.y_cf, data.train.label_mask, data.train.timestamps,
            data.valid_x, data.valid.y_cf, data.valid.label_mask, config, seed, checkpoint,
            data.train_aux, data.train_aux_mask, max_epochs_override=epochs,
        )
        frame = prediction_frame(data, result.prediction_cf, config["experiment_id"], seed)
        metric = {"stage": stage, "experiment_id": config["experiment_id"], "fold": fold, "seed": seed,
                  "lambda_ficr": config.get("lambda_ficr", 0.0), "winter_weight": config.get("winter_weight", 1.0),
                  "year_weights": repr(config.get("year_weights", {})), "best_epoch": result.best_epoch,
                  "total_score": result.best_total_score, "one_minus_nmae": result.best_one_minus_nmae,
                  "ficr": result.best_ficr, "training_seconds": result.training_seconds, "device": result.device}
        rows.append(metric); frame["stage"] = stage; predictions.append(frame)
        _sync_result(drive_run, config["experiment_id"], seed, fold, checkpoint, frame, metric)
        pd.DataFrame(rows).to_csv(output_root / "metrics/training_variant_scores.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(output_root / "predictions/ficr_aware_predictions.csv", index=False)

    if not weighting_only:
        for config in smoke_configs:
            run_one(config, "fold_b", 42, 5, "smoke")
    smoke_rows = pd.DataFrame(rows)
    ficr_smoke = smoke_rows.loc[smoke_rows["experiment_id"].str.startswith("ficr_lambda")]
    best_ficr_id = str(ficr_smoke.sort_values("total_score", ascending=False).iloc[0]["experiment_id"])
    best_lambda = float(ficr_smoke.loc[ficr_smoke["experiment_id"].eq(best_ficr_id), "lambda_ficr"].iloc[0])
    if smoke_only:
        return {"best_ficr_smoke": best_ficr_id, "best_lambda": best_lambda, "rows": len(rows)}

    if not weighting_only:
        full_configs = [variant_config("official_mask", 0.0), variant_config(best_ficr_id, best_lambda)]
        for config in full_configs:
            for seed in (42, 52, 62):
                for fold in ("fold_a", "fold_b"):
                    run_one(config, fold, seed, None, "full")

    all_predictions = pd.concat(predictions, ignore_index=True)
    full_frames = all_predictions.loc[all_predictions["stage"].eq("full")].copy()
    ensembles = seed_ensemble(full_frames.drop(columns="stage"))
    ensemble_rows = []
    for (experiment_id, fold), part in ensembles.groupby(["experiment_id", "fold"]):
        ensemble_rows.append({"experiment_id": experiment_id, "fold": fold, **_score_prediction_frame(part)})
    ensemble_table = pd.DataFrame(ensemble_rows)
    ensemble_table.to_csv(output_root / "metrics/training_variant_ensemble_scores.csv", index=False)
    selected = str(
        ensemble_table.loc[ensemble_table["fold"].eq("fold_b")]
        .sort_values("total_score", ascending=False).iloc[0]["experiment_id"]
    )
    selected_lambda = best_lambda if selected == best_ficr_id else 0.0
    for suffix in ("winter", "recency"):
        config = variant_config(f"{selected}_{suffix}", selected_lambda)
        for fold in ("fold_a", "fold_b"):
            run_one(config, fold, 42, None, "weighting")

    summary = {"best_ficr_smoke": best_ficr_id, "best_lambda": best_lambda,
               "selected_full_variant": selected, "selected_lambda": selected_lambda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_full_test_training(output_root: Path, drive_run: Path | None = None) -> dict:
    for name in ("checks", "metrics", "predictions", "checkpoints"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    score_path = output_root / "metrics/training_variant_scores.csv"
    if not score_path.exists():
        raise FileNotFoundError("full-test-only requires validation training_variant_scores.csv")
    scores = pd.read_csv(score_path)
    selected = scores.loc[
        scores["stage"].eq("full") & scores["experiment_id"].eq("ficr_lambda_02")
        & scores["fold"].eq("fold_b")
    ]
    epochs = int(np.median(selected["best_epoch"].to_numpy(dtype=int)))
    config = variant_config("ficr_lambda_02", 0.20)
    baseline_cfg = baseline_config(); train_features, test_features, labels = raw_artifacts(baseline_cfg)
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"]))
    train_bundle, test_bundle, train_x, test_x, aux, aux_mask, _, _ = build_full_data(
        baseline_cfg, train_features, test_features, labels, scada, output_root
    )
    seed_predictions, devices = [], []
    for seed in (42, 52, 62):
        seed_everything(seed); model = build_model(config, train_x.shape[-1])
        checkpoint = output_root / "checkpoints" / f"ficr_lambda_02_full_seed_{seed}.pt"
        model, history, device = train_official_fixed_epochs(
            model, train_x, train_bundle.y_cf, train_bundle.label_mask, train_bundle.timestamps,
            config, seed, epochs, checkpoint, aux, aux_mask,
        )
        prediction = predict(model, test_x, int(config["training"].get("batch_size", 32)), torch.device(device))
        seed_predictions.append(prediction); devices.append(device)
        seed_path = output_root / "predictions" / f"ficr_lambda_02_full_seed_{seed}.npz"
        np.savez_compressed(seed_path, prediction_cf=prediction, timestamps=test_bundle.timestamps)
        if drive_run is not None:
            destination = drive_run / "full" / str(seed); destination.mkdir(parents=True, exist_ok=True)
            for source in (checkpoint, checkpoint.with_suffix(".history.json"), seed_path):
                shutil.copy2(source, destination / source.name)
    capacity = np.asarray([21600.0, 21600.0, 21000.0])
    ensemble_kwh = np.maximum(np.mean(seed_predictions, axis=0).reshape(-1, 3) * capacity, 0.0)
    timestamps = test_bundle.timestamps.reshape(-1)
    path = output_root / "predictions/ficr_aware_full_ensemble_test.csv"
    pd.DataFrame(ensemble_kwh, columns=["kpx_group_1", "kpx_group_2", "kpx_group_3"]).assign(
        forecast_kst_dtm=timestamps
    ).to_csv(path, index=False)
    if drive_run is not None:
        shutil.copy2(path, drive_run / path.name)
    summary = {"variant": "ficr_lambda_02", "lambda_ficr": 0.20, "epochs": epochs,
               "seeds": [42, 52, 62], "devices": devices, "test_rows": len(timestamps),
               "test_issue_blocks": len(test_bundle.x), "prediction_path": str(path)}
    (output_root / "full_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if drive_run is not None:
        shutil.copy2(output_root / "full_training_summary.json", drive_run / "full_training_summary.json")
    return summary


def _prepare_expanding_quarter(
    quarter: str,
    baseline_cfg: dict,
    train_features: pd.DataFrame,
    labels: pd.DataFrame,
    scada: pd.DataFrame,
):
    window = expanding_quarter_window(quarter)
    times = pd.to_datetime(train_features["forecast_kst_dtm"])
    fit_mask = ((times >= window["train_start"]) & (times <= window["train_end"])).to_numpy()
    builder = SelectedFeatureUnionBuilder(baseline_cfg)
    selected = builder.fit_transform(train_features, fit_mask)
    bundle, incomplete = build_sequences(selected, issue_mapping(baseline_cfg, "train"), labels, scada)
    if not incomplete.empty:
        raise ValueError(f"incomplete issue blocks in {quarter}: {len(incomplete)}")
    first, last = bundle.timestamps[:, 0], bundle.timestamps[:, -1]
    train_idx = np.flatnonzero(
        (first >= np.datetime64(window["train_start"])) & (last <= np.datetime64(window["train_end"]))
    )
    valid_idx = np.flatnonzero(
        (first >= np.datetime64(window["valid_start"])) & (last <= np.datetime64(window["valid_end"]))
    )
    train, valid = bundle.subset(train_idx), bundle.subset(valid_idx)
    # The requested contract evaluates groups 1/2 throughout 2023. Group 3 has
    # no 2022 labels, so its first valid expanding split is train-through-2023Q4
    # -> 2024Q1. Do not let an untrained group-3 head affect 2023 checkpoints.
    if quarter.startswith("2023"):
        train.label_mask[:, :, 2] = False
        valid.label_mask[:, :, 2] = False
        if train.aux_mask is not None:
            train.aux_mask[:, :, 2] = False
        if valid.aux_mask is not None:
            valid.aux_mask[:, :, 2] = False
    preprocessor = NeuralFoldPreprocessor(); train_x = preprocessor.fit_transform(train.x)
    valid_x = preprocessor.transform(valid.x)
    aux_scaler = AuxiliaryTargetScaler().fit(train.aux_wind, train.aux_mask)
    train_aux, train_aux_mask = aux_scaler.transform(train.aux_wind, train.aux_mask)
    wind_index = train.feature_names.index("gfs__ws100__mean")
    high_wind_threshold = float(np.nanquantile(train.x[:, :, wind_index], 0.90))
    return SimpleNamespace(
        fold=quarter, train=train, valid=valid, train_x=train_x, valid_x=valid_x,
        train_aux=train_aux, train_aux_mask=train_aux_mask, high_wind_threshold=high_wind_threshold,
    )


def run_rolling_retraining(output_root: Path, drive_run: Path | None = None) -> dict:
    for name in ("checks", "metrics", "predictions", "checkpoints"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    baseline_cfg = baseline_config(); train_features, _, labels = raw_artifacts(baseline_cfg)
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"]))
    rows, prediction_parts = [], []
    configs = [variant_config("official_mask", 0.0), variant_config("ficr_lambda_02", 0.20)]
    for quarter in ROLLING_QUARTERS:
        data = _prepare_expanding_quarter(quarter, baseline_cfg, train_features, labels, scada)
        window = expanding_quarter_window(quarter)
        for config in configs:
            seed = 42; seed_everything(seed); model = build_model(config, data.train_x.shape[-1])
            checkpoint = output_root / "checkpoints" / f"rolling_{config['experiment_id']}_{quarter}_seed_{seed}.pt"
            result = train_official_model(
                model, data.train_x, data.train.y_cf, data.train.label_mask, data.train.timestamps,
                data.valid_x, data.valid.y_cf, data.valid.label_mask, config, seed, checkpoint,
                data.train_aux, data.train_aux_mask,
            )
            frame = prediction_frame(data, result.prediction_cf, config["experiment_id"], seed)
            frame["quarter"] = quarter; prediction_parts.append(frame)
            row = {"experiment_id": config["experiment_id"], "quarter": quarter, "seed": seed,
                   "train_end": window["train_end"], "valid_start": window["valid_start"],
                   "valid_end": window["valid_end"], "train_issue_blocks": len(data.train.x),
                   "valid_issue_blocks": len(data.valid.x), "best_epoch": result.best_epoch,
                   "total_score": result.best_total_score, "one_minus_nmae": result.best_one_minus_nmae,
                   "ficr": result.best_ficr, "groups_available": len(result.group_metrics),
                   "training_seconds": result.training_seconds, "device": result.device}
            rows.append(row)
            _sync_result(drive_run, f"rolling_{config['experiment_id']}", seed, quarter,
                         checkpoint, frame, row)
            pd.DataFrame(rows).to_csv(output_root / "metrics/rolling_retrained_scores.csv", index=False)
            pd.concat(prediction_parts, ignore_index=True).to_csv(
                output_root / "predictions/rolling_retrained_predictions.csv", index=False
            )
    result = pd.DataFrame(rows)
    summary = {
        "quarters": ROLLING_QUARTERS,
        "models": [config["experiment_id"] for config in configs],
        "ficr_improved_quarters": int(
            (result.pivot(index="quarter", columns="experiment_id", values="total_score")["ficr_lambda_02"]
             > result.pivot(index="quarter", columns="experiment_id", values="total_score")["official_mask"]).sum()
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_root / "rolling_retraining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if drive_run is not None:
        for source in (output_root / "metrics/rolling_retrained_scores.csv",
                       output_root / "predictions/rolling_retrained_predictions.csv",
                       output_root / "rolling_retraining_summary.json"):
            shutil.copy2(source, drive_run / source.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--drive-run", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--weighting-only", action="store_true")
    parser.add_argument("--full-test-only", action="store_true")
    parser.add_argument("--rolling-backtest-only", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (
        args.smoke_only, args.weighting_only, args.full_test_only, args.rolling_backtest_only
    )) > 1:
        parser.error("phase-only flags are mutually exclusive")
    if args.full_test_only:
        print(json.dumps(run_full_test_training(args.output_root.resolve(), args.drive_run), indent=2))
        return
    if args.rolling_backtest_only:
        print(json.dumps(run_rolling_retraining(args.output_root.resolve(), args.drive_run), indent=2))
        return
    print(json.dumps(run_training_variants(
        args.output_root.resolve(), args.drive_run, args.smoke_only, args.weighting_only
    ), indent=2))


if __name__ == "__main__":
    main()
