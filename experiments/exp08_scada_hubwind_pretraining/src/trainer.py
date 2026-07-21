"""A100-ready Stage-1/Stage-2 trainers with official Exp03 loss reuse."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.exp02_daily_tcn_scada_aux.src.trainer import seed_everything
from experiments.exp03_official_score_calibration.src.ficr_surrogate import total_official_loss
from experiments.exp03_official_score_calibration.src.train_variants import official_validation_score
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import RawModelInputs

from .scada_hourly_targets import HubWindTargetScaler
from .stage1_dataset import Stage1Dataset
from .stage1_model import group_balanced_distribution_loss
from .stage2_dataset import Stage2Dataset
from .transfer import apply_transfer_policy, optimizer_groups


@dataclass
class Stage1TrainingResult:
    prediction_mps: np.ndarray
    best_epoch: int
    best_mae: float
    history: list[dict]
    checkpoint_path: Path
    device: str
    training_seconds: float


@dataclass
class Stage2TrainingResult:
    prediction_cf: np.ndarray
    best_epoch: int
    best_score: float
    best_one_minus_nmae: float
    best_ficr: float
    history: list[dict]
    checkpoint_path: Path
    device: str
    training_seconds: float


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move(batch, device: torch.device):
    return [value.to(device, non_blocking=True) for value in batch]


def predict_stage1(
    model: torch.nn.Module,
    inputs: RawModelInputs,
    scaler: HubWindTargetScaler,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(Stage1Dataset(inputs), batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ldaps, gfs, common, group = _move(batch[:4], device)
            prediction, _, _ = model(ldaps, gfs, common, group)
            outputs.append(prediction.detach().cpu().numpy())
    scaled = np.concatenate(outputs)
    if scaled.shape[-1] < 4:
        padded = np.zeros((*scaled.shape[:-1], 4), dtype=np.float32)
        padded[..., :scaled.shape[-1]] = scaled
        scaled = padded
    return scaler.inverse_transform(scaled)


def _stage1_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, active: int) -> float:
    group_losses = []
    for group in range(3):
        valid = mask[..., group, :active]
        if valid.any():
            group_losses.append(float(np.abs(prediction[..., group, :active] - target[..., group, :active])[valid].mean()))
    return float(np.mean(group_losses)) if group_losses else np.inf


def train_stage1(
    model: torch.nn.Module,
    train_inputs: RawModelInputs,
    train_target_scaled: np.ndarray,
    train_mask: np.ndarray,
    valid_inputs: RawModelInputs,
    valid_target_raw: np.ndarray,
    valid_mask: np.ndarray,
    scaler: HubWindTargetScaler,
    config: dict,
    seed: int,
    checkpoint_path: Path,
    *,
    max_epochs_override: int | None = None,
) -> Stage1TrainingResult:
    seed_everything(seed)
    device = _device()
    model.to(device)
    training = config["training"]
    batch_size = int(training.get("batch_size", 16))
    max_epochs = int(max_epochs_override or training.get("max_epochs", 80))
    patience = int(training.get("patience", 10))
    dataset = Stage1Dataset(train_inputs, train_target_scaled[..., :model.target_count], train_mask[..., :model.target_count])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
                                  weight_decay=float(training.get("weight_decay", 1e-4)))
    amp = bool(training.get("amp", True) and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best_state, best_mae, best_epoch, stale = None, np.inf, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []
        for batch in loader:
            ldaps, gfs, common, group, target, mask = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                prediction, _, _ = model(ldaps, gfs, common, group)
                loss = group_balanced_distribution_loss(prediction, target, mask)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            grad_scaler.step(optimizer); grad_scaler.update(); losses.append(float(loss.detach().cpu()))
        prediction_mps = predict_stage1(model, valid_inputs, scaler, batch_size, device)
        mae = _stage1_mae(prediction_mps, valid_target_raw, valid_mask, model.target_count)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_group_balanced_mae_mps": mae})
        print(f"stage1 epoch={epoch:03d} loss={np.mean(losses):.6f} mae={mae:.6f}", flush=True)
        if mae < best_mae - 1e-12:
            best_state, best_mae, best_epoch, stale = copy.deepcopy(model.state_dict()), mae, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Stage-1 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": config, "seed": seed, "best_epoch": best_epoch,
                "best_mae_mps": best_mae, "target_scaler": scaler.state_dict()}, checkpoint_path)
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    prediction_mps = predict_stage1(model, valid_inputs, scaler, batch_size, device)
    return Stage1TrainingResult(prediction_mps, best_epoch, best_mae, history, checkpoint_path,
                                str(device), time.perf_counter() - started)


def predict_stage2(
    model: torch.nn.Module,
    inputs: RawModelInputs,
    hub_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(Stage2Dataset(inputs, hub_features), batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ldaps, gfs, common, group, hub = _move(batch[:5], device)
            prediction, _, _ = model(ldaps, gfs, common, group, hub)
            outputs.append(prediction.detach().cpu().numpy())
    return np.concatenate(outputs)


def _make_stage2_optimizer(model: torch.nn.Module, variant: str, config: dict) -> torch.optim.Optimizer:
    groups = optimizer_groups(model, variant)
    return torch.optim.AdamW(groups, weight_decay=float(config["training"].get("weight_decay", 1e-4)))


def train_stage2(
    model: torch.nn.Module,
    train_inputs: RawModelInputs,
    train_hub_features: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    valid_inputs: RawModelInputs,
    valid_hub_features: np.ndarray,
    valid_target: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
    seed: int,
    checkpoint_path: Path,
    *,
    retention_target: np.ndarray | None = None,
    retention_mask: np.ndarray | None = None,
    max_epochs_override: int | None = None,
) -> Stage2TrainingResult:
    seed_everything(seed)
    device = _device(); model.to(device)
    training = config["training"]
    variant = config.get("stage2", {}).get("variant", "distribution_hubwind")
    batch_size = int(training.get("batch_size", 16))
    max_epochs = int(max_epochs_override or training.get("max_epochs", 100))
    patience = int(training.get("patience", 12))
    dataset = Stage2Dataset(train_inputs, train_hub_features, train_target, train_mask,
                            retention_target, retention_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed))
    policy = apply_transfer_policy(model, variant, epoch=0)
    optimizer = _make_stage2_optimizer(model, variant, config)
    amp = bool(training.get("amp", True) and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    best_nmae = best_ficr = np.nan
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if epoch == 6 and variant in {"explicit_hubwind", "distribution_hubwind"}:
            policy = apply_transfer_policy(model, variant, epoch=5)
            optimizer = _make_stage2_optimizer(model, variant, config)
        model.train(); totals = []
        for batch in loader:
            ldaps, gfs, common, group, hub, target, mask, retain, retain_mask = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                power, retention, _ = model(ldaps, gfs, common, group, hub)
                total, _, _, _ = total_official_loss(
                    power, target, mask, retention, retain, retain_mask,
                    aux_weight=float(config.get("retention_weight", 0.05) if retention is not None else 0.0),
                    lambda_ficr=float(config.get("lambda_ficr", 0.20)),
                    temperature=float(config.get("temperature", 0.005)),
                )
            grad_scaler.scale(total).backward(); grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            grad_scaler.step(optimizer); grad_scaler.update(); totals.append(float(total.detach().cpu()))
        prediction = predict_stage2(model, valid_inputs, valid_hub_features, batch_size, device)
        score, one_minus_nmae, ficr, _ = official_validation_score(prediction, valid_target, valid_mask)
        history.append({"epoch": epoch, "train_total_loss": float(np.mean(totals)),
                        "valid_total_score": score, "valid_one_minus_nmae": one_minus_nmae,
                        "valid_ficr": ficr, "trainable_parameters": len(policy["trainable"])})
        print(f"stage2 epoch={epoch:03d} loss={np.mean(totals):.6f} score={score:.6f}", flush=True)
        if score > best_score + 1e-12:
            best_state, best_score, best_nmae, best_ficr, best_epoch, stale = (
                copy.deepcopy(model.state_dict()), score, one_minus_nmae, ficr, epoch, 0
            )
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Stage-2 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": config, "seed": seed, "best_epoch": best_epoch,
                "best_total_score": best_score, "best_one_minus_nmae": best_nmae,
                "best_ficr": best_ficr, "transfer_policy": policy}, checkpoint_path)
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    prediction = predict_stage2(model, valid_inputs, valid_hub_features, batch_size, device)
    return Stage2TrainingResult(prediction, best_epoch, best_score, best_nmae, best_ficr,
                                history, checkpoint_path, str(device), time.perf_counter() - started)
