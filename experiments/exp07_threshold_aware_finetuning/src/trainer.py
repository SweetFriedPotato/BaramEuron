"""AMP fine-tuners that select checkpoints exclusively on inner validation."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.exp02_daily_tcn_scada_aux.src.trainer import predict, seed_everything
from experiments.exp03_official_score_calibration.src.train_variants import official_validation_score
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import (
    RawGridDataset,
    RawModelInputs,
    predict_raw,
)

from .freeze_policy import FreezeManifest, apply_freeze_policy, optimizer_groups
from .nested_finetune import select_inner_checkpoint
from .threshold_loss import scheduled_tau, threshold_aware_loss


class TCNFineTuneDataset(Dataset):
    input_names = ("forecast_features",)

    def __init__(self, x: np.ndarray, target: np.ndarray, label_mask: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        mask = np.asarray(label_mask, dtype=bool)
        values = np.asarray(target, dtype=np.float32).copy()
        values[~mask] = 0.0
        self.target = torch.from_numpy(values)
        self.label_mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.target[index], self.label_mask[index]


@dataclass
class FineTuneResult:
    outer_prediction_cf: np.ndarray
    inner_prediction_cf: np.ndarray
    history: list[dict]
    best_epoch: int
    best_total_score: float
    best_one_minus_nmae: float
    best_ficr: float
    best_parameter_distance: float
    checkpoint_path: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    freeze_manifest: dict
    training_seconds: float
    device: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_distance(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> float:
    squared, count = 0.0, 0
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name not in initial or not value.is_floating_point():
                continue
            difference = value.detach().cpu().float() - initial[name].detach().cpu().float()
            squared += float(difference.square().sum())
            count += difference.numel()
    return float(np.sqrt(squared / max(count, 1)))


def _optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    policy = str(config.get("freeze_policy", "head_only"))
    if policy == "head_only":
        groups = optimizer_groups(
            model, policy,
            head_learning_rate=float(training.get("learning_rate", training.get("head_learning_rate", 1e-4))),
        )
    else:
        groups = optimizer_groups(
            model, policy,
            head_learning_rate=float(training.get("head_learning_rate", 5e-5)),
            block_learning_rate=float(training.get("block_learning_rate", 1e-5)),
        )
    return torch.optim.AdamW(groups, weight_decay=float(training.get("weight_decay", 1e-5)))


def _fit(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    forward_train: Callable,
    predict_inner: Callable[[], np.ndarray],
    predict_outer: Callable[[], np.ndarray],
    inner_y: np.ndarray,
    inner_mask: np.ndarray,
    config: dict,
    seed: int,
    source_checkpoint: Path,
    checkpoint_path: Path,
    require_cuda: bool,
) -> FineTuneResult:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Exp07 neural fine-tuning requires the configured A100 CUDA runtime")
    model.to(device)
    policy = str(config.get("freeze_policy", "head_only"))
    manifest: FreezeManifest = apply_freeze_policy(model, policy)
    optimizer = _optimizer(model, config)
    training = config["training"]
    max_epochs = int(training.get("max_epochs", 20))
    patience = int(training.get("patience", 5))
    clip = float(training.get("gradient_clip", 0.5))
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    initial = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    best_record: dict | None = None
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        tau = scheduled_tau(
            epoch, max_epochs, float(config.get("tau_start", 0.006)),
            float(config.get("tau_end", config.get("tau_start", 0.006))),
            str(config.get("tau_schedule", "fixed")),
        )
        model.train(); totals, bases, softs, boundaries = [], [], [], []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power, target, mask = forward_train(batch, device)
                parts = threshold_aware_loss(
                    power, target, mask, tau=tau,
                    sigma=float(config.get("sigma", 0.006)),
                    soft_ficr_weight=float(config.get("soft_ficr_weight", 0.20)),
                    lambda_boundary=float(config.get("lambda_boundary", 0.05)),
                )
            scaler.scale(parts.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad], clip
            )
            scaler.step(optimizer); scaler.update()
            totals.append(float(parts.total.detach().cpu()))
            bases.append(float(parts.base_nmae.detach().cpu()))
            softs.append(float(parts.soft_ficr.detach().cpu()))
            boundaries.append(float(parts.boundary.detach().cpu()))
        inner_prediction = predict_inner()
        score, one_minus_nmae, ficr, _ = official_validation_score(
            inner_prediction, inner_y, inner_mask
        )
        distance = parameter_distance(model, initial)
        record = {
            "epoch": epoch,
            "tau": tau,
            "train_total_loss": float(np.mean(totals)),
            "train_base_nmae_loss": float(np.mean(bases)),
            "train_soft_ficr_loss": float(np.mean(softs)),
            "train_boundary_loss": float(np.mean(boundaries)),
            "total_score": score,
            "one_minus_nmae": one_minus_nmae,
            "ficr": ficr,
            "parameter_distance": distance,
        }
        history.append(record)
        selected = select_inner_checkpoint(
            [value for value in (best_record, record) if value is not None]
        )
        if selected is record:
            best_record = copy.deepcopy(record)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} tau={tau:.5f} inner={score:.6f} "
            f"ficr={ficr:.6f} distance={distance:.8f}", flush=True,
        )
        if stale >= patience:
            break
    if best_record is None or best_state is None:
        raise RuntimeError("fine-tuning did not produce an inner-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    inner_prediction = predict_inner()
    outer_prediction = predict_outer()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "config": config,
        "seed": seed,
        "best_epoch": best_record["epoch"],
        "best_total_score": best_record["total_score"],
        "best_one_minus_nmae": best_record["one_minus_nmae"],
        "best_ficr": best_record["ficr"],
        "best_parameter_distance": best_record["parameter_distance"],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "freeze_manifest": manifest.to_dict(),
    }
    torch.save(payload, checkpoint_path)
    checkpoint_path.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return FineTuneResult(
        outer_prediction_cf=outer_prediction,
        inner_prediction_cf=inner_prediction,
        history=history,
        best_epoch=int(best_record["epoch"]),
        best_total_score=float(best_record["total_score"]),
        best_one_minus_nmae=float(best_record["one_minus_nmae"]),
        best_ficr=float(best_record["ficr"]),
        best_parameter_distance=float(best_record["parameter_distance"]),
        checkpoint_path=str(checkpoint_path),
        source_checkpoint=str(source_checkpoint),
        source_checkpoint_sha256=_sha256(source_checkpoint),
        freeze_manifest=manifest.to_dict(),
        training_seconds=time.perf_counter() - started,
        device=str(device),
    )


def finetune_tcn(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    inner_mask: np.ndarray,
    outer_x: np.ndarray,
    config: dict,
    seed: int,
    source_checkpoint: Path,
    checkpoint_path: Path,
    *,
    require_cuda: bool = True,
) -> FineTuneResult:
    batch_size = int(config["training"].get("batch_size", 32))
    loader = DataLoader(
        TCNFineTuneDataset(train_x, train_y, train_mask), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    def forward_train(batch, device):
        x, target, mask = (value.to(device, non_blocking=True) for value in batch)
        power, _ = model(x)
        return power, target, mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _fit(
        model=model, loader=loader, forward_train=forward_train,
        predict_inner=lambda: predict(model, inner_x, batch_size, device),
        predict_outer=lambda: predict(model, outer_x, batch_size, device),
        inner_y=inner_y, inner_mask=inner_mask, config=config, seed=seed,
        source_checkpoint=Path(source_checkpoint), checkpoint_path=Path(checkpoint_path),
        require_cuda=require_cuda,
    )


def finetune_raw(
    model: torch.nn.Module,
    train_inputs: RawModelInputs,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    inner_inputs: RawModelInputs,
    inner_y: np.ndarray,
    inner_mask: np.ndarray,
    outer_inputs: RawModelInputs,
    config: dict,
    seed: int,
    source_checkpoint: Path,
    checkpoint_path: Path,
    *,
    require_cuda: bool = True,
) -> FineTuneResult:
    batch_size = int(config["training"].get("batch_size", 16))
    loader = DataLoader(
        RawGridDataset(train_inputs, train_y, train_mask), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    def forward_train(batch, device):
        values = [value.to(device, non_blocking=True) for value in batch]
        ldaps, gfs, common, group, target, mask = values[:6]
        power, _, _ = model(ldaps, gfs, common, group)
        return power, target, mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _fit(
        model=model, loader=loader, forward_train=forward_train,
        predict_inner=lambda: predict_raw(model, inner_inputs, batch_size, device)[0],
        predict_outer=lambda: predict_raw(model, outer_inputs, batch_size, device)[0],
        inner_y=inner_y, inner_mask=inner_mask, config=config, seed=seed,
        source_checkpoint=Path(source_checkpoint), checkpoint_path=Path(checkpoint_path),
        require_cuda=require_cuda,
    )

