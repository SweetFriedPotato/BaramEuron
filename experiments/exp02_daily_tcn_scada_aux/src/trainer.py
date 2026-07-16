"""Masked multi-output neural trainer with AMP and validation nMAE selection."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import total_multitask_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        aux: np.ndarray | None = None,
        aux_mask: np.ndarray | None = None,
    ) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        mask_array = np.asarray(mask, dtype=bool)
        y_array = np.asarray(y, dtype=np.float32).copy()
        y_array[~mask_array] = 0.0
        self.y = torch.from_numpy(y_array)
        self.mask = torch.from_numpy(mask_array)
        shape = y.shape
        if aux is None:
            self.aux = torch.zeros(shape, dtype=torch.float32)
            self.aux_mask = torch.zeros(shape, dtype=torch.bool)
        else:
            aux_mask_array = np.asarray(aux_mask, dtype=bool)
            aux_array = np.asarray(aux, dtype=np.float32).copy()
            aux_array[~aux_mask_array] = 0.0
            self.aux = torch.from_numpy(aux_array)
            self.aux_mask = torch.from_numpy(aux_mask_array)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.mask[index], self.aux[index], self.aux_mask[index]


def macro_nmae_cf(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, list[float | None]]:
    prediction = np.maximum(np.asarray(prediction), 0.0)
    group_values: list[float | None] = []
    for group in range(3):
        valid = mask[:, :, group]
        group_values.append(
            float(np.mean(np.abs(prediction[:, :, group][valid] - target[:, :, group][valid])))
            if valid.any() else None
        )
    available = [value for value in group_values if value is not None]
    return float(np.mean(available)), group_values


@dataclass
class TrainingResult:
    prediction_cf: np.ndarray
    history: list[dict]
    best_epoch: int
    best_macro_nmae: float
    group_nmae: list[float | None]
    checkpoint_path: Path
    training_seconds: float
    device: str


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                power, _ = model(batch)
            parts.append(power.float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def train_model(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
    seed: int,
    checkpoint_path: Path,
    train_aux: np.ndarray | None = None,
    train_aux_mask: np.ndarray | None = None,
    valid_aux: np.ndarray | None = None,
    valid_aux_mask: np.ndarray | None = None,
    max_epochs_override: int | None = None,
) -> TrainingResult:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = config["training"]
    batch_size = int(training.get("batch_size", 32))
    max_epochs = int(max_epochs_override or training.get("max_epochs", 100))
    patience = int(training.get("patience", 12))
    aux_weight = float(config.get("aux_weight", 0.0))
    dataset = SequenceDataset(train_x, train_y, train_mask, train_aux, train_aux_mask)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history = []
    best_metric = float("inf")
    best_epoch = 0
    best_state = None
    best_groups: list[float | None] = [None, None, None]
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        totals, mains, auxiliaries = [], [], []
        for x, y, mask, aux, aux_mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            aux, aux_mask = aux.to(device), aux_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power_prediction, aux_prediction = model(x)
                total, main, auxiliary = total_multitask_loss(
                    power_prediction, y, mask, aux_prediction, aux, aux_mask, aux_weight
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(total.detach().cpu()))
            mains.append(float(main.detach().cpu()))
            auxiliaries.append(float(auxiliary.detach().cpu()))

        validation_prediction = predict(model, valid_x, batch_size, device)
        validation_metric, group_metrics = macro_nmae_cf(validation_prediction, valid_y, valid_mask)
        scheduler.step(validation_metric)
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(totals)),
            "train_power_loss": float(np.mean(mains)),
            "train_aux_loss": float(np.mean(auxiliaries)),
            "valid_macro_nmae": validation_metric,
            "valid_group_1_nmae": group_metrics[0],
            "valid_group_2_nmae": group_metrics[1],
            "valid_group_3_nmae": group_metrics[2],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['train_power_loss']:.6f} "
            f"aux={row['train_aux_loss']:.6f} valid={validation_metric:.6f}",
            flush=True,
        )
        if validation_metric < best_metric - 1e-7:
            best_metric = validation_metric
            best_epoch = epoch
            best_groups = group_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_macro_nmae": best_metric,
            "feature_dim": int(train_x.shape[-1]),
        },
        checkpoint_path,
    )
    (checkpoint_path.with_suffix(".history.json")).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_prediction = predict(model, valid_x, batch_size, device)
    return TrainingResult(
        prediction_cf=final_prediction,
        history=history,
        best_epoch=best_epoch,
        best_macro_nmae=best_metric,
        group_nmae=best_groups,
        checkpoint_path=checkpoint_path,
        training_seconds=float(time.perf_counter() - started),
        device=str(device),
    )


def train_fixed_epochs(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    config: dict,
    seed: int,
    epochs: int,
    checkpoint_path: Path,
    aux: np.ndarray | None = None,
    aux_mask: np.ndarray | None = None,
) -> tuple[torch.nn.Module, list[dict], str]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = config["training"]
    batch_size = int(training.get("batch_size", 32))
    dataset = SequenceDataset(x, y, mask, aux, aux_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    aux_weight = float(config.get("aux_weight", 0.0))
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for bx, by, bm, ba, bam in loader:
            bx, by, bm, ba, bam = bx.to(device), by.to(device), bm.to(device), ba.to(device), bam.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power, auxiliary = model(bx)
                total, main, aux_loss = total_multitask_loss(power, by, bm, auxiliary, ba, bam, aux_weight)
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer); scaler.update(); losses.append(float(total.detach().cpu()))
        history.append({"epoch": epoch, "train_total_loss": float(np.mean(losses))})
        print(f"full epoch={epoch:03d} loss={history[-1]['train_total_loss']:.6f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed, "epochs": epochs,
                "feature_dim": int(x.shape[-1])}, checkpoint_path)
    return model, history, str(device)
