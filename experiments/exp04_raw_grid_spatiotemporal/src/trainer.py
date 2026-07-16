"""A100 trainer for multi-input raw-grid models."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.exp02_daily_tcn_scada_aux.src.trainer import seed_everything
from experiments.exp03_official_score_calibration.src.ficr_surrogate import total_official_loss
from experiments.exp03_official_score_calibration.src.train_variants import official_validation_score


@dataclass
class RawModelInputs:
    ldaps: np.ndarray
    gfs: np.ndarray
    engineered_common: np.ndarray
    engineered_group: np.ndarray

    def subset(self, indices: np.ndarray) -> "RawModelInputs":
        return RawModelInputs(
            self.ldaps[indices], self.gfs[indices],
            self.engineered_common[indices], self.engineered_group[indices],
        )

    def __len__(self) -> int:
        return len(self.ldaps)


class RawGridDataset(Dataset):
    """SCADA appears only in auxiliary targets, never in the input tuple."""

    input_names = ("ldaps_dynamic", "gfs_dynamic", "engineered_common", "engineered_group")

    def __init__(
        self,
        inputs: RawModelInputs,
        target: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        auxiliary: np.ndarray | None = None,
        auxiliary_mask: np.ndarray | None = None,
    ) -> None:
        self.ldaps = torch.as_tensor(inputs.ldaps, dtype=torch.float32)
        self.gfs = torch.as_tensor(inputs.gfs, dtype=torch.float32)
        self.common = torch.as_tensor(inputs.engineered_common, dtype=torch.float32)
        self.group = torch.as_tensor(inputs.engineered_group, dtype=torch.float32)
        shape = (len(inputs), inputs.ldaps.shape[1], 3)
        mask = np.zeros(shape, dtype=bool) if label_mask is None else np.asarray(label_mask, dtype=bool)
        values = np.zeros(shape, dtype=np.float32) if target is None else np.asarray(target, dtype=np.float32).copy()
        values[~mask] = 0.0
        self.target = torch.from_numpy(values)
        self.label_mask = torch.from_numpy(mask)
        aux_mask = np.zeros(shape, dtype=bool) if auxiliary_mask is None else np.asarray(auxiliary_mask, dtype=bool)
        aux_values = np.zeros(shape, dtype=np.float32) if auxiliary is None else np.asarray(auxiliary, dtype=np.float32).copy()
        aux_values[~aux_mask] = 0.0
        self.auxiliary = torch.from_numpy(aux_values)
        self.auxiliary_mask = torch.from_numpy(aux_mask)

    def __len__(self) -> int:
        return len(self.ldaps)

    def __getitem__(self, index: int):
        return (
            self.ldaps[index], self.gfs[index], self.common[index], self.group[index],
            self.target[index], self.label_mask[index], self.auxiliary[index], self.auxiliary_mask[index],
        )


@dataclass
class RawTrainingResult:
    prediction_cf: np.ndarray
    diagnostics: dict[str, np.ndarray | None]
    history: list[dict]
    best_epoch: int
    best_total_score: float
    best_one_minus_nmae: float
    best_ficr: float
    group_metrics: list[dict]
    checkpoint_path: Path
    training_seconds: float
    device: str
    peak_gpu_memory_mb: float


def _move(batch, device: torch.device):
    return [value.to(device, non_blocking=True) for value in batch]


def predict_raw(
    model: torch.nn.Module,
    inputs: RawModelInputs,
    batch_size: int,
    device: torch.device,
    capture_diagnostics: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
    dataset = RawGridDataset(inputs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    diagnostics: dict[str, list[np.ndarray]] = {
        "ldaps_attention": [], "gfs_attention": [], "source_gate": [],
        "cross_group_attention": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ldaps, gfs, common, group = _move(batch[:4], device)
            power, _, values = model(ldaps, gfs, common, group)
            predictions.append(power.detach().cpu().numpy())
            if capture_diagnostics:
                for name in diagnostics:
                    value = values.get(name)
                    if value is not None:
                        diagnostics[name].append(value.detach().cpu().numpy())
    merged = {
        name: (np.concatenate(parts, axis=0) if parts else None)
        for name, parts in diagnostics.items()
    }
    return np.concatenate(predictions, axis=0), merged


def train_raw_model(
    model: torch.nn.Module,
    train_inputs: RawModelInputs,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    valid_inputs: RawModelInputs,
    valid_y: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
    seed: int,
    checkpoint_path: Path,
    train_aux: np.ndarray | None = None,
    train_aux_mask: np.ndarray | None = None,
    max_epochs_override: int | None = None,
) -> RawTrainingResult:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = config["training"]
    batch_size = int(training.get("batch_size", 16))
    max_epochs = int(max_epochs_override or training.get("max_epochs", 100))
    patience = int(training.get("patience", 12))
    dataset = RawGridDataset(train_inputs, train_y, train_mask, train_aux, train_aux_mask)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict] = []
    best_state = None
    best_score, best_nmae, best_ficr, best_epoch, stale = -np.inf, np.nan, np.nan, 0, 0
    best_groups: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        totals, mains, ficrs, auxiliaries = [], [], [], []
        for batch in loader:
            ldaps, gfs, common, group, target, mask, aux, aux_mask = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power, auxiliary, _ = model(ldaps, gfs, common, group)
                total, main, ficr_loss, aux_loss = total_official_loss(
                    power, target, mask, auxiliary, aux, aux_mask,
                    aux_weight=float(config.get("aux_weight", 0.05)),
                    lambda_ficr=float(config.get("lambda_ficr", 0.20)),
                    temperature=float(config.get("temperature", 0.005)),
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(total.detach().cpu()))
            mains.append(float(main.detach().cpu()))
            ficrs.append(float(ficr_loss.detach().cpu()))
            auxiliaries.append(float(aux_loss.detach().cpu()))
        validation, _ = predict_raw(model, valid_inputs, batch_size, device)
        score, one_minus_nmae, ficr, groups = official_validation_score(validation, valid_y, valid_mask)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(totals)),
            "train_power_loss": float(np.mean(mains)),
            "train_ficr_loss": float(np.mean(ficrs)),
            "train_aux_loss": float(np.mean(auxiliaries)),
            "valid_total_score": score,
            "valid_one_minus_nmae": one_minus_nmae,
            "valid_ficr": ficr,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_total_loss']:.6f} score={score:.6f} "
            f"1-nmae={one_minus_nmae:.6f} ficr={ficr:.6f}", flush=True,
        )
        if score > best_score + 1e-12:
            best_score, best_nmae, best_ficr, best_epoch = score, one_minus_nmae, ficr, epoch
            best_state, best_groups, stale = copy.deepcopy(model.state_dict()), groups, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("raw-grid trainer did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_total_score": best_score,
            "best_one_minus_nmae": best_nmae,
            "best_ficr": best_ficr,
        },
        checkpoint_path,
    )
    checkpoint_path.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction, diagnostics = predict_raw(model, valid_inputs, batch_size, device, capture_diagnostics=True)
    peak = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    return RawTrainingResult(
        prediction, diagnostics, history, best_epoch, best_score, best_nmae, best_ficr,
        best_groups, checkpoint_path, time.perf_counter() - started, str(device), peak,
    )


def train_raw_fixed_epochs(
    model: torch.nn.Module,
    inputs: RawModelInputs,
    target: np.ndarray,
    label_mask: np.ndarray,
    config: dict,
    seed: int,
    epochs: int,
    checkpoint_path: Path,
    auxiliary: np.ndarray | None = None,
    auxiliary_mask: np.ndarray | None = None,
) -> tuple[torch.nn.Module, list[dict], str]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = config["training"]
    dataset = RawGridDataset(inputs, target, label_mask, auxiliary, auxiliary_mask)
    loader = DataLoader(
        dataset, batch_size=int(training.get("batch_size", 16)), shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train(); totals = []
        for batch in loader:
            ldaps, gfs, common, group, y, mask, aux, aux_mask = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                power, auxiliary_prediction, _ = model(ldaps, gfs, common, group)
                total, _, _, _ = total_official_loss(
                    power, y, mask, auxiliary_prediction, aux, aux_mask,
                    aux_weight=float(config.get("aux_weight", 0.05)),
                    lambda_ficr=float(config.get("lambda_ficr", 0.20)),
                    temperature=float(config.get("temperature", 0.005)),
                )
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer); scaler.update(); totals.append(float(total.detach().cpu()))
        row = {"epoch": epoch, "train_total_loss": float(np.mean(totals))}
        history.append(row); print(f"full epoch={epoch:03d} loss={row['train_total_loss']:.6f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed, "epochs": epochs}, checkpoint_path)
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history, str(device)
