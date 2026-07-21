"""Stage-1 raw-weather dataset; SCADA exists only in the target tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.exp04_raw_grid_spatiotemporal.src.trainer import RawModelInputs

from .scada_hourly_targets import HubWindTargetScaler


STAGE1_INPUT_NAMES = (
    "ldaps_dynamic",
    "gfs_dynamic",
    "engineered_common",
    "engineered_group",
)


def assert_stage1_input_schema(names: tuple[str, ...] | list[str]) -> None:
    forbidden = ("scada", "target", "lag", "disagreement")
    matches = [name for name in names if any(token in name.lower() for token in forbidden)]
    if matches:
        raise ValueError(f"forbidden Stage-1 inputs: {matches}")


class Stage1Dataset(Dataset):
    input_names = STAGE1_INPUT_NAMES

    def __init__(
        self,
        inputs: RawModelInputs,
        targets: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
    ) -> None:
        assert_stage1_input_schema(self.input_names)
        self.ldaps = torch.as_tensor(inputs.ldaps, dtype=torch.float32)
        self.gfs = torch.as_tensor(inputs.gfs, dtype=torch.float32)
        self.common = torch.as_tensor(inputs.engineered_common, dtype=torch.float32)
        self.group = torch.as_tensor(inputs.engineered_group, dtype=torch.float32)
        target_count = 4 if targets is None else int(np.asarray(targets).shape[-1])
        if target_count not in {1, 2, 4}:
            raise ValueError("Stage-1 target count must be 1, 2, or 4")
        shape = (len(inputs), inputs.ldaps.shape[1], 3, target_count)
        values = np.zeros(shape, dtype=np.float32) if targets is None else np.asarray(targets, dtype=np.float32).copy()
        mask = np.zeros(shape, dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool).copy()
        if values.shape != shape or mask.shape != shape:
            raise ValueError(f"Stage-1 target schema must be {shape}")
        mask &= np.isfinite(values)
        values[~mask] = 0.0
        self.targets = torch.from_numpy(values)
        self.target_mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return len(self.ldaps)

    def __getitem__(self, index: int):
        return (
            self.ldaps[index], self.gfs[index], self.common[index], self.group[index],
            self.targets[index], self.target_mask[index],
        )


@dataclass
class PreparedStage1Targets:
    train: np.ndarray
    train_mask: np.ndarray
    valid_raw: np.ndarray
    valid_mask: np.ndarray
    scaler: HubWindTargetScaler


def prepare_stage1_targets(
    train_raw: np.ndarray,
    train_mask: np.ndarray,
    valid_raw: np.ndarray,
    valid_mask: np.ndarray,
) -> PreparedStage1Targets:
    scaler = HubWindTargetScaler().fit(train_raw, train_mask)
    train, clean_train_mask = scaler.transform(train_raw, train_mask)
    # Validation remains raw for interpretable m/s metrics; transform only for loss if needed.
    return PreparedStage1Targets(train, clean_train_mask, valid_raw, valid_mask, scaler)
