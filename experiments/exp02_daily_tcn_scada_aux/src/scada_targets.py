"""SCADA-derived auxiliary wind targets. SCADA never enters model inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TIME_COL


SCADA_GROUPS = {
    1: ("vestas", [f"vestas_wtg{i:02d}_ws" for i in range(1, 7)]),
    2: ("vestas", [f"vestas_wtg{i:02d}_ws" for i in range(7, 13)]),
    3: ("unison", [f"unison_wtg{i:02d}_ws" for i in range(1, 6)]),
}


def _load_scada(data_root: Path, kind: str) -> pd.DataFrame:
    path = data_root / "train" / f"scada_{kind}_train.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"])
    return frame


def _hourly_group_target(frame: pd.DataFrame, columns: list[str], group_id: int) -> pd.DataFrame:
    values = frame[["kst_dtm", *columns]].copy()
    values[columns] = values[columns].apply(pd.to_numeric, errors="coerce")
    values[columns] = values[columns].where((values[columns] >= 0) & (values[columns] <= 60))
    values[TIME_COL] = values["kst_dtm"].dt.ceil("h")
    grouped = values.groupby(TIME_COL, sort=True)
    means = grouped[columns].mean()
    counts = grouped[columns].count()
    means = means.where(counts >= 6)
    valid_turbines = means.notna().sum(axis=1)
    required = int(np.ceil(len(columns) / 2))
    target = means.median(axis=1, skipna=True).where(valid_turbines >= required)
    out = pd.DataFrame(
        {
            TIME_COL: target.index,
            f"aux_group_{group_id}": target.to_numpy(dtype=float),
            f"aux_group_{group_id}_mask": target.notna().to_numpy(),
            f"aux_group_{group_id}_valid_turbines": valid_turbines.to_numpy(dtype=int),
        }
    )
    if group_id == 3:
        mask_2022 = pd.to_datetime(out[TIME_COL]).dt.year == 2022
        out.loc[mask_2022, f"aux_group_{group_id}"] = np.nan
        out.loc[mask_2022, f"aux_group_{group_id}_mask"] = False
    return out


def build_scada_aux_targets(data_root: Path) -> pd.DataFrame:
    loaded = {kind: _load_scada(data_root, kind) for kind in ("vestas", "unison")}
    parts = [
        _hourly_group_target(loaded[kind], columns, group_id)
        for group_id, (kind, columns) in SCADA_GROUPS.items()
    ]
    output = parts[0]
    for part in parts[1:]:
        output = output.merge(part, on=TIME_COL, how="outer", validate="one_to_one")
    for group_id in (1, 2, 3):
        mask_column = f"aux_group_{group_id}_mask"
        output[mask_column] = output[mask_column].fillna(False).astype(bool)
    return output.sort_values(TIME_COL).reset_index(drop=True)


def write_scada_checks(targets: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for group_id in (1, 2, 3):
        value = f"aux_group_{group_id}"
        mask = targets[f"aux_group_{group_id}_mask"]
        valid = targets.loc[mask, value]
        rows.append(
            {
                "group_id": group_id,
                "valid_hours": int(mask.sum()),
                "missing_hours": int((~mask).sum()),
                "start": None if valid.empty else str(targets.loc[mask, TIME_COL].min()),
                "end": None if valid.empty else str(targets.loc[mask, TIME_COL].max()),
                "mean_mps": None if valid.empty else float(valid.mean()),
                "std_mps": None if valid.empty else float(valid.std()),
                "min_mps": None if valid.empty else float(valid.min()),
                "max_mps": None if valid.empty else float(valid.max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "scada_aux_summary.csv", index=False)
    alignment = {
        "timestamp_rule": "10-minute observations in (hour-1h, hour] are assigned with ceil('h')",
        "per_turbine_requirement": "six valid 10-minute wind readings",
        "group_aggregation": "median of turbine hourly means",
        "minimum_valid_turbines": {"group_1": 3, "group_2": 3, "group_3": 3},
        "static_invalid_rule": "wind < 0 or wind > 60 m/s is invalid",
        "group_3_2022_aux_masked": True,
        "scada_is_model_input": False,
        "test_builder_reads_scada": False,
    }
    (output_dir / "scada_aux_alignment.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@dataclass
class AuxiliaryTargetScaler:
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "AuxiliaryTargetScaler":
        self.lower_ = np.zeros(3, dtype=np.float32)
        self.upper_ = np.zeros(3, dtype=np.float32)
        self.mean_ = np.zeros(3, dtype=np.float32)
        self.scale_ = np.ones(3, dtype=np.float32)
        for group in range(3):
            valid = values[:, :, group][mask[:, :, group] & np.isfinite(values[:, :, group])]
            if valid.size == 0:
                continue
            self.lower_[group], self.upper_[group] = np.quantile(
                valid, [self.lower_quantile, self.upper_quantile]
            ).astype(np.float32)
            retained = valid[(valid >= self.lower_[group]) & (valid <= self.upper_[group])]
            self.mean_[group] = float(retained.mean())
            scale = float(retained.std())
            self.scale_[group] = scale if scale > 1e-6 else 1.0
        return self

    def transform(self, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = np.array(values, dtype=np.float32, copy=True)
        valid_mask = np.array(mask, dtype=bool, copy=True) & np.isfinite(transformed)
        for group in range(3):
            valid_mask[:, :, group] &= (
                (transformed[:, :, group] >= self.lower_[group])
                & (transformed[:, :, group] <= self.upper_[group])
            )
            transformed[:, :, group] = (
                transformed[:, :, group] - self.mean_[group]
            ) / self.scale_[group]
        transformed[~valid_mask] = 0.0
        return transformed, valid_mask

    def state(self) -> dict:
        return {
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower": self.lower_.tolist(),
            "upper": self.upper_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }
