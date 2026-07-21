"""Fold-train-cleaned, hour-ending SCADA hub-wind distribution targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .scada_contract import (
    GROUP_SOURCES,
    GROUP_TURBINES,
    TARGET_NAMES,
    TIME_COLUMN,
    load_scada_wind,
    write_source_contract,
)


def hour_ending(timestamps: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(timestamps)).ceil("h")


@dataclass
class SourceCleaningState:
    lower: float
    upper: float
    finite_nonnegative_samples: int
    fit_start: str
    fit_end: str
    available_in_fold_train: bool = True


class FoldScadaCleaner:
    """Fit source-specific wind limits on the fold training interval only."""

    def __init__(self, lower_quantile: float = 0.001, upper_quantile: float = 0.999) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("invalid SCADA cleaning quantiles")
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.states: dict[str, SourceCleaningState] = {}

    def fit(self, frames: dict[str, pd.DataFrame], fit_end: pd.Timestamp | None = None) -> "FoldScadaCleaner":
        self.states = {}
        for source, frame in frames.items():
            fit = frame if fit_end is None else frame.loc[frame[TIME_COLUMN] <= pd.Timestamp(fit_end)]
            columns = [column for column in fit if column.endswith("_ws")]
            raw = fit[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).reshape(-1)
            valid = raw[np.isfinite(raw) & (raw >= 0.0)]
            if valid.size == 0:
                # Group 3 starts in 2023. For the 2023Q1 outer fold there is no
                # earlier UNISON supervision, so every group-3 target must be
                # masked rather than borrowing thresholds from validation.
                self.states[source] = SourceCleaningState(
                    0.0, 0.0, 0,
                    "unavailable", "unavailable", False,
                )
                continue
            lower, upper = np.quantile(valid, [self.lower_quantile, self.upper_quantile])
            self.states[source] = SourceCleaningState(
                float(max(0.0, lower)), float(upper), int(valid.size),
                str(fit[TIME_COLUMN].min()), str(fit[TIME_COLUMN].max()),
            )
        return self

    def transform(self, frame: pd.DataFrame, source: str) -> pd.DataFrame:
        if source not in self.states:
            raise RuntimeError("SCADA cleaner must be fit before transform")
        state = self.states[source]
        out = frame.copy()
        columns = [column for column in out if column.endswith("_ws")]
        numeric = out[columns].apply(pd.to_numeric, errors="coerce")
        if not state.available_in_fold_train:
            out[columns] = np.nan
            return out
        valid = np.isfinite(numeric) & numeric.ge(0.0) & numeric.ge(state.lower) & numeric.le(state.upper)
        out[columns] = numeric.where(valid)
        return out

    def state_dict(self) -> dict:
        return {
            "fit_scope": "fold_train_only",
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "fixed_high_wind_clip": None,
            "metadata_note": "supplied metadata defines units but no authoritative maximum wind speed",
            "sources": {name: vars(state) for name, state in self.states.items()},
        }


def _hourly_group(frame: pd.DataFrame, group_id: int) -> pd.DataFrame:
    columns = list(GROUP_TURBINES[group_id])
    values = frame[[TIME_COLUMN, *columns]].copy()
    values["hour_ending"] = hour_ending(values[TIME_COLUMN])
    grouped = values.groupby("hour_ending", sort=True)
    # A turbine-hour is valid with at least half of the expected six readings.
    means = grouped[columns].mean()
    counts = grouped[columns].count()
    turbine_valid = counts >= 3
    turbine_means = means.where(turbine_valid)
    valid_turbines = turbine_means.notna().sum(axis=1)
    required_turbines = int(np.ceil(len(columns) / 2.0))
    group_mask = valid_turbines >= required_turbines
    flattened = grouped[columns].agg(lambda part: np.nanstd(part.to_numpy(dtype=float), ddof=0))
    within_hour_variability = flattened.mean(axis=1, skipna=True)
    q75 = turbine_means.quantile(0.75, axis=1)
    q25 = turbine_means.quantile(0.25, axis=1)
    prefix = f"group_{group_id}__"
    result = pd.DataFrame(
        {
            TIME_COLUMN: turbine_means.index,
            prefix + "hub_ws_median": turbine_means.median(axis=1, skipna=True),
            prefix + "hub_ws_mean": turbine_means.mean(axis=1, skipna=True),
            prefix + "hub_ws_std": turbine_means.std(axis=1, skipna=True, ddof=0),
            prefix + "hub_ws_iqr": q75 - q25,
            prefix + "valid_turbines": valid_turbines,
            prefix + "within_hour_variability": within_hour_variability,
            prefix + "target_mask": group_mask,
        }
    )
    target_columns = [prefix + name for name in TARGET_NAMES]
    result.loc[~group_mask, target_columns] = np.nan
    return result.reset_index(drop=True)


def build_hourly_scada_targets(
    data_root: Path,
    *,
    fit_end: pd.Timestamp | None = None,
    label_timestamps: pd.Series | pd.DatetimeIndex | None = None,
    cleaner: FoldScadaCleaner | None = None,
) -> tuple[pd.DataFrame, FoldScadaCleaner]:
    raw = {source: load_scada_wind(data_root, source) for source in ("vestas", "unison")}
    fitted = cleaner or FoldScadaCleaner().fit(raw, fit_end=fit_end)
    clean = {source: fitted.transform(frame, source) for source, frame in raw.items()}
    parts = [_hourly_group(clean[GROUP_SOURCES[group_id]], group_id) for group_id in (1, 2, 3)]
    output = parts[0]
    for part in parts[1:]:
        output = output.merge(part, on=TIME_COLUMN, how="outer", validate="one_to_one")
    if label_timestamps is not None:
        labels = pd.DataFrame({TIME_COLUMN: pd.DatetimeIndex(pd.to_datetime(label_timestamps))})
        output = labels.merge(output, on=TIME_COLUMN, how="left", validate="one_to_one")
    for group_id in (1, 2, 3):
        mask_column = f"group_{group_id}__target_mask"
        output[mask_column] = output[mask_column].fillna(False).astype(bool)
    return output.sort_values(TIME_COLUMN).reset_index(drop=True), fitted


def target_arrays(
    targets: pd.DataFrame, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    shape = (*timestamps.shape, 3, len(TARGET_NAMES))
    indexed = targets.set_index(TIME_COLUMN).reindex(pd.DatetimeIndex(timestamps.reshape(-1)))
    values = np.full(shape, np.nan, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    for group_index, group_id in enumerate((1, 2, 3)):
        columns = [f"group_{group_id}__{name}" for name in TARGET_NAMES]
        group_values = indexed[columns].to_numpy(dtype=np.float32).reshape(*timestamps.shape, len(columns))
        group_mask = indexed[f"group_{group_id}__target_mask"].fillna(False).to_numpy(dtype=bool)
        group_mask = group_mask.reshape(timestamps.shape)
        values[..., group_index, :] = group_values
        mask[..., group_index, :] = group_mask[..., None] & np.isfinite(group_values)
    return values, mask


def write_hourly_checks(
    targets: pd.DataFrame,
    cleaner: FoldScadaCleaner,
    data_root: Path,
    checks_dir: Path,
) -> None:
    checks_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows, statistic_rows = [], []
    for group_id in (1, 2, 3):
        mask = targets[f"group_{group_id}__target_mask"].astype(bool)
        coverage_rows.append(
            {
                "group_id": group_id,
                "source": GROUP_SOURCES[group_id],
                "available_hours": int(mask.sum()),
                "missing_hours": int((~mask).sum()),
                "coverage": float(mask.mean()),
                "first_available": None if not mask.any() else str(targets.loc[mask, TIME_COLUMN].min()),
                "last_available": None if not mask.any() else str(targets.loc[mask, TIME_COLUMN].max()),
            }
        )
        for target in TARGET_NAMES:
            column = f"group_{group_id}__{target}"
            valid = targets.loc[mask, column].dropna()
            statistic_rows.append(
                {
                    "group_id": group_id,
                    "target": target,
                    "samples": int(len(valid)),
                    "mean": float(valid.mean()) if len(valid) else np.nan,
                    "std": float(valid.std(ddof=0)) if len(valid) else np.nan,
                    "minimum": float(valid.min()) if len(valid) else np.nan,
                    "maximum": float(valid.max()) if len(valid) else np.nan,
                }
            )
    pd.DataFrame(coverage_rows).to_csv(checks_dir / "scada_target_coverage.csv", index=False)
    pd.DataFrame(statistic_rows).to_csv(checks_dir / "scada_target_statistics.csv", index=False)
    extra = {
        "cleaning": cleaner.state_dict(),
        "per_turbine_minimum_readings": 3,
        "expected_readings_per_hour": 6,
        "minimum_valid_turbines": {"group_1": 3, "group_2": 3, "group_3": 3},
        "hourly_statistics": [*TARGET_NAMES, "valid_turbines", "within_hour_variability"],
    }
    write_source_contract(data_root, checks_dir / "scada_hourly_contract.json", extra)


@dataclass
class HubWindTargetScaler:
    """Group/target-specific transform fit only on training targets."""

    epsilon: float = 1e-6

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "HubWindTargetScaler":
        if values.shape != mask.shape or values.shape[-2:] != (3, 4):
            raise ValueError("hub target scaler expects [...,3,4]")
        transformed = self._log_variability(values)
        self.mean_ = np.zeros((3, 4), dtype=np.float32)
        self.scale_ = np.ones((3, 4), dtype=np.float32)
        for group in range(3):
            for target in range(4):
                valid = transformed[..., group, target][mask[..., group, target]]
                valid = valid[np.isfinite(valid)]
                if valid.size:
                    self.mean_[group, target] = float(valid.mean())
                    std = float(valid.std())
                    self.scale_[group, target] = std if std > self.epsilon else 1.0
        return self

    @staticmethod
    def _log_variability(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32).copy()
        out[..., 2:] = np.log1p(np.maximum(out[..., 2:], 0.0))
        return out

    def transform(self, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = (self._log_variability(values) - self.mean_) / self.scale_
        valid = np.asarray(mask, dtype=bool) & np.isfinite(out)
        out[~valid] = 0.0
        return out.astype(np.float32), valid

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32) * self.scale_ + self.mean_
        out[..., 2:] = np.expm1(out[..., 2:]).clip(min=0.0)
        return out

    def state_dict(self) -> dict:
        return {"fit_scope": "fold_train_only", "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
