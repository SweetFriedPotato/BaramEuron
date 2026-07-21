"""Strictly temporal Stage-1 cross-fitting and forecast fallback contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp03_official_score_calibration.src.backtest import ROLLING_QUARTERS, expanding_quarter_window


@dataclass(frozen=True)
class CrossfitWindow:
    prediction_block: str
    train_start: str | None
    train_end: str | None
    predict_start: str
    predict_end: str
    train_indices: tuple[int, ...]
    predict_indices: tuple[int, ...]
    fallback: bool
    role: str = "historical_oof"


def issue_period(timestamps: np.ndarray) -> np.ndarray:
    first = pd.DatetimeIndex(timestamps[:, 0]) - pd.Timedelta(hours=1)
    return first.to_period("Q").astype(str).to_numpy()


def expanding_crossfit_windows(
    timestamps: np.ndarray,
    outer_quarter: str,
    *,
    min_train_blocks: int = 90,
) -> list[CrossfitWindow]:
    if timestamps.ndim != 2 or timestamps.shape[1] != 24:
        raise ValueError("cross-fit timestamps must be issue blocks [N,24]")
    window = expanding_quarter_window(outer_quarter)
    first = pd.DatetimeIndex(timestamps[:, 0])
    outer_train = np.flatnonzero((first >= window["train_start"]) & (first <= window["train_end"]))
    outer_valid = np.flatnonzero((first >= window["valid_start"]) & (first <= window["valid_end"]))
    periods = issue_period(timestamps)
    records: list[CrossfitWindow] = []
    for block in sorted(set(periods[outer_train])):
        predict = outer_train[periods[outer_train] == block]
        train = outer_train[pd.PeriodIndex(periods[outer_train], freq="Q") < pd.Period(block, freq="Q")]
        # Prior official outer-quarter models are the learned historical OOF
        # registry. 2022 has no earlier canonical outer model and must fallback.
        registry_eligible = block in ROLLING_QUARTERS and pd.Period(block, freq="Q") < pd.Period(outer_quarter, freq="Q")
        fallback = (not registry_eligible) or len(train) < int(min_train_blocks)
        usable = np.array([], dtype=int) if fallback else train
        records.append(_window(block, usable, predict, first, fallback))
    records.append(_window(outer_quarter, outer_train, outer_valid, first, False, role="outer_validation"))
    assert_temporal_crossfit(records, first)
    return records


def _window(
    name: str,
    train: np.ndarray,
    predict: np.ndarray,
    first: pd.DatetimeIndex,
    fallback: bool,
    role: str = "historical_oof",
) -> CrossfitWindow:
    if len(predict) == 0:
        raise ValueError(f"empty cross-fit prediction block: {name}")
    return CrossfitWindow(
        prediction_block=name,
        train_start=None if len(train) == 0 else str(first[train].min()),
        train_end=None if len(train) == 0 else str(first[train].max()),
        predict_start=str(first[predict].min()),
        predict_end=str(first[predict].max()),
        train_indices=tuple(int(value) for value in train),
        predict_indices=tuple(int(value) for value in predict),
        fallback=bool(fallback),
        role=role,
    )


def assert_temporal_crossfit(records: list[CrossfitWindow], first_times: pd.DatetimeIndex | None = None) -> None:
    for record in records:
        if set(record.train_indices) & set(record.predict_indices):
            raise ValueError(f"in-sample Stage-1 prediction in {record.prediction_block}")
        if not record.fallback and record.train_end is not None:
            if pd.Timestamp(record.train_end) >= pd.Timestamp(record.predict_start):
                raise ValueError(f"Stage-1 cross-fit is not strictly temporal in {record.prediction_block}")
        if first_times is not None and record.train_indices and record.predict_indices:
            if first_times[list(record.train_indices)].max() >= first_times[list(record.predict_indices)].min():
                raise ValueError("cross-fit index timestamps violate ordering")


def forecast_hubwind_fallback(gfs_ws100: np.ndarray) -> np.ndarray:
    """Target-free fallback in physical units, shape ``[N,24,3,4]``."""
    values = np.asarray(gfs_ws100, dtype=np.float32)
    if values.ndim == 3:
        values = np.nanmean(values, axis=2)
    if values.ndim != 2 or values.shape[1] != 24:
        raise ValueError("fallback expects GFS ws100 [N,24,grid] or [N,24]")
    output = np.zeros((*values.shape, 3, 4), dtype=np.float32)
    output[..., 0] = values[..., None]
    output[..., 1] = values[..., None]
    return output


def initialize_crossfit_arrays(gfs_ws100: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = forecast_hubwind_fallback(gfs_ws100)
    feature_mask = np.zeros(prediction.shape[:-1], dtype=bool)
    fallback = np.ones(prediction.shape[:-1], dtype=np.float32)
    return prediction, feature_mask, fallback


def assign_crossfit_prediction(
    destination: np.ndarray,
    feature_mask: np.ndarray,
    fallback: np.ndarray,
    record: CrossfitWindow,
    values: np.ndarray,
) -> None:
    if record.fallback:
        raise ValueError("fallback windows must not receive learned Stage-1 predictions")
    indices = np.asarray(record.predict_indices, dtype=int)
    if values.shape != destination[indices].shape:
        raise ValueError("cross-fit prediction schema mismatch")
    destination[indices] = values
    feature_mask[indices] = True
    fallback[indices] = 0.0


def write_crossfit_contract(
    records_by_outer: dict[str, list[CrossfitWindow]],
    path: Path,
) -> dict:
    payload = {
        "protocol": "expanding-window; each learned feature is generated by strictly earlier SCADA",
        "early_history_policy": "mask=0, target-free GFS ws100 fallback, fallback_indicator=1",
        "in_sample_stage1_features": False,
        "outer_quarters": {
            outer: [asdict(record) for record in records]
            for outer, records in records_by_outer.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
