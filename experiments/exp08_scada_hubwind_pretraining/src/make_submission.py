"""Accepted-only Exp08 submission contract helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TARGETS, TIME_COL


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame, expected_rows: int = 8760) -> None:
    if len(frame) != expected_rows or len(sample) != expected_rows:
        raise ValueError(f"submission must contain {expected_rows} rows")
    if list(frame.columns) != list(sample.columns):
        raise ValueError("submission columns/order differ from sample")
    if not frame[["forecast_id", TIME_COL]].equals(sample[["forecast_id", TIME_COL]]):
        raise ValueError("submission keys/order differ from sample")
    if frame["forecast_id"].duplicated().any() or frame[TIME_COL].duplicated().any():
        raise ValueError("submission contains duplicate keys")
    values = frame[TARGETS]
    if not all(pd.api.types.is_numeric_dtype(values[column]) for column in TARGETS):
        raise ValueError("submission targets must be numeric")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("submission contains NaN/inf")


def make_submission(sample: pd.DataFrame, predictions_kwh: np.ndarray, path: Path) -> Path:
    if predictions_kwh.shape != (len(sample), 3):
        raise ValueError("test prediction shape differs from sample")
    frame = sample.copy()
    frame[TARGETS] = np.maximum(np.asarray(predictions_kwh, dtype=float), 0.0)
    validate_submission(frame, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
