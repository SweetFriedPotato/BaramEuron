"""Submission writer with the exact 8,760-row BARAM contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TARGETS, TIME_COL


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame, expected_rows: int = 8760) -> None:
    if len(frame) != expected_rows or len(sample) != expected_rows:
        raise ValueError(f"submission must contain {expected_rows:,} rows")
    if list(frame.columns) != list(sample.columns):
        raise ValueError("submission columns/order differ from sample")
    key_columns = [column for column in ("forecast_id", TIME_COL) if column in sample]
    if key_columns and not frame[key_columns].equals(sample[key_columns]):
        raise ValueError("submission key/timestamp order differs from sample")
    if key_columns and frame.duplicated(key_columns).any():
        raise ValueError("submission contains duplicate keys")
    values = frame[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("submission contains NaN/inf")
    if any(not pd.api.types.is_numeric_dtype(frame[target]) for target in TARGETS):
        raise TypeError("submission target columns must be numeric")


def create_finetuned_submission(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    *,
    prediction_column: str = "y_pred_kwh",
    expected_rows: int = 8760,
) -> Path:
    required = {TIME_COL, "target", prediction_column}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    if predictions.duplicated([TIME_COL, "target"]).any():
        raise ValueError("test predictions contain duplicate timestamp/target keys")
    wide = predictions.pivot(index=TIME_COL, columns="target", values=prediction_column)
    wide.index = pd.to_datetime(wide.index)
    output = sample.copy()
    timestamp = pd.to_datetime(output[TIME_COL])
    for target in TARGETS:
        if target not in wide:
            raise ValueError(f"prediction is missing {target}")
        output[target] = wide.loc[timestamp, target].to_numpy(dtype=float)
    validate_submission(output, sample, expected_rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return path

