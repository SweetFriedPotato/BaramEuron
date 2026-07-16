"""Create at most three contract-valid Exp05 submissions without uploading them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TARGETS, TIME_COL
from baram.submission import create_submission, validate_submission_contract


def make_submission(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    prediction_column: str = "y_pred_kwh",
) -> pd.DataFrame:
    if {TIME_COL, "target", prediction_column}.issubset(predictions):
        if predictions.duplicated([TIME_COL, "target"]).any():
            raise ValueError("long test predictions contain duplicate timestamp/target rows")
        wide = predictions.pivot(index=TIME_COL, columns="target", values=prediction_column).reset_index()
    else:
        wide = predictions.copy()
    if set(TARGETS) - set(wide):
        raise ValueError("test prediction is missing one or more group columns")
    if wide[TIME_COL].duplicated().any():
        raise ValueError("test prediction contains duplicate timestamps")
    lookup = wide.assign(**{TIME_COL: pd.to_datetime(wide[TIME_COL])}).set_index(TIME_COL)
    sample_times = pd.to_datetime(sample[TIME_COL])
    if not sample_times.isin(lookup.index).all():
        raise ValueError("test prediction timestamps do not cover the sample submission")
    values = lookup.loc[sample_times, TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("submission prediction contains NaN/inf")
    frame = create_submission(
        sample,
        {target: np.maximum(values[:, index], 0.0) for index, target in enumerate(TARGETS)},
        Path(path),
    )
    validate_submission_contract(frame, sample)
    return frame


def validate_submission_limit(paths: list[Path], maximum: int = 3) -> None:
    if len(paths) > maximum:
        raise ValueError(f"at most {maximum} submissions may be generated")
