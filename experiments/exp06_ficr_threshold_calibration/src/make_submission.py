"""Contract-valid Exp06 diagnostic submissions (never auto-uploaded)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.exp05_cross_group_transfer.src.make_submission import make_submission


def write_diagnostic_submission(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    prediction_column: str,
    accepted: bool,
) -> pd.DataFrame:
    frame = make_submission(sample, predictions, path, prediction_column)
    metadata = {
        "submission": str(path), "prediction_column": prediction_column,
        "diagnostic_only": not bool(accepted), "accepted": bool(accepted), "auto_submitted": False,
    }
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame
