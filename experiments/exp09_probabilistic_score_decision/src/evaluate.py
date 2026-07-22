from __future__ import annotations

import numpy as np

from experiments.exp08_scada_hubwind_pretraining.src.evaluate import reproduce_exp04_reference
from . import QUANTILE_LEVELS


def quantile_diagnostics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    valid = np.asarray(mask, bool)
    coverage = []
    for index, level in enumerate(QUANTILE_LEVELS):
        coverage.append({"level": level, "empirical": float((target[valid] <= prediction[..., index][valid]).mean())})
    pit = np.mean(target[..., None] > prediction, axis=-1)[valid]
    return {"coverage": coverage, "interval_90_coverage": float(((target >= prediction[..., 0]) &
             (target <= prediction[..., -1]))[valid].mean()), "pit": pit}


def decision_shift(decision: np.ndarray, reference: np.ndarray) -> dict:
    shift = np.abs(np.asarray(decision) - np.asarray(reference))
    return {"mean": float(np.mean(shift)), "p95": float(np.quantile(shift, 0.95)), "maximum": float(np.max(shift))}
