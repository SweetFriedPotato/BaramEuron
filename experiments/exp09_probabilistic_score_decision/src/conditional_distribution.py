from __future__ import annotations

import numpy as np

from . import QUANTILE_LEVELS


def interpolate_quantile_function(quantiles: np.ndarray, u: np.ndarray) -> np.ndarray:
    values = np.asarray(quantiles, dtype=float)
    if values.shape[-1] != 11 or np.any(np.diff(values, axis=-1) < -1e-10):
        raise ValueError("quantiles must be monotone with 11 levels")
    grid = np.asarray(u, dtype=float)
    flat = values.reshape(-1, 11)
    out = np.stack([np.interp(grid, QUANTILE_LEVELS, row, left=row[0], right=row[-1]) for row in flat])
    return out.reshape(*values.shape[:-1], len(grid))


def deterministic_samples(quantiles: np.ndarray, count: int = 401) -> np.ndarray:
    if count < 3:
        raise ValueError("sample count must be at least three")
    return interpolate_quantile_function(quantiles, (np.arange(count) + 0.5) / count)
