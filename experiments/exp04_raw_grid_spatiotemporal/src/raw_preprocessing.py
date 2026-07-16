"""Leakage-safe per-source dynamic preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChannelState:
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    fit_samples: int

    def metadata(self, names: list[str]) -> dict:
        return {
            name: {
                "median": float(self.median[index]),
                "clip_lower": float(self.lower[index]),
                "clip_upper": float(self.upper[index]),
                "mean": float(self.mean[index]),
                "std": float(self.std[index]),
            }
            for index, name in enumerate(names)
        }


class FoldRawPreprocessor:
    def __init__(self, lower_quantile: float = 0.001, upper_quantile: float = 0.999) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("invalid clipping quantiles")
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.states: dict[str, ChannelState] = {}

    def _fit_one(self, values: np.ndarray) -> ChannelState:
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        flat[~np.isfinite(flat)] = np.nan
        median = np.nanmedian(flat, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(flat), median[None, :], flat)
        lower = np.quantile(filled, self.lower_quantile, axis=0)
        upper = np.quantile(filled, self.upper_quantile, axis=0)
        clipped = np.clip(filled, lower, upper)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        return ChannelState(median, lower, upper, mean, std, len(flat))

    def fit(self, ldaps: np.ndarray, gfs: np.ndarray) -> "FoldRawPreprocessor":
        self.states = {"ldaps": self._fit_one(ldaps), "gfs": self._fit_one(gfs)}
        return self

    def _transform_one(self, values: np.ndarray, source: str) -> np.ndarray:
        if source not in self.states:
            raise RuntimeError("preprocessor must be fit before transform")
        state = self.states[source]
        data = np.asarray(values, dtype=np.float32).copy()
        data[~np.isfinite(data)] = np.nan
        data = np.where(np.isnan(data), state.median, data)
        data = np.clip(data, state.lower, state.upper)
        transformed = (data - state.mean) / state.std
        if not np.isfinite(transformed).all():
            raise ValueError(f"{source} preprocessing produced NaN/inf")
        return transformed.astype(np.float32)

    def transform(self, ldaps: np.ndarray, gfs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._transform_one(ldaps, "ldaps"), self._transform_one(gfs, "gfs")

    def fit_transform(self, ldaps: np.ndarray, gfs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(ldaps, gfs).transform(ldaps, gfs)

    def save_metadata(self, path: Path, ldaps_names: list[str], gfs_names: list[str]) -> None:
        if set(self.states) != {"ldaps", "gfs"}:
            raise RuntimeError("cannot save an unfitted preprocessor")
        payload = {
            "fit_scope": "fold_train_only",
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "sources": {
                "ldaps": self.states["ldaps"].metadata(ldaps_names),
                "gfs": self.states["gfs"].metadata(gfs_names),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
