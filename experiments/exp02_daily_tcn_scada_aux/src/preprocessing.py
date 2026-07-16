"""Fold-train-only neural input preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass
class NeuralFoldPreprocessor:
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999

    def fit(self, x: np.ndarray) -> "NeuralFoldPreprocessor":
        flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        flat[~np.isfinite(flat)] = np.nan
        self.median_ = np.nanmedian(flat, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        imputed = np.where(np.isnan(flat), self.median_[None, :], flat)
        self.lower_ = np.quantile(imputed, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(imputed, self.upper_quantile, axis=0)
        clipped = np.clip(imputed, self.lower_, self.upper_)
        self.mean_ = clipped.mean(axis=0)
        self.scale_ = clipped.std(axis=0)
        self.scale_ = np.where(self.scale_ > 1e-8, self.scale_, 1.0)
        self.fit_rows_ = int(flat.shape[0])
        self.feature_count_ = int(flat.shape[1])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64).copy()
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.median_, values)
        values = np.clip(values, self.lower_, self.upper_)
        values = (values - self.mean_) / self.scale_
        return values.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "NeuralFoldPreprocessor":
        return joblib.load(path)

    def state(self, feature_names: list[str] | None = None) -> dict:
        return {
            "order": ["inf_to_nan", "median_imputation", "winsorization", "standard_scaler"],
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "fit_rows": self.fit_rows_,
            "feature_count": self.feature_count_,
            "feature_names": feature_names,
            "median": self.median_.tolist(),
            "lower": self.lower_.tolist(),
            "upper": self.upper_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }
