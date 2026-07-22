from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GroupConformalOffsets:
    offsets: np.ndarray | None = None

    def fit(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> "GroupConformalOffsets":
        if prediction.shape[:-1] != target.shape or target.shape != mask.shape:
            raise ValueError("calibration schema differs")
        self.offsets = np.zeros((3, prediction.shape[-1]), dtype=np.float32)
        for group in range(3):
            valid = mask[..., group]
            if valid.any():
                self.offsets[group] = np.nanmedian(target[..., group, None][valid] - prediction[..., group, :][valid], axis=0)
        return self

    def transform(self, prediction: np.ndarray) -> np.ndarray:
        if self.offsets is None:
            raise RuntimeError("calibrator is not fit")
        return np.maximum.accumulate(np.asarray(prediction) + self.offsets, axis=-1)
