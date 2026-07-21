"""Stage-2 explicit predicted-hub-wind features and leakage-safe imputation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.exp04_raw_grid_spatiotemporal.src.trainer import RawModelInputs


STAGE2_HUB_FEATURES = (
    "predicted_hub_ws_median",
    "predicted_hub_ws_mean",
    "predicted_hub_ws_std",
    "predicted_hub_ws_iqr",
    "stage1_ensemble_seed_std",
    "forecast_minus_predicted_hub_ws",
    "predicted_to_forecast_hub_ws_ratio",
    "stage1_fallback_indicator",
)
EXPLICIT_FEATURE_INDICES = (0, 1, 5, 6, 7)
DISTRIBUTION_FEATURE_INDICES = tuple(range(8))


def assert_stage2_feature_schema(names: tuple[str, ...] | list[str]) -> None:
    forbidden = ("target", "lag", "availability", "scada_")
    matches = [name for name in names if any(token in name.lower() for token in forbidden)]
    if matches:
        raise ValueError(f"forbidden Stage-2 features: {matches}")


def build_stage2_hub_features(
    stage1_prediction: np.ndarray,
    forecast_hub_ws: np.ndarray,
    *,
    seed_std: np.ndarray | None = None,
    fallback_indicator: np.ndarray | None = None,
    ratio_epsilon: float = 0.25,
) -> np.ndarray:
    prediction = np.asarray(stage1_prediction, dtype=np.float32)
    if prediction.ndim != 4 or prediction.shape[-2:] != (3, 4):
        raise ValueError("Stage-1 prediction must be [N,24,3,4]")
    forecast = np.asarray(forecast_hub_ws, dtype=np.float32)
    if forecast.ndim == 2:
        forecast = np.repeat(forecast[..., None], 3, axis=-1)
    if forecast.shape != prediction.shape[:-1]:
        raise ValueError("forecast/Stage-1 hub-wind schema differs")
    uncertainty = np.zeros_like(forecast) if seed_std is None else np.asarray(seed_std, dtype=np.float32)
    fallback = np.zeros_like(forecast) if fallback_indicator is None else np.asarray(fallback_indicator, dtype=np.float32)
    if uncertainty.shape != forecast.shape or fallback.shape != forecast.shape:
        raise ValueError("seed uncertainty/fallback schema differs")
    median = prediction[..., 0]
    ratio = np.full_like(forecast, np.nan)
    np.divide(median, forecast, out=ratio, where=np.abs(forecast) >= float(ratio_epsilon))
    features = np.stack(
        [
            median,
            prediction[..., 1],
            np.maximum(prediction[..., 2], 0.0),
            np.maximum(prediction[..., 3], 0.0),
            np.maximum(uncertainty, 0.0),
            forecast - median,
            ratio,
            fallback,
        ],
        axis=-1,
    ).astype(np.float32)
    assert_stage2_feature_schema(STAGE2_HUB_FEATURES)
    return features


@dataclass
class FoldHubFeatureImputer:
    medians_: np.ndarray | None = None

    def fit(self, train_features: np.ndarray) -> "FoldHubFeatureImputer":
        values = np.asarray(train_features, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        medians = np.nanmedian(values.reshape(-1, values.shape[-1]), axis=0)
        self.medians_ = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("hub feature imputer must be fit on fold training features")
        values = np.asarray(features, dtype=np.float32).copy()
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.broadcast_to(self.medians_, values.shape)[invalid]
        return values

    def save(self, path: Path) -> None:
        if self.medians_ is None:
            raise RuntimeError("cannot save an unfitted hub feature imputer")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "fit_scope": "fold_train_only",
            "feature_names": STAGE2_HUB_FEATURES,
            "medians": self.medians_.tolist(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


class Stage2Dataset(Dataset):
    raw_input_names = ("ldaps_dynamic", "gfs_dynamic", "engineered_common", "engineered_group")

    def __init__(
        self,
        inputs: RawModelInputs,
        hub_features: np.ndarray,
        target: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        retention_target: np.ndarray | None = None,
        retention_mask: np.ndarray | None = None,
    ) -> None:
        assert_stage2_feature_schema(STAGE2_HUB_FEATURES)
        self.ldaps = torch.as_tensor(inputs.ldaps, dtype=torch.float32)
        self.gfs = torch.as_tensor(inputs.gfs, dtype=torch.float32)
        self.common = torch.as_tensor(inputs.engineered_common, dtype=torch.float32)
        self.group = torch.as_tensor(inputs.engineered_group, dtype=torch.float32)
        expected_hub = (len(inputs), inputs.ldaps.shape[1], 3, len(STAGE2_HUB_FEATURES))
        if hub_features.shape != expected_hub or not np.isfinite(hub_features).all():
            raise ValueError(f"Stage-2 hub features must be finite with shape {expected_hub}")
        self.hub = torch.as_tensor(hub_features, dtype=torch.float32)
        shape = expected_hub[:-1]
        values = np.zeros(shape, dtype=np.float32) if target is None else np.asarray(target, dtype=np.float32).copy()
        mask = np.zeros(shape, dtype=bool) if label_mask is None else np.asarray(label_mask, dtype=bool).copy()
        values[~mask] = 0.0
        self.target, self.label_mask = torch.from_numpy(values), torch.from_numpy(mask)
        retention_shape = (*shape, 4)
        retain = np.zeros(retention_shape, dtype=np.float32) if retention_target is None else np.asarray(retention_target, dtype=np.float32).copy()
        retain_mask = np.zeros(retention_shape, dtype=bool) if retention_mask is None else np.asarray(retention_mask, dtype=bool).copy()
        retain[~retain_mask] = 0.0
        self.retention, self.retention_mask = torch.from_numpy(retain), torch.from_numpy(retain_mask)

    def __len__(self) -> int:
        return len(self.ldaps)

    def __getitem__(self, index: int):
        return (
            self.ldaps[index], self.gfs[index], self.common[index], self.group[index], self.hub[index],
            self.target[index], self.label_mask[index], self.retention[index], self.retention_mask[index],
        )


def write_stage2_feature_schema(path: Path, variant: str, selected_indices: tuple[int, ...]) -> dict:
    selected = [STAGE2_HUB_FEATURES[index] for index in selected_indices]
    assert_stage2_feature_schema(selected)
    payload = {
        "variant": variant,
        "raw_representation": "Exp04 raw_hybrid_gated",
        "all_hub_features": list(STAGE2_HUB_FEATURES),
        "selected_hub_features": selected,
        "stage1_target_availability_feature": False,
        "target_or_target_lag_feature": False,
        "ratio_small_denominator": "NaN then fold-train median imputation",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
