"""Data contracts and deterministic exp01-selected feature reconstruction."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = PROJECT_ROOT / "baseline" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from baram.config import load_config
from baram.constants import TARGETS, TIME_COL
from baram.data import load_gfs, load_ldaps, load_metadata
from baram.feature_builder import get_features_for_group, load_raw_feature_artifacts
from experiments.exp01_catboost_physics.src.feature_blocks import FeatureBlockPipeline, add_spatial_features


AVAILABLE_COL = "data_available_kst_dtm"
SELECTED_BLOCKS = {
    "spatial": True,
    "wind_physics": True,
    "thermodynamic": True,
    "forecast_disagreement": False,
}
WIND_CONFIG = {
    "alpha_quantiles": [0.01, 0.99],
    "minimum_wind_speed": 0.1,
    "bins_mps": [3, 7, 11, 20],
}
FOLD_WINDOWS = {
    "fold_a": {
        "train_start": "2022-01-01 01:00:00",
        "train_end": "2023-01-01 00:00:00",
        "valid_start": "2023-01-01 01:00:00",
        "valid_end": "2024-01-01 00:00:00",
    },
    "fold_b": {
        "train_start": "2022-01-01 01:00:00",
        "train_end": "2024-01-01 00:00:00",
        "valid_start": "2024-01-01 01:00:00",
        "valid_end": "2025-01-01 00:00:00",
    },
    "full": {
        "train_start": "2022-01-01 01:00:00",
        "train_end": "2025-01-01 00:00:00",
    },
}


def baseline_config() -> dict[str, Any]:
    return load_config(PROJECT_ROOT / "baseline" / "configs" / "preprocessing.yaml")


def issue_mapping(config: dict[str, Any], split: str) -> pd.DataFrame:
    """Return one forecast-to-issue mapping after cross-source equality checks."""
    gfs = load_gfs(split, config)[[TIME_COL, AVAILABLE_COL]].drop_duplicates().sort_values(TIME_COL)
    ldaps = load_ldaps(split, config)[[TIME_COL, AVAILABLE_COL]].drop_duplicates().sort_values(TIME_COL)
    if gfs[TIME_COL].duplicated().any() or ldaps[TIME_COL].duplicated().any():
        raise ValueError("forecast timestamp maps to multiple issue times")
    if not gfs.reset_index(drop=True).equals(ldaps.reset_index(drop=True)):
        raise ValueError("LDAPS/GFS issue mappings differ")
    return gfs.reset_index(drop=True)


def inspect_issue_blocks(mapping: pd.DataFrame, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for issue_time, part in mapping.groupby(AVAILABLE_COL, sort=True):
        times = pd.DatetimeIndex(part[TIME_COL].sort_values())
        diffs = times.to_series().diff().dropna()
        leads = (times - pd.Timestamp(issue_time)).total_seconds() / 3600
        rows.append(
            {
                "split": split,
                AVAILABLE_COL: issue_time,
                "forecast_hours": len(times),
                "first_forecast": times.min(),
                "last_forecast": times.max(),
                "hourly_contiguous": bool(diffs.eq(pd.Timedelta(hours=1)).all()),
                "interval_hours_min": None if diffs.empty else float(diffs.min().total_seconds() / 3600),
                "interval_hours_max": None if diffs.empty else float(diffs.max().total_seconds() / 3600),
                "lead_hours_min": float(leads.min()),
                "lead_hours_max": float(leads.max()),
                "complete": bool(len(times) == 24 and diffs.eq(pd.Timedelta(hours=1)).all()),
            }
        )
    frame = pd.DataFrame(rows)
    incomplete = frame.loc[~frame["complete"]].copy()
    summary = {
        "split": split,
        "issue_blocks": int(len(frame)),
        "complete_issue_blocks": int(frame["complete"].sum()),
        "incomplete_issue_blocks": int((~frame["complete"]).sum()),
        "forecast_hours_distribution": {str(k): int(v) for k, v in frame["forecast_hours"].value_counts().items()},
        "lead_hours_min": float(frame["lead_hours_min"].min()),
        "lead_hours_max": float(frame["lead_hours_max"].max()),
        "all_hourly_contiguous": bool(frame["hourly_contiguous"].all()),
        "all_features_available_at_issue_time": True,
        "label_forecast_alignment": f"labels.kst_dtm == features.{TIME_COL}",
        "non_causal_tcn_allowed": bool(frame["complete"].all()),
    }
    return summary, incomplete


def write_issue_contract(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    incomplete_parts = []
    for split in ("train", "test"):
        summary, incomplete = inspect_issue_blocks(issue_mapping(config, split), split)
        summaries[split] = summary
        incomplete_parts.append(incomplete)
    contract = {
        "train": summaries["train"],
        "test": summaries["test"],
        "non_causal_tcn_allowed": bool(
            summaries["train"]["non_causal_tcn_allowed"] and summaries["test"]["non_causal_tcn_allowed"]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "issue_block_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    incomplete = pd.concat(incomplete_parts, ignore_index=True)
    incomplete.to_csv(output_dir / "incomplete_issue_blocks.csv", index=False)
    return contract


@dataclass
class SelectedFeatureUnionBuilder:
    """Rebuild exp01 selected features once for a three-group neural input."""

    config: dict[str, Any]
    fit_time_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pipelines: dict[int, FeatureBlockPipeline] = {}
        self._train_spatial: dict[int, pd.DataFrame] = {}
        self.feature_columns_: list[str] = []
        self.common_columns_: list[str] = []
        self.group_columns_: dict[int, list[str]] = {}
        self.wind_states_: dict[int, dict[str, Any] | None] = {}

    def fit(self, train_features: pd.DataFrame, fit_time_mask: np.ndarray) -> "SelectedFeatureUnionBuilder":
        if len(fit_time_mask) != len(train_features):
            raise ValueError("fit_time_mask length differs from training features")
        ldaps = load_ldaps("train", self.config)
        gfs = load_gfs("train", self.config)
        metadata = load_metadata(self.config)
        for group_id in (1, 2, 3):
            base = get_features_for_group(train_features, group_id)
            spatial = add_spatial_features(base, ldaps, gfs, metadata, group_id)
            pipeline = FeatureBlockPipeline(SELECTED_BLOCKS, group_id, WIND_CONFIG)
            pipeline.fit(spatial.loc[fit_time_mask])
            self.pipelines[group_id] = pipeline
            self._train_spatial[group_id] = spatial
            self.wind_states_[group_id] = pipeline.wind_state_
        self.fit_time_mask = np.asarray(fit_time_mask, dtype=bool)
        return self

    def _transform_frames(self, split: str, raw_features: pd.DataFrame) -> dict[int, pd.DataFrame]:
        if not self.pipelines:
            raise RuntimeError("SelectedFeatureUnionBuilder must be fit first")
        if split == "train":
            spatial_frames = self._train_spatial
        else:
            ldaps = load_ldaps(split, self.config)
            gfs = load_gfs(split, self.config)
            metadata = load_metadata(self.config)
            spatial_frames = {
                group_id: add_spatial_features(
                    get_features_for_group(raw_features, group_id), ldaps, gfs, metadata, group_id
                )
                for group_id in (1, 2, 3)
            }
        return {
            group_id: self.pipelines[group_id].transform(spatial_frames[group_id])
            for group_id in (1, 2, 3)
        }

    def transform(self, split: str, raw_features: pd.DataFrame) -> pd.DataFrame:
        frames = self._transform_frames(split, raw_features)
        common = [column for column in frames[1].columns if column != TIME_COL and not column.startswith("group_")]
        for group_id in (2, 3):
            other_common = [column for column in frames[group_id].columns if column != TIME_COL and not column.startswith("group_")]
            if common != other_common:
                raise ValueError("common feature schema differs across group reconstruction")
            left = frames[1][common].to_numpy(dtype=float)
            right = frames[group_id][common].to_numpy(dtype=float)
            if not np.allclose(left, right, equal_nan=True):
                raise ValueError("common feature values differ across group reconstruction")

        parts = [frames[1][[TIME_COL]].reset_index(drop=True), frames[1][common].reset_index(drop=True)]
        group_columns = {}
        for group_id in (1, 2, 3):
            prefix = f"group_{group_id}__"
            columns = [column for column in frames[group_id].columns if column.startswith(prefix)]
            parts.append(frames[group_id][columns].reset_index(drop=True))
            group_columns[group_id] = columns
        output = pd.concat(parts, axis=1)
        output[TIME_COL] = pd.to_datetime(output[TIME_COL])
        output = output.replace([np.inf, -np.inf], np.nan)
        self.common_columns_ = common
        self.group_columns_ = group_columns
        self.feature_columns_ = [column for column in output.columns if column != TIME_COL]
        return output

    def fit_transform(self, train_features: pd.DataFrame, fit_time_mask: np.ndarray) -> pd.DataFrame:
        return self.fit(train_features, fit_time_mask).transform("train", train_features)

    def manifest(self, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        if list(train.columns) != list(test.columns):
            raise ValueError("selected train/test feature schemas differ")
        names = [column for column in train.columns if column != TIME_COL]
        lowered = [name.lower() for name in names]
        forbidden = {
            "forecast_disagreement": any("disagreement" in name or "_vs_" in name for name in lowered),
            "scada": any("scada" in name for name in lowered),
            "target": any(name in TARGETS or "target_lag" in name for name in lowered),
        }
        if any(forbidden.values()):
            raise ValueError(f"forbidden feature found: {forbidden}")
        return {
            "selected_blocks": SELECTED_BLOCKS,
            "feature_count": len(names),
            "common_feature_count": len(self.common_columns_),
            "group_feature_counts": {str(group): len(columns) for group, columns in self.group_columns_.items()},
            "common_features": self.common_columns_,
            "group_features": {str(group): columns for group, columns in self.group_columns_.items()},
            "feature_order": names,
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "train_test_schema_equal": list(train.columns) == list(test.columns),
            "train_inf_count": int(np.isinf(train[names].to_numpy(dtype=float)).sum()),
            "test_inf_count": int(np.isinf(test[names].to_numpy(dtype=float)).sum()),
            "forbidden_feature_flags": forbidden,
            "common_features_not_duplicated": True,
            "wind_alpha_states": self.wind_states_,
        }


def raw_artifacts(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_raw_feature_artifacts(config)


def fold_time_mask(times: pd.Series, fold: str, part: str = "train") -> np.ndarray:
    window = FOLD_WINDOWS[fold]
    values = pd.to_datetime(times)
    start = pd.Timestamp(window[f"{part}_start"])
    end = pd.Timestamp(window[f"{part}_end"])
    return ((values >= start) & (values <= end)).to_numpy()
