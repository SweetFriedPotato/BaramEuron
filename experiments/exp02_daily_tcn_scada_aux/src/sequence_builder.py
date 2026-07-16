"""Issue-aligned 24-hour sequence construction and fold slicing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL

from .data_contract import AVAILABLE_COL, FOLD_WINDOWS


@dataclass
class SequenceBundle:
    x: np.ndarray
    y_cf: np.ndarray
    label_mask: np.ndarray
    timestamps: np.ndarray
    issue_times: np.ndarray
    feature_names: list[str]
    aux_wind: np.ndarray | None = None
    aux_mask: np.ndarray | None = None

    def subset(self, indices: np.ndarray) -> "SequenceBundle":
        return replace(
            self,
            x=self.x[indices],
            y_cf=self.y_cf[indices],
            label_mask=self.label_mask[indices],
            timestamps=self.timestamps[indices],
            issue_times=self.issue_times[indices],
            aux_wind=None if self.aux_wind is None else self.aux_wind[indices],
            aux_mask=None if self.aux_mask is None else self.aux_mask[indices],
        )


def build_sequences(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    aux_targets: pd.DataFrame | None = None,
) -> tuple[SequenceBundle, pd.DataFrame]:
    """Build complete issue sequences; incomplete blocks are reported and excluded."""
    feature_names = [column for column in features.columns if column != TIME_COL]
    merged = features.merge(mapping, on=TIME_COL, how="inner", validate="one_to_one")
    if len(merged) != len(features):
        raise ValueError("issue mapping does not cover every feature timestamp")
    if labels is not None:
        merged = merged.merge(labels[[TIME_COL, *TARGETS]], on=TIME_COL, how="left", validate="one_to_one")
    if aux_targets is not None:
        aux_columns = [f"aux_group_{group}" for group in (1, 2, 3)]
        aux_masks = [f"aux_group_{group}_mask" for group in (1, 2, 3)]
        merged = merged.merge(aux_targets[[TIME_COL, *aux_columns, *aux_masks]], on=TIME_COL, how="left", validate="one_to_one")

    xs, ys, masks, timestamp_blocks, issue_times, aux_values, aux_valid = [], [], [], [], [], [], []
    incomplete = []
    capacity = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=np.float32)
    for issue_time, part in merged.groupby(AVAILABLE_COL, sort=True):
        part = part.sort_values(TIME_COL)
        times = pd.DatetimeIndex(part[TIME_COL])
        valid = len(part) == 24 and times.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
        if not valid:
            incomplete.append(
                {
                    AVAILABLE_COL: issue_time,
                    "rows": len(part),
                    "first_forecast": times.min() if len(times) else None,
                    "last_forecast": times.max() if len(times) else None,
                }
            )
            continue
        xs.append(part[feature_names].to_numpy(dtype=np.float32))
        timestamp_blocks.append(times.to_numpy(dtype="datetime64[ns]"))
        issue_times.append(np.datetime64(issue_time, "ns"))
        if labels is None:
            ys.append(np.full((24, 3), np.nan, dtype=np.float32))
            masks.append(np.zeros((24, 3), dtype=bool))
        else:
            raw_y = part[TARGETS].to_numpy(dtype=np.float32)
            ys.append(raw_y / capacity)
            masks.append(np.isfinite(raw_y))
        if aux_targets is not None:
            values = part[[f"aux_group_{group}" for group in (1, 2, 3)]].to_numpy(dtype=np.float32)
            valid_mask = part[[f"aux_group_{group}_mask" for group in (1, 2, 3)]].fillna(False).to_numpy(dtype=bool)
            aux_values.append(values)
            aux_valid.append(valid_mask & np.isfinite(values))

    bundle = SequenceBundle(
        x=np.asarray(xs, dtype=np.float32),
        y_cf=np.asarray(ys, dtype=np.float32),
        label_mask=np.asarray(masks, dtype=bool),
        timestamps=np.asarray(timestamp_blocks, dtype="datetime64[ns]"),
        issue_times=np.asarray(issue_times, dtype="datetime64[ns]"),
        feature_names=feature_names,
        aux_wind=None if aux_targets is None else np.asarray(aux_values, dtype=np.float32),
        aux_mask=None if aux_targets is None else np.asarray(aux_valid, dtype=bool),
    )
    return bundle, pd.DataFrame(incomplete)


def fold_indices(bundle: SequenceBundle, fold: str, part: str) -> np.ndarray:
    window = FOLD_WINDOWS[fold]
    start = np.datetime64(pd.Timestamp(window[f"{part}_start"]), "ns")
    end = np.datetime64(pd.Timestamp(window[f"{part}_end"]), "ns")
    first = bundle.timestamps[:, 0]
    last = bundle.timestamps[:, -1]
    return np.flatnonzero((first >= start) & (last <= end))


def fold_bundle(bundle: SequenceBundle, fold: str, part: str) -> SequenceBundle:
    selected = bundle.subset(fold_indices(bundle, fold, part))
    if fold == "fold_a":
        selected.label_mask[:, :, 2] = False
        if selected.aux_mask is not None:
            selected.aux_mask[:, :, 2] = False
    return selected


def flatten_predictions(bundle: SequenceBundle, prediction_cf: np.ndarray) -> pd.DataFrame:
    if prediction_cf.shape != bundle.y_cf.shape:
        raise ValueError("prediction shape differs from target sequence shape")
    capacity = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=float)
    rows = []
    for group_index, target in enumerate(TARGETS):
        mask = bundle.label_mask[:, :, group_index]
        rows.append(
            pd.DataFrame(
                {
                    TIME_COL: bundle.timestamps[:, :,].reshape(-1)[mask.reshape(-1)],
                    "target": target,
                    "group_id": group_index + 1,
                    "y_true_kwh": (bundle.y_cf[:, :, group_index] * capacity[group_index]).reshape(-1)[mask.reshape(-1)],
                    "y_pred_kwh": (prediction_cf[:, :, group_index] * capacity[group_index]).reshape(-1)[mask.reshape(-1)],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
