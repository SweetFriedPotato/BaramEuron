"""Aggregate attention and source gates without using them for model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _attention_rows(source: str, values: np.ndarray, timestamps: np.ndarray, wind: np.ndarray) -> pd.DataFrame:
    blocks, steps, groups, grids = values.shape
    month = pd.DatetimeIndex(timestamps.reshape(-1)).month.to_numpy().reshape(blocks, steps)
    lead = np.arange(12, 12 + steps)
    wind_band = np.where(wind < 4.0, "low", np.where(wind < 8.0, "mid", "high"))
    rows = []
    for group in range(groups):
        for grid in range(grids):
            weight = values[:, :, group, grid]
            rows.append({"source": source, "group_id": group + 1, "grid_id": grid + 1,
                         "mean_attention": float(weight.mean())})
            for value in sorted(np.unique(month)):
                mask = month == value
                rows.append({"source": source, "group_id": group + 1, "grid_id": grid + 1,
                             "month": int(value), "mean_attention": float(weight[mask].mean()),
                             "aggregation": "month"})
            for value in lead:
                index = int(value - 12)
                rows.append({"source": source, "group_id": group + 1, "grid_id": grid + 1,
                             "lead_time_h": int(value), "mean_attention": float(weight[:, index].mean()),
                             "aggregation": "lead_time"})
            for value in ("low", "mid", "high"):
                mask = wind_band == value
                if mask.any():
                    rows.append({"source": source, "group_id": group + 1, "grid_id": grid + 1,
                                 "wind_regime": value, "mean_attention": float(weight[mask].mean()),
                                 "aggregation": "wind_regime"})
    return pd.DataFrame(rows)


def attention_tables(
    diagnostics: dict[str, np.ndarray | None], timestamps: np.ndarray, validation_wind: np.ndarray
) -> dict[str, pd.DataFrame]:
    frames = []
    for source in ("ldaps", "gfs"):
        values = diagnostics.get(f"{source}_attention")
        if values is not None:
            frames.append(_attention_rows(source, values, timestamps, validation_wind))
    all_rows = pd.concat(frames, ignore_index=True)
    group = all_rows.loc[all_rows["aggregation"].isna() if "aggregation" in all_rows else np.ones(len(all_rows), bool)]
    month = all_rows.loc[all_rows.get("aggregation", "").eq("month")]
    lead = all_rows.loc[all_rows.get("aggregation", "").eq("lead_time")]
    wind = all_rows.loc[all_rows.get("aggregation", "").eq("wind_regime")]
    gate = diagnostics.get("source_gate")
    gate_rows = []
    if gate is not None:
        for group_id in range(3):
            for lead_index in range(gate.shape[1]):
                gate_rows.append({
                    "group_id": group_id + 1,
                    "lead_time_h": lead_index + 12,
                    "ldaps_gate_mean": float(gate[:, lead_index, group_id, 0].mean()),
                    "gfs_gate_mean": float(1.0 - gate[:, lead_index, group_id, 0].mean()),
                })
    return {
        "ldaps_group": group.loc[group["source"].eq("ldaps")].dropna(axis=1, how="all"),
        "gfs_group": group.loc[group["source"].eq("gfs")].dropna(axis=1, how="all"),
        "month": month.dropna(axis=1, how="all"),
        "lead_time": lead.dropna(axis=1, how="all"),
        "wind_regime": wind.dropna(axis=1, how="all"),
        "source_gate": pd.DataFrame(gate_rows),
    }
