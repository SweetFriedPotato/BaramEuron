"""Leakage-safe OOF/test relation, time, and raw-weather stacker features."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_loader import (
    STATIC_DISTANCE_INDEX,
    RawGridBundle,
    load_raw_grid_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def raw_weather_summary(bundle: RawGridBundle) -> pd.DataFrame:
    ldaps = bundle.ldaps
    gfs = bundle.gfs
    times = bundle.forecast_times.reshape(-1)
    ldaps_values = ldaps.dynamic.reshape(-1, ldaps.dynamic.shape[2], ldaps.dynamic.shape[3])
    gfs_values = gfs.dynamic.reshape(-1, gfs.dynamic.shape[2], gfs.dynamic.shape[3])
    def l(name: str): return ldaps_values[..., ldaps.channel_names.index(name)]
    def g(name: str): return gfs_values[..., gfs.channel_names.index(name)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        result = pd.DataFrame(
            {
                TIME_COL: times,
                "ldaps_ws50_mean": np.nanmean(l("ws50_mid"), axis=1),
                "ldaps_ws50_max": np.nanmax(l("ws50_mid"), axis=1),
                "gfs_ws100_mean": np.nanmean(g("ws100"), axis=1),
                "gfs_ws100_max": np.nanmax(g("ws100"), axis=1),
                "gfs_ws850_mean": np.nanmean(g("ws850"), axis=1),
                "gust_mean": np.nanmean(g("gust"), axis=1),
            }
        )
        pressure = np.nanmean(l("surface_pressure"), axis=1)
        temperature = np.nanmean(l("t2"), axis=1)
    result["air_density"] = pressure / (287.05 * np.maximum(temperature, 150.0))
    wind = l("ws50_mid")
    for group_index, group_id in enumerate((1, 2, 3)):
        distance = bundle.ldaps_group_static[group_index, :, STATIC_DISTANCE_INDEX]
        nearest = int(np.argmin(distance))
        weights = 1.0 / np.maximum(distance, 0.1)
        weights = weights / weights.sum()
        result[f"g{group_id}_nearest_hub_wind"] = wind[:, nearest]
        result[f"g{group_id}_distance_weighted_hub_wind"] = np.nansum(wind * weights[None, :], axis=1)
    return result


def load_weather_summaries(root: Path = PROJECT_ROOT / "open") -> tuple[pd.DataFrame, pd.DataFrame]:
    train = raw_weather_summary(load_raw_grid_bundle(root, "train"))
    test = raw_weather_summary(load_raw_grid_bundle(root, "test"))
    columns = [column for column in train if column != TIME_COL]
    medians = train[columns].replace([np.inf, -np.inf], np.nan).median()
    train[columns] = train[columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
    test[columns] = test[columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
    if not np.isfinite(train[columns].to_numpy()).all() or not np.isfinite(test[columns].to_numpy()).all():
        raise ValueError("train-median weather imputation left NaN/inf")
    return train, test


def _wide_predictions(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for column, prefix in (
        ("exp03_prediction", "exp03"),
        ("raw_prediction", "raw"),
        ("base_prediction", "base"),
    ):
        wide = data.pivot_table(index=TIME_COL, columns="target", values=column, aggfunc="first")
        wide = wide.reindex(columns=TARGETS)
        wide.columns = [f"{prefix}_{target}" for target in TARGETS]
        pieces.append(wide)
    result = pd.concat(pieces, axis=1).reset_index()
    for prefix in ("exp03", "raw", "base"):
        columns = [f"{prefix}_{target}" for target in TARGETS]
        available = result[columns].notna().astype(float)
        fallback = result[columns[:2]].mean(axis=1)
        for column in columns:
            result[column] = result[column].fillna(fallback)
        for index, target in enumerate(TARGETS):
            result[f"{prefix}_{target}_available"] = available.iloc[:, index]
    return result


def build_stacker_features(
    data: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = {TIME_COL, "target", "group_id", "exp03_prediction", "raw_prediction", "base_prediction"}
    if required - set(data):
        raise ValueError(f"stacker input missing columns: {sorted(required-set(data))}")
    frame = data.copy(); frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])
    wide = _wide_predictions(frame)
    frame = frame.merge(wide, on=TIME_COL, how="left", validate="many_to_one")
    weather = weather.copy(); weather[TIME_COL] = pd.to_datetime(weather[TIME_COL])
    frame = frame.merge(weather, on=TIME_COL, how="left", validate="many_to_one")
    hour = frame[TIME_COL].dt.hour.to_numpy(dtype=float)
    month = frame[TIME_COL].dt.month.to_numpy(dtype=float)
    day = frame[TIME_COL].dt.dayofyear.to_numpy(dtype=float)
    frame["hour_sin"], frame["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    frame["month_sin"], frame["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    frame["dayofyear_sin"], frame["dayofyear_cos"] = np.sin(2*np.pi*day/366), np.cos(2*np.pi*day/366)
    if "lead_time_h" not in frame:
        issue = (frame[TIME_COL] - pd.Timedelta(hours=1)).dt.normalize() - pd.Timedelta(hours=11)
        frame["lead_time_h"] = (frame[TIME_COL] - issue).dt.total_seconds() / 3600.0
    for target in TARGETS:
        frame[f"relation_{target}"] = frame[f"raw_{target}"] - frame[f"exp03_{target}"]
        frame[f"base_cf_{target}"] = frame[f"base_{target}"] / float(CAPACITY_KWH[target])
    base_columns = [f"base_{target}" for target in TARGETS]
    frame["farm_mean_prediction"] = frame[base_columns].mean(axis=1)
    for target in TARGETS:
        frame[f"deviation_{target}"] = frame[f"base_{target}"] - frame["farm_mean_prediction"]
    frame["difference_g1_g2"] = frame["base_kpx_group_1"] - frame["base_kpx_group_2"]
    frame["difference_g1_g3"] = frame["base_kpx_group_1"] - frame["base_kpx_group_3"]
    frame["difference_g2_g3"] = frame["base_kpx_group_2"] - frame["base_kpx_group_3"]
    feature_columns = [
        *[f"{prefix}_{target}" for prefix in ("exp03", "raw", "base") for target in TARGETS],
        *[f"{prefix}_{target}_available" for prefix in ("exp03", "raw", "base") for target in TARGETS],
        *[f"relation_{target}" for target in TARGETS],
        *[f"base_cf_{target}" for target in TARGETS],
        "farm_mean_prediction", *[f"deviation_{target}" for target in TARGETS],
        "difference_g1_g2", "difference_g1_g3", "difference_g2_g3",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
        "lead_time_h", "ldaps_ws50_mean", "ldaps_ws50_max", "gfs_ws100_mean", "gfs_ws100_max",
        "gfs_ws850_mean", "gust_mean", "air_density",
        *[f"g{group}_nearest_hub_wind" for group in (1, 2, 3)],
        *[f"g{group}_distance_weighted_hub_wind" for group in (1, 2, 3)],
    ]
    forbidden = [name for name in feature_columns if "target" in name.lower() or "scada" in name.lower() or "lag" in name.lower()]
    if forbidden:
        raise ValueError(f"forbidden stacker features: {forbidden}")
    if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("stacker feature matrix contains NaN/inf")
    return frame, feature_columns


def write_schema(path: Path, columns: list[str]) -> None:
    payload = {
        "feature_columns": columns,
        "feature_count": len(columns),
        "target_or_target_lag_features": [],
        "scada_features": [],
        "seed_std": {"included": False, "reason": "rolling OOF is seed42 only"},
        "raw_source_gate": {
            "included": False,
            "reason": "per-quarter OOF gate dumps unavailable; using full-model gate would leak evaluation labels",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
