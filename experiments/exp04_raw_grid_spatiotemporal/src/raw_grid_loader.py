"""Load issue-aligned LDAPS/GFS grids without spatial aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from baram.data import load_metadata
from baram.features.spatial import _decimal


AVAILABLE_COL = "data_available_kst_dtm"
GRID_COL = "grid_id"

LDAPS_WIND_CHANNELS = [
    "u10", "v10", "u50_mid", "v50_mid", "u50_range", "v50_range",
    "ws10", "ws50_mid", "ws50_maxcomp", "ws50_mincomp",
]
LDAPS_THERMO_CHANNELS = ["t2", "dpt2", "rh2", "surface_pressure", "msl_pressure", "blh"]
GFS_WIND_CHANNELS = [
    "u10", "v10", "u80", "v80", "u100", "v100", "u_pbl", "v_pbl",
    "u850", "v850", "u700", "v700", "gust", "ws10", "ws80", "ws100",
    "ws_pbl", "ws850", "ws700",
]
GFS_THERMO_CHANNELS = [
    "t2", "dpt2", "rh2", "surface_pressure", "msl_pressure", "t850", "rh850",
]

STATIC_CHANNELS = [
    "latitude", "longitude", "surface_height", "latitude_normalized", "longitude_normalized",
    "delta_latitude", "delta_longitude", "distance_km", "height_minus_hub",
    "inverse_distance", "normalized_abs_height_difference",
]
STATIC_DISTANCE_INDEX = STATIC_CHANNELS.index("distance_km")
STATIC_HEIGHT_INDEX = STATIC_CHANNELS.index("normalized_abs_height_difference")


_LDAPS_RAW = {
    TIME_COL, AVAILABLE_COL, GRID_COL, "latitude", "longitude", "surface_0_h",
    "heightAboveGround_10_10u", "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax", "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax", "heightAboveGround_50_50MVmin",
    "heightAboveGround_2_t", "heightAboveGround_2_dpt", "heightAboveGround_2_r",
    "surface_0_sp", "meanSea_0_prmsl", "etc_0_blh",
}
_GFS_RAW = {
    TIME_COL, AVAILABLE_COL, GRID_COL, "latitude", "longitude",
    "heightAboveGround_10_10u", "heightAboveGround_10_10v",
    "heightAboveGround_80_u", "heightAboveGround_80_v",
    "heightAboveGround_100_100u", "heightAboveGround_100_100v",
    "planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v",
    "isobaricInhPa_850_u", "isobaricInhPa_850_v",
    "isobaricInhPa_700_u", "isobaricInhPa_700_v", "surface_0_gust",
    "heightAboveGround_2_2t", "heightAboveGround_2_2d", "heightAboveGround_2_2r",
    "surface_0_sp", "meanSea_0_prmsl", "isobaricInhPa_850_t", "isobaricInhPa_850_r",
}


@dataclass
class RawSourceData:
    source: str
    split: str
    dynamic: np.ndarray
    channel_names: list[str]
    forecast_times: np.ndarray
    issue_times: np.ndarray
    grid_ids: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    surface_height: np.ndarray

    @property
    def wind_channel_count(self) -> int:
        return len(LDAPS_WIND_CHANNELS if self.source == "ldaps" else GFS_WIND_CHANNELS)

    def selected_dynamic(self, use_thermo: bool) -> np.ndarray:
        count = len(self.channel_names) if use_thermo else self.wind_channel_count
        return self.dynamic[..., :count]

    def selected_channels(self, use_thermo: bool) -> list[str]:
        count = len(self.channel_names) if use_thermo else self.wind_channel_count
        return self.channel_names[:count]


@dataclass
class RawGridBundle:
    split: str
    ldaps: RawSourceData
    gfs: RawSourceData
    targets_cf: np.ndarray
    label_mask: np.ndarray
    ldaps_group_static: np.ndarray
    gfs_group_static: np.ndarray

    @property
    def forecast_times(self) -> np.ndarray:
        return self.ldaps.forecast_times

    @property
    def issue_times(self) -> np.ndarray:
        return self.ldaps.issue_times

    def subset(self, indices: np.ndarray) -> "RawGridBundle":
        idx = np.asarray(indices, dtype=int)
        return RawGridBundle(
            self.split,
            RawSourceData(**{**self.ldaps.__dict__, "dynamic": self.ldaps.dynamic[idx],
                             "forecast_times": self.ldaps.forecast_times[idx],
                             "issue_times": self.ldaps.issue_times[idx]}),
            RawSourceData(**{**self.gfs.__dict__, "dynamic": self.gfs.dynamic[idx],
                             "forecast_times": self.gfs.forecast_times[idx],
                             "issue_times": self.gfs.issue_times[idx]}),
            self.targets_cf[idx], self.label_mask[idx],
            self.ldaps_group_static, self.gfs_group_static,
        )


def _wind_speed(u: pd.Series, v: pd.Series) -> np.ndarray:
    return np.hypot(u.to_numpy(dtype=np.float32), v.to_numpy(dtype=np.float32))


def _derive_ldaps(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["u10"] = frame["heightAboveGround_10_10u"]
    out["v10"] = frame["heightAboveGround_10_10v"]
    umax, umin = frame["heightAboveGround_50_50MUmax"], frame["heightAboveGround_50_50MUmin"]
    vmax, vmin = frame["heightAboveGround_50_50MVmax"], frame["heightAboveGround_50_50MVmin"]
    out["u50_mid"] = (umax + umin) / 2.0
    out["v50_mid"] = (vmax + vmin) / 2.0
    out["u50_range"] = umax - umin
    out["v50_range"] = vmax - vmin
    out["ws10"] = _wind_speed(out["u10"], out["v10"])
    out["ws50_mid"] = _wind_speed(out["u50_mid"], out["v50_mid"])
    out["ws50_maxcomp"] = _wind_speed(umax, vmax)
    out["ws50_mincomp"] = _wind_speed(umin, vmin)
    for name, raw in zip(
        LDAPS_THERMO_CHANNELS,
        ["heightAboveGround_2_t", "heightAboveGround_2_dpt", "heightAboveGround_2_r",
         "surface_0_sp", "meanSea_0_prmsl", "etc_0_blh"],
    ):
        out[name] = frame[raw]
    return out


def _derive_gfs(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    pairs = {
        "10": ("u10", "v10", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        "80": ("u80", "v80", "heightAboveGround_80_u", "heightAboveGround_80_v"),
        "100": ("u100", "v100", "heightAboveGround_100_100u", "heightAboveGround_100_100v"),
        "pbl": ("u_pbl", "v_pbl", "planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
        "850": ("u850", "v850", "isobaricInhPa_850_u", "isobaricInhPa_850_v"),
        "700": ("u700", "v700", "isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    }
    for _, (u_alias, v_alias, u_name, v_name) in pairs.items():
        out[u_alias], out[v_alias] = frame[u_name], frame[v_name]
    out["gust"] = frame["surface_0_gust"]
    for suffix, (u_alias, v_alias, _, _) in pairs.items():
        ws_alias = "ws_pbl" if suffix == "pbl" else f"ws{suffix}"
        out[ws_alias] = _wind_speed(out[u_alias], out[v_alias])
    for name, raw in zip(
        GFS_THERMO_CHANNELS,
        ["heightAboveGround_2_2t", "heightAboveGround_2_2d", "heightAboveGround_2_2r",
         "surface_0_sp", "meanSea_0_prmsl", "isobaricInhPa_850_t", "isobaricInhPa_850_r"],
    ):
        out[name] = frame[raw]
    return out


def load_raw_source(root: Path, split: str, source: str) -> RawSourceData:
    if split not in {"train", "test"} or source not in {"ldaps", "gfs"}:
        raise ValueError("split/source must be train|test and ldaps|gfs")
    grid_count = 16 if source == "ldaps" else 9
    raw_columns = _LDAPS_RAW if source == "ldaps" else _GFS_RAW
    path = Path(root) / split / f"{source}_{split}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=sorted(raw_columns))
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])
    frame[AVAILABLE_COL] = pd.to_datetime(frame[AVAILABLE_COL])
    frame = frame.sort_values([AVAILABLE_COL, TIME_COL, GRID_COL]).reset_index(drop=True)
    if frame.duplicated([TIME_COL, GRID_COL]).any():
        raise ValueError(f"{source} {split}: duplicate timestamp/grid")
    grid_ids = np.arange(1, grid_count + 1, dtype=np.int16)
    observed = frame.groupby(TIME_COL, sort=False)[GRID_COL].agg(tuple)
    expected_tuple = tuple(int(value) for value in grid_ids)
    if not observed.map(lambda values: values == expected_tuple).all():
        raise ValueError(f"{source} {split}: grid order/schema differs across timestamps")
    time_rows = frame[[AVAILABLE_COL, TIME_COL]].drop_duplicates().sort_values([AVAILABLE_COL, TIME_COL])
    block_sizes = time_rows.groupby(AVAILABLE_COL, sort=True).size()
    if not block_sizes.eq(24).all():
        raise ValueError(f"{source} {split}: every issue must contain 24 forecast hours")
    issue_times = time_rows[AVAILABLE_COL].drop_duplicates().to_numpy(dtype="datetime64[ns]")
    expected_blocks = 1096 if split == "train" else 365
    if len(issue_times) != expected_blocks:
        raise ValueError(f"{source} {split}: expected {expected_blocks} issue blocks, got {len(issue_times)}")
    forecast_times = time_rows[TIME_COL].to_numpy(dtype="datetime64[ns]").reshape(expected_blocks, 24)
    derived = _derive_ldaps(frame) if source == "ldaps" else _derive_gfs(frame)
    channels = LDAPS_WIND_CHANNELS + LDAPS_THERMO_CHANNELS if source == "ldaps" else GFS_WIND_CHANNELS + GFS_THERMO_CHANNELS
    values = derived[channels].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    dynamic = values.reshape(expected_blocks, 24, grid_count, len(channels))
    grid_meta = frame.groupby(GRID_COL, sort=True)[["latitude", "longitude"]].first()
    latitude = grid_meta["latitude"].to_numpy(dtype=np.float32)
    longitude = grid_meta["longitude"].to_numpy(dtype=np.float32)
    surface = (
        frame.groupby(GRID_COL, sort=True)["surface_0_h"].median().to_numpy(dtype=np.float32)
        if source == "ldaps" else np.full(grid_count, np.nan, dtype=np.float32)
    )
    return RawSourceData(source, split, dynamic, channels, forecast_times, issue_times,
                         grid_ids, latitude, longitude, surface)


def _haversine_km(lat: np.ndarray, lon: np.ndarray, centre_lat: float, centre_lon: float) -> np.ndarray:
    radius = 6371.0088
    lat_rad, centre_lat_rad = np.radians(lat), np.radians(centre_lat)
    dlat = lat_rad - centre_lat_rad
    dlon = np.radians(lon - centre_lon)
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(centre_lat_rad) * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def _idw_height(target_lat: np.ndarray, target_lon: np.ndarray, source: RawSourceData) -> np.ndarray:
    result = []
    for lat, lon in zip(target_lat, target_lon):
        distance = _haversine_km(source.latitude, source.longitude, float(lat), float(lon))
        weights = 1.0 / np.maximum(distance, 0.1) ** 2
        result.append(float(np.sum(weights * source.surface_height) / np.sum(weights)))
    return np.asarray(result, dtype=np.float32)


def _metadata_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy()
    result["KPX그룹"] = result["KPX그룹"].ffill().astype(int)
    coords = result["좌표(Google)"].map(_decimal)
    result["group_latitude"] = [value[0] for value in coords]
    result["group_longitude"] = [value[1] for value in coords]
    return result


def build_group_static(
    source: RawSourceData, metadata: pd.DataFrame, ldaps_reference: RawSourceData
) -> np.ndarray:
    meta = _metadata_groups(metadata)
    centres = meta.groupby("KPX그룹")[["group_latitude", "group_longitude"]].mean()
    hubs = meta.groupby("KPX그룹")["Hub Height(m)"].mean()
    height = source.surface_height.copy()
    if not np.isfinite(height).all():
        height = _idw_height(source.latitude, source.longitude, ldaps_reference)
    lat_scale = max(float(np.std(source.latitude)), 1e-6)
    lon_scale = max(float(np.std(source.longitude)), 1e-6)
    height_scale = max(float(np.std(height)), 1.0)
    outputs = []
    for group_id in (1, 2, 3):
        centre_lat, centre_lon = centres.loc[group_id]
        delta_lat = source.latitude - float(centre_lat)
        delta_lon = source.longitude - float(centre_lon)
        distance = _haversine_km(source.latitude, source.longitude, float(centre_lat), float(centre_lon))
        height_diff = height - float(hubs.loc[group_id])
        outputs.append(
            np.column_stack(
                [
                    source.latitude, source.longitude, height,
                    (source.latitude - source.latitude.mean()) / lat_scale,
                    (source.longitude - source.longitude.mean()) / lon_scale,
                    delta_lat, delta_lon, distance, height_diff,
                    1.0 / (1.0 + distance), np.abs(height_diff) / height_scale,
                ]
            ).astype(np.float32)
        )
    return np.stack(outputs)


def _aligned_targets(root: Path, forecast_times: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    shape = (*forecast_times.shape, 3)
    if split == "test":
        return np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, dtype=bool)
    labels = pd.read_csv(Path(root) / "train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(pd.DatetimeIndex(forecast_times.reshape(-1)))
    raw = labels[TARGETS].to_numpy(dtype=np.float32).reshape(shape)
    capacity = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=np.float32)
    return raw / capacity, np.isfinite(raw)


def load_raw_grid_bundle(root: Path, split: str) -> RawGridBundle:
    root = Path(root)
    ldaps = load_raw_source(root, split, "ldaps")
    gfs = load_raw_source(root, split, "gfs")
    if not np.array_equal(ldaps.forecast_times, gfs.forecast_times) or not np.array_equal(ldaps.issue_times, gfs.issue_times):
        raise ValueError(f"{split}: LDAPS and GFS issue/timestamp blocks differ")
    metadata = pd.read_excel(root / "info.xlsx", sheet_name="info", header=3).dropna(axis=1, how="all")
    targets, mask = _aligned_targets(root, ldaps.forecast_times, split)
    return RawGridBundle(
        split, ldaps, gfs, targets, mask,
        build_group_static(ldaps, metadata, ldaps),
        build_group_static(gfs, metadata, ldaps),
    )


def channel_manifest(bundle: RawGridBundle) -> dict:
    forbidden = ("scada", "target", "lag")
    all_names = bundle.ldaps.channel_names + bundle.gfs.channel_names + STATIC_CHANNELS
    return {
        "ldaps_wind": LDAPS_WIND_CHANNELS,
        "ldaps_thermodynamic": LDAPS_THERMO_CHANNELS,
        "gfs_wind": GFS_WIND_CHANNELS,
        "gfs_thermodynamic": GFS_THERMO_CHANNELS,
        "static": STATIC_CHANNELS,
        "forbidden_input_matches": [name for name in all_names if any(token in name.lower() for token in forbidden)],
        "gfs_surface_height_source": "inverse-distance interpolation of deterministic LDAPS surface_0_h grid metadata",
    }
