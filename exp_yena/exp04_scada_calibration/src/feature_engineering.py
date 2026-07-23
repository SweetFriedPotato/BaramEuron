from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from baram.data import load_gfs, load_ldaps, load_metadata
from baram.features.spatial import group_centres
from exp_yena.exp02_catboost_feature.src.feature_blocks import FeatureBlockPipeline


VECTOR_COLUMNS = {
    "ldaps": {
        "10m": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    },
    "gfs": {
        "10m": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        "80m": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
        "100m": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
        "850hpa": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    },
}


def _nearest_grid(weather: pd.DataFrame, centre: pd.Series) -> int:
    lat = weather.groupby("grid_id")["latitude"].first()
    lon = weather.groupby("grid_id")["longitude"].first()
    distance = (lat - float(centre["lat"])) ** 2 + (lon - float(centre["lon"])) ** 2
    return int(distance.idxmin())


def build_direction_table(split: str, config: dict) -> pd.DataFrame:
    """Build common and group-nearest U/V, speed, direction and sector features."""
    centres = group_centres(load_metadata(config))
    parts: list[pd.DataFrame] = []
    for kind, weather in (("ldaps", load_ldaps(split, config)), ("gfs", load_gfs(split, config))):
        common = weather.groupby(TIME_COL, as_index=False)[
            [column for pair in VECTOR_COLUMNS[kind].values() for column in pair]
        ].mean()
        common_features = pd.DataFrame({TIME_COL: common[TIME_COL]})
        for level, (u_col, v_col) in VECTOR_COLUMNS[kind].items():
            _add_vector_features(common_features, common[u_col], common[v_col], f"{kind}__vector_{level}")
        parts.append(common_features)

        for group_id in (1, 2, 3):
            grid_id = _nearest_grid(weather, centres.loc[group_id])
            nearest = weather.loc[weather["grid_id"] == grid_id].sort_values(TIME_COL)
            group_features = pd.DataFrame({TIME_COL: nearest[TIME_COL].to_numpy()})
            for level, (u_col, v_col) in VECTOR_COLUMNS[kind].items():
                _add_vector_features(
                    group_features,
                    nearest[u_col].reset_index(drop=True),
                    nearest[v_col].reset_index(drop=True),
                    f"group_{group_id}__{kind}_nearest_vector_{level}",
                )
            parts.append(group_features)

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=TIME_COL, how="inner", validate="one_to_one")
    return result.sort_values(TIME_COL).reset_index(drop=True)


def _add_vector_features(out: pd.DataFrame, u: pd.Series, v: pd.Series, prefix: str) -> None:
    u_values = np.asarray(u, dtype=float)
    v_values = np.asarray(v, dtype=float)
    direction = (np.degrees(np.arctan2(-u_values, -v_values)) + 360.0) % 360.0
    out[f"{prefix}__u"] = u_values
    out[f"{prefix}__v"] = v_values
    out[f"{prefix}__speed"] = np.hypot(u_values, v_values)
    out[f"{prefix}__direction_sin"] = np.sin(np.radians(direction))
    out[f"{prefix}__direction_cos"] = np.cos(np.radians(direction))
    out[f"{prefix}__sector_12"] = np.floor((direction + 15.0) % 360.0 / 30.0).astype(int)


def select_group_direction_features(table: pd.DataFrame, group_id: int) -> pd.DataFrame:
    blocked = [f"group_{other}__" for other in (1, 2, 3) if other != group_id]
    columns = [
        column for column in table.columns
        if column == TIME_COL or not any(column.startswith(prefix) for prefix in blocked)
    ]
    return table[columns].copy()


def add_lead_time_interactions(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "lead_time_h" not in out:
        raise ValueError("lead_time_h is required for lead-time interaction experiments")
    lead = out["lead_time_h"]
    out["lead_time_bin_6h"] = (lead // 6).astype(float)
    candidates = [
        "gfs__ws100__mean",
        "gfs__ws850__mean",
        "ldaps__ws50_mid__mean",
        "ldaps_ws50_mean_vs_gfs_ws100_mean__absolute_difference",
        "gfs__ws100__grid_cv_abs_mean",
        "ldaps__ws50_mid__grid_cv_abs_mean",
    ]
    for column in candidates:
        if column in out:
            out[f"{column}__x_lead_time"] = out[column] * lead
    return out


def add_direction_interactions(features: pd.DataFrame, group_id: int) -> pd.DataFrame:
    out = features.copy()
    ldaps = f"group_{group_id}__ldaps_nearest_vector_10m"
    gfs = f"group_{group_id}__gfs_nearest_vector_10m"
    if f"{ldaps}__u" in out and f"{gfs}__u" in out:
        out[f"group_{group_id}__ldaps_gfs_u_difference"] = out[f"{ldaps}__u"] - out[f"{gfs}__u"]
        out[f"group_{group_id}__ldaps_gfs_v_difference"] = out[f"{ldaps}__v"] - out[f"{gfs}__v"]
        dot = out[f"{ldaps}__u"] * out[f"{gfs}__u"] + out[f"{ldaps}__v"] * out[f"{gfs}__v"]
        denom = out[f"{ldaps}__speed"] * out[f"{gfs}__speed"] + 1e-6
        out[f"group_{group_id}__ldaps_gfs_direction_cosine"] = dot / denom
    return out


def build_feature_pipeline(config: dict, group_id: int) -> FeatureBlockPipeline:
    flags = config.get("features", {})
    return FeatureBlockPipeline(
        blocks={
            "wind_physics": flags.get("wind_physics", False),
            "thermodynamic": flags.get("thermodynamic", False),
            "forecast_disagreement": flags.get("forecast_disagreement", False),
            "advanced_meteorology": flags.get("advanced_meteorology", True),
        },
        group_id=group_id,
        wind_config=config.get("wind_physics", {}),
    )
