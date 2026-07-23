"""Leak-safe feature blocks for the CatBoost physics ablation.

Only fold-train rows determine alpha clipping limits. Spatial weights use turbine
and grid coordinates only; no target or SCADA data is read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from baram.features.spatial import group_centres
from baram.features.weather import add_weather_aliases


EPS = 1e-6
WIND_FEATURES = {
    "ldaps": ["ws10", "ws50_mid", "ws50_maxcomp", "ws50_mincomp"],
    "gfs": ["ws10", "ws80", "ws100", "ws_pbl", "ws850", "ws700", "ws500", "gust"],
}


def _finite_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _haversine_km(lat: np.ndarray, lon: np.ndarray, centre_lat: float, centre_lon: float) -> np.ndarray:
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = np.radians(float(centre_lat))
    lon2 = np.radians(float(centre_lon))
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _distance_weighted_wind(
    weather: pd.DataFrame,
    kind: str,
    group_id: int,
    centre: pd.Series,
    eps: float,
) -> pd.DataFrame:
    data = add_weather_aliases(weather, kind, thermodynamic=False)
    columns = WIND_FEATURES[kind]
    grids = data.groupby("grid_id", sort=False)[["latitude", "longitude"]].first()
    distances = _haversine_km(
        grids["latitude"].to_numpy(), grids["longitude"].to_numpy(), centre["lat"], centre["lon"]
    )
    weights = pd.Series(1.0 / (distances + eps), index=grids.index, name="_spatial_weight")
    row_weights = data["grid_id"].map(weights).astype(float)
    numer = data[columns].mul(row_weights, axis=0).groupby(data[TIME_COL]).sum(min_count=1)
    denom = data[columns].notna().mul(row_weights, axis=0).groupby(data[TIME_COL]).sum()
    out = numer.div(denom.where(denom > 0))
    out.columns = [f"group_{group_id}__{kind}_distance_weighted__{name}" for name in columns]
    return out.reset_index()


def add_spatial_features(
    features: pd.DataFrame,
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    metadata: pd.DataFrame,
    group_id: int,
    eps: float = EPS,
) -> pd.DataFrame:
    """Add dispersion, coordinate-weighted, and nearest-minus-mean wind features."""
    out = features.copy()
    group_id = int(group_id)

    for kind, names in WIND_FEATURES.items():
        for name in names:
            mean = out[f"{kind}__{name}__mean"]
            maximum = out[f"{kind}__{name}__max"]
            minimum = out[f"{kind}__{name}__min"]
            std = out[f"{kind}__{name}__std"]
            stem = f"{kind}__{name}__grid"
            out[f"{stem}_max_minus_min"] = maximum - minimum
            out[f"{stem}_max_minus_mean"] = maximum - mean
            out[f"{stem}_cv_abs_mean"] = std / (mean.abs() + eps)

            nearest_name = f"group_{group_id}__{kind}_nearest__{name}"
            if nearest_name in out:
                out[f"group_{group_id}__{kind}_nearest_minus_grid_mean__{name}"] = out[nearest_name] - mean
                out[f"group_{group_id}__{kind}_nearest_over_grid_mean__{name}"] = out[nearest_name] / mean.where(mean.abs() > eps)

    centres = group_centres(metadata)
    centre = centres.loc[group_id]
    for kind, weather in (("ldaps", ldaps), ("gfs", gfs)):
        weighted = _distance_weighted_wind(weather, kind, group_id, centre, eps)
        out = out.merge(weighted, on=TIME_COL, how="left", validate="one_to_one")
    return _finite_frame(out)


def _wind_pairs(group_id: int) -> list[tuple[str, str, str, float, float]]:
    group_id = int(group_id)
    pairs = [
        ("gfs__ws80__mean", "gfs__ws100__mean", "gfs__ws80_100__mean", 80.0, 100.0),
        ("gfs__ws80__max", "gfs__ws100__max", "gfs__ws80_100__max", 80.0, 100.0),
        ("ldaps__ws10__mean", "ldaps__ws50_mid__mean", "ldaps__ws10_50__mean", 10.0, 50.0),
        ("ldaps__ws10__max", "ldaps__ws50_mid__max", "ldaps__ws10_50__max", 10.0, 50.0),
    ]
    for scope in ("nearest", "distance_weighted"):
        pairs.extend(
            [
                (
                    f"group_{group_id}__gfs_{scope}__ws80",
                    f"group_{group_id}__gfs_{scope}__ws100",
                    f"group_{group_id}__gfs_{scope}__ws80_100",
                    80.0,
                    100.0,
                ),
                (
                    f"group_{group_id}__ldaps_{scope}__ws10",
                    f"group_{group_id}__ldaps_{scope}__ws50_mid",
                    f"group_{group_id}__ldaps_{scope}__ws10_50",
                    10.0,
                    50.0,
                ),
            ]
        )
    return pairs


def _raw_alpha(low: pd.Series, high: pd.Series, low_h: float, high_h: float, minimum: float) -> pd.Series:
    valid = (low > minimum) & (high > minimum)
    alpha = np.log(high.where(valid) / low.where(valid)) / np.log(high_h / low_h)
    return alpha.replace([np.inf, -np.inf], np.nan)


def fit_wind_physics_state(
    train_features: pd.DataFrame,
    group_id: int,
    quantiles: tuple[float, float] = (0.01, 0.99),
    minimum_wind_speed: float = 0.1,
) -> dict[str, Any]:
    """Fit per-feature shear-alpha clipping limits on fold-train rows only."""
    bounds: dict[str, list[float]] = {}
    for low, high, stem, low_h, high_h in _wind_pairs(group_id):
        if low not in train_features or high not in train_features:
            continue
        alpha = _raw_alpha(train_features[low], train_features[high], low_h, high_h, minimum_wind_speed)
        finite = alpha.dropna()
        if finite.empty:
            bounds[stem] = [-1.0, 1.0]
        else:
            bounds[stem] = [float(finite.quantile(quantiles[0])), float(finite.quantile(quantiles[1]))]
    return {
        "alpha_bounds": bounds,
        "alpha_quantiles": [float(quantiles[0]), float(quantiles[1])],
        "minimum_wind_speed": float(minimum_wind_speed),
    }


def _nonlinear_sources(group_id: int) -> list[str]:
    group_id = int(group_id)
    common = [
        "ldaps__ws10__mean", "ldaps__ws10__max", "ldaps__ws50_mid__mean", "ldaps__ws50_mid__max",
        "gfs__ws80__mean", "gfs__ws100__mean", "gfs__ws850__mean", "gfs__gust__mean",
    ]
    scoped = []
    for scope in ("nearest", "distance_weighted"):
        scoped.extend(
            f"group_{group_id}__{kind}_{scope}__{name}"
            for kind, name in [
                ("ldaps", "ws10"), ("ldaps", "ws50_mid"), ("gfs", "ws80"),
                ("gfs", "ws100"), ("gfs", "ws850"), ("gfs", "gust"),
            ]
        )
    return common + scoped


def _bin_features(out: pd.DataFrame, source: str, bins: list[float]) -> None:
    if source not in out:
        return
    values = out[source]
    edges = [-np.inf, *bins, np.inf]
    labels = ["lt3", "ge3_lt7", "ge7_lt11", "ge11_lt20", "ge20"]
    for left, right, label in zip(edges[:-1], edges[1:], labels, strict=True):
        indicator = ((values >= left) & (values < right)).astype(float)
        out[f"{source}__wind_bin_{label}"] = indicator.where(values.notna())


def add_wind_physics_features(
    features: pd.DataFrame,
    group_id: int,
    state: dict[str, Any],
    bins_mps: list[float] | None = None,
) -> pd.DataFrame:
    """Apply hub-height, vertical shear, nonlinear wind, and bin features."""
    out = features.copy()
    minimum = float(state["minimum_wind_speed"])
    hub_sources = []
    for low, high, stem, low_h, high_h in _wind_pairs(group_id):
        if stem not in state["alpha_bounds"] or low not in out or high not in out:
            continue
        alpha = _raw_alpha(out[low], out[high], low_h, high_h, minimum)
        lower, upper = state["alpha_bounds"][stem]
        clipped = alpha.clip(lower, upper)
        alpha_name = f"{stem}__alpha_train_q01_q99"
        out[alpha_name] = clipped
        if stem.startswith("gfs__") or "__gfs_" in stem:
            ws117_name = stem.replace("ws80_100", "ws117_from_80_100")
        else:
            ws117_name = stem.replace("ws10_50", "ws117_from_10_50")
        out[ws117_name] = out[high] * (117.0 / high_h) ** clipped
        hub_sources.append(ws117_name)

    for source in _nonlinear_sources(group_id):
        if source in out:
            out[f"{source}__squared"] = out[source] ** 2
            out[f"{source}__cubed"] = out[source] ** 3

    bins = list(bins_mps or [3.0, 7.0, 11.0, 20.0])
    for source in hub_sources:
        if "gfs" in source and ("__mean" in source or "nearest" in source or "distance_weighted" in source):
            _bin_features(out, source, bins)
    return _finite_frame(out)


def _moist_air_density(temperature_k: pd.Series, relative_humidity_pct: pd.Series, pressure_pa: pd.Series) -> pd.Series:
    """Moist-air density using separate dry-air and water-vapour partial pressures."""
    temperature_c = temperature_k - 273.15
    saturation_vapour_pa = 610.94 * np.exp(17.625 * temperature_c / (temperature_c + 243.04))
    vapour_pa = relative_humidity_pct.clip(0, 100) / 100.0 * saturation_vapour_pa
    dry_air_pa = pressure_pa - vapour_pa
    return dry_air_pa / (287.05 * temperature_k) + vapour_pa / (461.495 * temperature_k)


def add_thermodynamic_features(features: pd.DataFrame, group_id: int) -> pd.DataFrame:
    """Add Celsius, dew-point depression, moist density, and pressure-delta features."""
    out = features.copy()
    group_id = int(group_id)
    scopes = [(kind, f"{kind}__", "__mean") for kind in ("ldaps", "gfs")]
    scopes += [
        (kind, f"group_{group_id}__{kind}_nearest__", "") for kind in ("ldaps", "gfs")
    ]
    for kind, prefix, suffix in scopes:
        temp = f"{prefix}temperature_2m{suffix}"
        dew = f"{prefix}dew_point_2m{suffix}"
        rh = f"{prefix}relative_humidity_2m{suffix}"
        surface = f"{prefix}surface_pressure{suffix}"
        msl = f"{prefix}msl_pressure{suffix}"
        if not all(name in out for name in (temp, dew, rh, surface, msl)):
            continue
        stem = f"{prefix}thermo{suffix}"
        out[f"{stem}__temperature_c"] = out[temp] - 273.15
        out[f"{stem}__dewpoint_c"] = out[dew] - 273.15
        out[f"{stem}__dewpoint_depression_c"] = out[temp] - out[dew]
        out[f"{stem}__moist_air_density_kg_m3"] = _moist_air_density(out[temp], out[rh], out[surface])
        out[f"{stem}__msl_minus_surface_pressure_pa"] = out[msl] - out[surface]

    for kind in ("ldaps", "gfs"):
        for stat in ("max", "min"):
            temp = f"{kind}__temperature_2m__{stat}"
            dew = f"{kind}__dew_point_2m__{stat}"
            surface = f"{kind}__surface_pressure__{stat}"
            msl = f"{kind}__msl_pressure__{stat}"
            out[f"{kind}__thermo__{stat}__temperature_c"] = out[temp] - 273.15
            out[f"{kind}__thermo__{stat}__dewpoint_c"] = out[dew] - 273.15
            out[f"{kind}__thermo__{stat}__msl_minus_surface_pressure_pa"] = out[msl] - out[surface]
    return _finite_frame(out)


def _add_disagreement_triplet(out: pd.DataFrame, left: str, right: str, name: str, eps: float) -> None:
    if left not in out or right not in out:
        return
    out[f"{name}__difference"] = out[left] - out[right]
    out[f"{name}__absolute_difference"] = (out[left] - out[right]).abs()
    out[f"{name}__ratio"] = out[left] / out[right].where(out[right].abs() > eps)


def add_forecast_disagreement_features(features: pd.DataFrame, group_id: int, eps: float = EPS) -> pd.DataFrame:
    """Add aligned LDAPS/GFS wind differences without using any targets."""
    out = features.copy()
    common_pairs = [
        ("ldaps__ws50_mid__mean", "gfs__ws80__mean", "ldaps_ws50_mean_vs_gfs_ws80_mean"),
        ("ldaps__ws50_mid__mean", "gfs__ws100__mean", "ldaps_ws50_mean_vs_gfs_ws100_mean"),
        ("ldaps__ws50_mid__max", "gfs__ws100__max", "ldaps_ws50_max_vs_gfs_ws100_max"),
        ("gfs__ws117_from_80_100__mean", "ldaps__ws117_from_10_50__mean", "gfs_ws117_mean_vs_ldaps_ws117_mean"),
        ("ldaps__ws10__mean", "gfs__ws10__mean", "ldaps_ws10_mean_vs_gfs_ws10_mean"),
    ]
    for left, right, name in common_pairs:
        _add_disagreement_triplet(out, left, right, name, eps)

    group_id = int(group_id)
    for scope in ("nearest", "distance_weighted"):
        prefix = f"group_{group_id}__"
        pairs = [
            (f"{prefix}ldaps_{scope}__ws50_mid", f"{prefix}gfs_{scope}__ws80", f"{prefix}{scope}__ldaps_ws50_vs_gfs_ws80"),
            (f"{prefix}ldaps_{scope}__ws50_mid", f"{prefix}gfs_{scope}__ws100", f"{prefix}{scope}__ldaps_ws50_vs_gfs_ws100"),
            (f"{prefix}ldaps_{scope}__ws10", f"{prefix}gfs_{scope}__ws10", f"{prefix}{scope}__ldaps_ws10_vs_gfs_ws10"),
            (
                f"{prefix}gfs_{scope}__ws117_from_80_100",
                f"{prefix}ldaps_{scope}__ws117_from_10_50",
                f"{prefix}{scope}__gfs_ws117_vs_ldaps_ws117",
            ),
        ]
        for left, right, name in pairs:
            _add_disagreement_triplet(out, left, right, name, eps)
    return _finite_frame(out)


@dataclass
class FeatureBlockPipeline:
    """Small fit/transform wrapper that isolates fold-derived feature state."""

    blocks: dict[str, bool]
    group_id: int
    wind_config: dict[str, Any] = field(default_factory=dict)
    wind_state_: dict[str, Any] | None = None

    def fit(self, train_features: pd.DataFrame) -> "FeatureBlockPipeline":
        if self.blocks.get("wind_physics", False):
            quantiles = tuple(self.wind_config.get("alpha_quantiles", [0.01, 0.99]))
            self.wind_state_ = fit_wind_physics_state(
                train_features,
                self.group_id,
                quantiles=(float(quantiles[0]), float(quantiles[1])),
                minimum_wind_speed=float(self.wind_config.get("minimum_wind_speed", 0.1)),
            )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        if self.blocks.get("wind_physics", False):
            if self.wind_state_ is None:
                raise RuntimeError("FeatureBlockPipeline must be fit before wind transform")
            out = add_wind_physics_features(
                out,
                self.group_id,
                self.wind_state_,
                bins_mps=self.wind_config.get("bins_mps", [3, 7, 11, 20]),
            )
        if self.blocks.get("thermodynamic", False):
            out = add_thermodynamic_features(out, self.group_id)
        if self.blocks.get("forecast_disagreement", False):
            out = add_forecast_disagreement_features(out, self.group_id)
        return _finite_frame(out)

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_features).transform(train_features)