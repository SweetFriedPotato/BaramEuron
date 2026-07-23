from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TIME_COL


SCADA_FILES = {
    "vestas": "scada_vestas_train.csv",
    "unison": "scada_unison_train.csv",
}
GROUP_TURBINES = {
    1: ("vestas", range(1, 7)),
    2: ("vestas", range(7, 13)),
    3: ("unison", range(1, 6)),
}


def _column(make: str, turbine: int, suffix: str) -> str:
    return f"{make}_wtg{turbine:02d}_{suffix}"


def load_hourly_scada(config: dict) -> dict[int, pd.DataFrame]:
    """Aggregate ten-minute SCADA observations into label interval-end hours."""
    train_dir = Path(config["data"]["train_dir"])
    raw = {
        make: pd.read_csv(train_dir / filename, encoding="utf-8-sig")
        for make, filename in SCADA_FILES.items()
    }
    output: dict[int, pd.DataFrame] = {}
    for group_id, (make, turbine_numbers) in GROUP_TURBINES.items():
        frame = raw[make].copy()
        frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"])
        frame[TIME_COL] = frame["kst_dtm"].dt.ceil("h")
        ws_cols = [_column(make, number, "ws") for number in turbine_numbers]
        wd_cols = [_column(make, number, "wd") for number in turbine_numbers]
        power_cols = [_column(make, number, "power_kw10m") for number in turbine_numbers]

        turbine_count = len(ws_cols)
        rated_kw = 3600.0 if make == "vestas" else 4200.0
        ws = frame[ws_cols].apply(pd.to_numeric, errors="coerce")
        ws = ws.where((ws >= 0) & (ws <= 40))
        direction_radians = np.radians(frame[wd_cols].apply(pd.to_numeric, errors="coerce"))
        power = frame[power_cols].apply(pd.to_numeric, errors="coerce")
        power = power.where((power >= 0) & (power <= rated_kw))
        valid_power_count = power.notna().sum(axis=1)
        frame["_ws_mean"] = ws.mean(axis=1)
        frame["_ws_std"] = ws.std(axis=1)
        frame["_u_mean"] = np.nanmean(-ws.to_numpy() * np.sin(direction_radians.to_numpy()), axis=1)
        frame["_v_mean"] = np.nanmean(-ws.to_numpy() * np.cos(direction_radians.to_numpy()), axis=1)
        frame["_power_kw"] = power.sum(axis=1, min_count=1) * turbine_count / valid_power_count.replace(0, np.nan)
        frame["_valid_turbines"] = ws.notna().sum(axis=1)

        hourly = frame.groupby(TIME_COL).agg(
            scada_ws_mean=("_ws_mean", "mean"),
            scada_ws_std=("_ws_std", "mean"),
            scada_u_mean=("_u_mean", "mean"),
            scada_v_mean=("_v_mean", "mean"),
            scada_power_kwh=("_power_kw", "mean"),
            scada_valid_turbines=("_valid_turbines", "mean"),
            scada_samples=("kst_dtm", "count"),
        ).reset_index()
        hourly["scada_direction_sin"] = -hourly["scada_u_mean"] / (
            np.hypot(hourly["scada_u_mean"], hourly["scada_v_mean"]) + 1e-6
        )
        hourly["scada_direction_cos"] = -hourly["scada_v_mean"] / (
            np.hypot(hourly["scada_u_mean"], hourly["scada_v_mean"]) + 1e-6
        )
        output[group_id] = hourly
    return output


def scada_target_columns() -> list[str]:
    return ["scada_ws_mean", "scada_ws_std", "scada_u_mean", "scada_v_mean"]


@dataclass
class AuxiliaryScadaModel:
    params: dict
    models_: dict[str, object] | None = None

    def fit(self, features: pd.DataFrame, scada: pd.DataFrame) -> "AuxiliaryScadaModel":
        from catboost import CatBoostRegressor

        self.models_ = {}
        valid = scada[scada_target_columns()].notna().all(axis=1)
        for target in scada_target_columns():
            model_params = dict(self.params)
            model_params.setdefault("iterations", 500)
            model_params.setdefault("depth", 7)
            model_params.setdefault("learning_rate", 0.05)
            model_params.setdefault("loss_function", "RMSE")
            model_params.setdefault("verbose", False)
            model = CatBoostRegressor(**model_params)
            model.fit(features.loc[valid], scada.loc[valid, target])
            self.models_[target] = model
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.models_ is None:
            raise RuntimeError("AuxiliaryScadaModel must be fit before predict")
        return pd.DataFrame({
            f"predicted_{target}": model.predict(features)
            for target, model in self.models_.items()
        }, index=features.index)


def cross_fitted_scada_predictions(
    features: pd.DataFrame,
    scada: pd.DataFrame,
    params: dict,
    blocks: int = 5,
) -> pd.DataFrame:
    """Generate out-of-fold auxiliary predictions without using held-out SCADA targets."""
    columns = [f"predicted_{target}" for target in scada_target_columns()]
    result = pd.DataFrame(index=features.index, columns=columns, dtype=float)
    positions = np.array_split(np.arange(len(features)), blocks)
    for held_out in positions:
        train_positions = np.setdiff1d(np.arange(len(features)), held_out)
        model = AuxiliaryScadaModel(params).fit(
            features.iloc[train_positions], scada.iloc[train_positions]
        )
        result.iloc[held_out, :] = model.predict(features.iloc[held_out]).to_numpy()
    return result.astype(float)


def cross_fitted_offsets(
    features: pd.DataFrame,
    actual_ws: pd.Series,
    forecast_column: str,
    smoothing: float,
    blocks: int = 5,
) -> pd.DataFrame:
    result = pd.DataFrame(index=features.index, columns=["scada_offset", "offset_corrected_ws"], dtype=float)
    positions = np.array_split(np.arange(len(features)), blocks)
    for held_out in positions:
        train_positions = np.setdiff1d(np.arange(len(features)), held_out)
        calibrator = OffsetCalibrator(smoothing=smoothing).fit(
            features.iloc[train_positions], actual_ws.iloc[train_positions], forecast_column
        )
        result.iloc[held_out] = calibrator.transform(
            features.iloc[held_out], forecast_column
        ).to_numpy()
    return result


def cross_fitted_power_curve(
    predicted_ws: pd.Series,
    scada_power: pd.Series,
    bin_width: float,
    smoothing: float,
    blocks: int = 5,
) -> pd.Series:
    result = pd.Series(index=predicted_ws.index, dtype=float)
    positions = np.array_split(np.arange(len(predicted_ws)), blocks)
    for held_out in positions:
        train_positions = np.setdiff1d(np.arange(len(predicted_ws)), held_out)
        curve = EmpiricalPowerCurve(bin_width=bin_width, smoothing=smoothing).fit(
            predicted_ws.iloc[train_positions], scada_power.iloc[train_positions]
        )
        result.iloc[held_out] = curve.predict(predicted_ws.iloc[held_out]).to_numpy()
    return result


@dataclass
class OffsetCalibrator:
    smoothing: float = 48.0
    table_: pd.DataFrame | None = None
    global_offset_: float = 0.0

    def fit(self, features: pd.DataFrame, actual_ws: pd.Series, forecast_column: str) -> "OffsetCalibrator":
        work = pd.DataFrame({
            "month": features["month"].astype(int),
            "lead_bin": (features["lead_time_h"] // 6).astype(int),
            "wind_bin": np.floor(features[forecast_column].clip(0, 30) / 2).astype(int),
            "residual": actual_ws.to_numpy() - features[forecast_column].to_numpy(),
        }).dropna()
        self.global_offset_ = float(work["residual"].mean())
        grouped = work.groupby(["month", "lead_bin", "wind_bin"])["residual"].agg(["mean", "count"]).reset_index()
        grouped["offset"] = (
            grouped["count"] * grouped["mean"] + self.smoothing * self.global_offset_
        ) / (grouped["count"] + self.smoothing)
        self.table_ = grouped[["month", "lead_bin", "wind_bin", "offset"]]
        return self

    def transform(self, features: pd.DataFrame, forecast_column: str) -> pd.DataFrame:
        if self.table_ is None:
            raise RuntimeError("OffsetCalibrator must be fit before transform")
        keys = pd.DataFrame({
            "month": features["month"].astype(int),
            "lead_bin": (features["lead_time_h"] // 6).astype(int),
            "wind_bin": np.floor(features[forecast_column].clip(0, 30) / 2).astype(int),
        }, index=features.index)
        merged = keys.reset_index().merge(self.table_, on=["month", "lead_bin", "wind_bin"], how="left").set_index("index")
        offset = merged["offset"].reindex(features.index).fillna(self.global_offset_)
        return pd.DataFrame({
            "scada_offset": offset,
            "offset_corrected_ws": features[forecast_column] + offset,
        }, index=features.index)


@dataclass
class EmpiricalPowerCurve:
    bin_width: float = 0.5
    smoothing: float = 24.0
    table_: pd.DataFrame | None = None
    global_power_: float = 0.0

    def fit(self, predicted_ws: pd.Series, scada_power: pd.Series) -> "EmpiricalPowerCurve":
        work = pd.DataFrame({"ws_bin": np.floor(predicted_ws / self.bin_width), "power": scada_power}).dropna()
        self.global_power_ = float(work["power"].mean())
        grouped = work.groupby("ws_bin")["power"].agg(["mean", "count"]).reset_index()
        grouped["power_curve"] = (
            grouped["count"] * grouped["mean"] + self.smoothing * self.global_power_
        ) / (grouped["count"] + self.smoothing)
        self.table_ = grouped[["ws_bin", "power_curve"]]
        return self

    def predict(self, predicted_ws: pd.Series) -> pd.Series:
        if self.table_ is None:
            raise RuntimeError("EmpiricalPowerCurve must be fit before predict")
        keys = pd.DataFrame({"ws_bin": np.floor(predicted_ws / self.bin_width)}, index=predicted_ws.index)
        merged = keys.reset_index().merge(self.table_, on="ws_bin", how="left").set_index("index")
        return merged["power_curve"].reindex(predicted_ws.index).fillna(self.global_power_)
