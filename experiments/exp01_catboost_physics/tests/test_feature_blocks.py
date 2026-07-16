import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp01_catboost_physics.src.feature_blocks import (
    add_forecast_disagreement_features,
    add_thermodynamic_features,
    add_wind_physics_features,
    fit_wind_physics_state,
)


def _wind_frame():
    return pd.DataFrame(
        {
            TIME_COL: pd.date_range("2022-01-01", periods=5, freq="h"),
            "gfs__ws80__mean": [2.0, 3.0, 4.0, 5.0, 6.0],
            "gfs__ws100__mean": [2.5, 3.5, 4.5, 5.5, 20.0],
            "gfs__ws80__max": [3.0, 4.0, 5.0, 6.0, 7.0],
            "gfs__ws100__max": [3.5, 4.5, 5.5, 6.5, 8.0],
            "ldaps__ws10__mean": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ldaps__ws50_mid__mean": [2.0, 3.0, 4.0, 5.0, 6.0],
            "ldaps__ws10__max": [2.0, 3.0, 4.0, 5.0, 6.0],
            "ldaps__ws50_mid__max": [3.0, 4.0, 5.0, 6.0, 7.0],
            "gfs__ws850__mean": [5.0] * 5,
            "gfs__gust__mean": [7.0] * 5,
        }
    )


def test_alpha_clipping_limits_are_fit_on_train_rows_only():
    frame = _wind_frame()
    state = fit_wind_physics_state(frame.iloc[:4], group_id=1, quantiles=(0.01, 0.99))
    transformed = add_wind_physics_features(frame, group_id=1, state=state)
    alpha = transformed["gfs__ws80_100__mean__alpha_train_q01_q99"]
    expected_upper = state["alpha_bounds"]["gfs__ws80_100__mean"][1]
    assert np.isclose(alpha.iloc[-1], expected_upper)
    assert "gfs__ws117_from_80_100__mean" in transformed
    assert not np.isinf(transformed.select_dtypes(include=[np.number]).to_numpy()).any()


def test_thermodynamic_features_use_kelvin_pa_and_moist_air_formula():
    frame = pd.DataFrame(
        {
            TIME_COL: [pd.Timestamp("2022-01-01")],
            "ldaps__temperature_2m__mean": [293.15],
            "ldaps__dew_point_2m__mean": [283.15],
            "ldaps__relative_humidity_2m__mean": [50.0],
            "ldaps__surface_pressure__mean": [100000.0],
            "ldaps__msl_pressure__mean": [101500.0],
            "gfs__temperature_2m__mean": [293.15],
            "gfs__dew_point_2m__mean": [283.15],
            "gfs__relative_humidity_2m__mean": [50.0],
            "gfs__surface_pressure__mean": [100000.0],
            "gfs__msl_pressure__mean": [101500.0],
        }
    )
    for kind in ("ldaps", "gfs"):
        for stat in ("max", "min"):
            frame[f"{kind}__temperature_2m__{stat}"] = 293.15
            frame[f"{kind}__dew_point_2m__{stat}"] = 283.15
            frame[f"{kind}__surface_pressure__{stat}"] = 100000.0
            frame[f"{kind}__msl_pressure__{stat}"] = 101500.0
    transformed = add_thermodynamic_features(frame, group_id=1)
    assert transformed["ldaps__thermo__mean__temperature_c"].iloc[0] == 20.0
    assert transformed["ldaps__thermo__mean__dewpoint_depression_c"].iloc[0] == 10.0
    assert 1.15 < transformed["ldaps__thermo__mean__moist_air_density_kg_m3"].iloc[0] < 1.25
    assert transformed["ldaps__thermo__mean__msl_minus_surface_pressure_pa"].iloc[0] == 1500.0


def test_disagreement_small_denominator_becomes_nan_not_inf():
    frame = pd.DataFrame(
        {
            "ldaps__ws50_mid__mean": [4.0],
            "gfs__ws80__mean": [0.0],
            "gfs__ws100__mean": [5.0],
            "ldaps__ws50_mid__max": [6.0],
            "gfs__ws100__max": [7.0],
            "ldaps__ws10__mean": [2.0],
            "gfs__ws10__mean": [3.0],
        }
    )
    transformed = add_forecast_disagreement_features(frame, group_id=1)
    assert pd.isna(transformed["ldaps_ws50_mean_vs_gfs_ws80_mean__ratio"].iloc[0])
    assert transformed["ldaps_ws50_mean_vs_gfs_ws100_mean__difference"].iloc[0] == -1.0
    assert not np.isinf(transformed.select_dtypes(include=[np.number]).to_numpy()).any()
