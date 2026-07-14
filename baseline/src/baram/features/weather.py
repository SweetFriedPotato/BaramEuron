from ..constants import TIME_COL
from .wind import (
    GFS_WIND_PAIRS,
    LDAPS_WIND_PAIRS,
    add_wind_features,
    wind_feature_names,
    wind_formulas,
)

LDAPS_THERMO = {
    "temperature_2m": "heightAboveGround_2_t",
    "dew_point_2m": "heightAboveGround_2_dpt",
    "relative_humidity_2m": "heightAboveGround_2_r",
    "surface_pressure": "surface_0_sp",
    "msl_pressure": "meanSea_0_prmsl",
}

GFS_THERMO = {
    "temperature_2m": "heightAboveGround_2_2t",
    "dew_point_2m": "heightAboveGround_2_2d",
    "relative_humidity_2m": "heightAboveGround_2_2r",
    "surface_pressure": "surface_0_sp",
    "msl_pressure": "meanSea_0_prmsl",
}

THERMO = {"ldaps": LDAPS_THERMO, "gfs": GFS_THERMO}
LDAPS_WIND = LDAPS_WIND_PAIRS
GFS_WIND = GFS_WIND_PAIRS


def add_derived(df, kind):
    return add_wind_features(df, kind)


def weather_feature_columns(kind, thermodynamic=True):
    names = wind_feature_names(kind)
    if thermodynamic:
        names += list(THERMO[kind])
    return names


def add_weather_aliases(df, kind, thermodynamic=True):
    out = add_wind_features(df, kind)
    if thermodynamic:
        for alias, source_col in THERMO[kind].items():
            out[alias] = out[source_col]
    return out


def summary_features(df, kind, thermodynamic=True):
    d = add_weather_aliases(df, kind, thermodynamic)
    cols = weather_feature_columns(kind, thermodynamic)
    out = d.groupby(TIME_COL)[cols].agg(["mean", "max", "min", "std"])
    out.columns = [f"{kind}__{feature}__{stat}" for feature, stat in out.columns]
    return out.reset_index()


def weather_feature_metadata(kind, thermodynamic=True):
    formulas = wind_formulas(kind)
    if thermodynamic:
        for alias, source_col in THERMO[kind].items():
            formulas[alias] = source_col
    return {
        name: {
            "source": kind,
            "formula": formulas[name],
            "unit": "m/s" if name.startswith("ws") or name == "gust" else (
                "K" if "temperature" in name or "dew_point" in name else
                "%" if "humidity" in name else "Pa"
            ),
        }
        for name in weather_feature_columns(kind, thermodynamic)
    }
