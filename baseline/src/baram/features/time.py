import numpy as np
import pandas as pd

from ..constants import TIME_COL


def build_time_features(weather):
    base = (
        weather.groupby(TIME_COL, as_index=False)["data_available_kst_dtm"]
        .first()
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    dt = pd.to_datetime(base[TIME_COL])
    available = pd.to_datetime(base["data_available_kst_dtm"])
    out = pd.DataFrame({TIME_COL: dt})
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["dayofyear"] = dt.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["lead_time_h"] = (dt - available).dt.total_seconds() / 3600
    return out


def time_feature_metadata():
    return {
        "hour": ("time", "forecast hour in KST", "hour", "common"),
        "dayofweek": ("time", "forecast weekday, Monday=0", "index", "common"),
        "month": ("time", "forecast month", "month", "common"),
        "dayofyear": ("time", "forecast day of year", "day", "common"),
        "hour_sin": ("time", "sin(2*pi*hour/24)", "unitless", "common"),
        "hour_cos": ("time", "cos(2*pi*hour/24)", "unitless", "common"),
        "month_sin": ("time", "sin(2*pi*month/12)", "unitless", "common"),
        "month_cos": ("time", "cos(2*pi*month/12)", "unitless", "common"),
        "dayofyear_sin": ("time", "sin(2*pi*dayofyear/365.25)", "unitless", "common"),
        "dayofyear_cos": ("time", "cos(2*pi*dayofyear/365.25)", "unitless", "common"),
        "lead_time_h": ("weather_time", "forecast_kst_dtm - data_available_kst_dtm", "hour", "common"),
    }
