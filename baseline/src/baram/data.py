from pathlib import Path
import pandas as pd
from .constants import LABEL_TIME_COL, TARGETS, TIME_COL

_CONFIG = None
def configure_data(config):
    global _CONFIG; _CONFIG = config

def _cfg(config=None):
    c = config or _CONFIG
    if c is None: raise ValueError("Data config is required; call configure_data(config) or pass config")
    return c

def _csv(path): return pd.read_csv(path, encoding="utf-8-sig")

def _path(config, section, key):
    return Path(_cfg(config)[section][key])

def _weather(kind, split, config=None):
    c = _cfg(config); split = split.lower()
    if split not in {"train", "test"}: raise ValueError("split must be 'train' or 'test'")
    path = Path(c["data"][f"{split}_dir"]) / f"{kind}_{split}.csv"
    df = _csv(path)
    required = {TIME_COL, "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL]); df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    expected = 16 if kind == "ldaps" else 9
    counts = df.groupby(TIME_COL)["grid_id"].nunique()
    if not counts.eq(expected).all(): raise ValueError(f"{kind} {split}: expected {expected} unique grids per forecast time")
    if df.duplicated([TIME_COL, "grid_id"]).any(): raise ValueError(f"{kind} {split}: duplicate timestamp/grid pairs")
    if split == "test" and df[TIME_COL].nunique() != 8760: raise ValueError(f"{kind} test: expected 8,760 forecast times")
    return df

def load_ldaps(split, config=None): return _weather("ldaps", split, config)
def load_gfs(split, config=None): return _weather("gfs", split, config)

def load_labels(config=None):
    path = Path(_cfg(config)["data"]["train_dir"]) / "train_labels.csv"; df = _csv(path)
    missing = {LABEL_TIME_COL, *TARGETS} - set(df.columns)
    if missing: raise ValueError(f"labels: missing columns {sorted(missing)}")
    df[LABEL_TIME_COL] = pd.to_datetime(df[LABEL_TIME_COL])
    if df[LABEL_TIME_COL].duplicated().any(): raise ValueError("labels: duplicate timestamps")
    return df

def load_metadata(config=None): return pd.read_excel(_cfg(config)["data"]["metadata"], sheet_name="info", header=3).dropna(axis=1, how="all")
def load_sample_submission(config=None):
    df = _csv(_cfg(config)["data"]["sample_submission"])
    required = {"forecast_id", TIME_COL, *TARGETS}
    if required - set(df): raise ValueError(f"submission: missing columns {sorted(required-set(df))}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if len(df) != 8760 or df[TIME_COL].duplicated().any(): raise ValueError("submission must contain 8,760 unique timestamps")
    return df

def validate_periods(config=None):
    lt, lx = load_ldaps("train", config), load_ldaps("test", config)
    gt, gx = load_gfs("train", config), load_gfs("test", config)
    for name, tr, te in [("ldaps",lt,lx),("gfs",gt,gx)]:
        if set(tr[TIME_COL].unique()) & set(te[TIME_COL].unique()): raise ValueError(f"{name}: train/test periods overlap")

def load_data_description(config=None):
    return (_path(config, "data", "root") / "data_description.md").read_text(encoding="utf-8")

def required_weather_columns(kind):
    common = {TIME_COL, "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    if kind == "ldaps":
        return common | {
            "heightAboveGround_10_10u", "heightAboveGround_10_10v",
            "heightAboveGround_50_50MUmax", "heightAboveGround_50_50MUmin",
            "heightAboveGround_50_50MVmax", "heightAboveGround_50_50MVmin",
            "heightAboveGround_2_t", "heightAboveGround_2_dpt", "heightAboveGround_2_r",
            "surface_0_sp", "meanSea_0_prmsl",
        }
    if kind == "gfs":
        return common | {
            "heightAboveGround_10_10u", "heightAboveGround_10_10v",
            "heightAboveGround_80_u", "heightAboveGround_80_v",
            "heightAboveGround_100_100u", "heightAboveGround_100_100v",
            "planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v",
            "isobaricInhPa_850_u", "isobaricInhPa_850_v",
            "isobaricInhPa_700_u", "isobaricInhPa_700_v",
            "isobaricInhPa_500_u", "isobaricInhPa_500_v",
            "surface_0_gust", "heightAboveGround_2_2t", "heightAboveGround_2_2d",
            "heightAboveGround_2_2r", "surface_0_sp", "meanSea_0_prmsl",
        }
    raise ValueError("kind must be 'ldaps' or 'gfs'")

def _time_summary(df, time_col):
    dt = pd.to_datetime(df[time_col])
    parsed_as_na = int(dt.isna().sum())
    diffs = dt.drop_duplicates().sort_values().diff().dropna()
    return {
        "rows": int(len(df)),
        "unique_timestamps": int(dt.nunique()),
        "duplicate_timestamps": int(dt.duplicated().sum()),
        "min": None if dt.empty else str(dt.min()),
        "max": None if dt.empty else str(dt.max()),
        "timezone": str(getattr(dt.dt, "tz", None)),
        "parse_failures": parsed_as_na,
        "hourly_unique_sequence": bool(diffs.eq(pd.Timedelta(hours=1)).all()) if len(diffs) else True,
    }

def _weather_contract(kind, split, config=None):
    df = load_ldaps(split, config) if kind == "ldaps" else load_gfs(split, config)
    expected_grids = 16 if kind == "ldaps" else 9
    grid_counts = df.groupby(TIME_COL)["grid_id"].nunique()
    lead = (df[TIME_COL] - df["data_available_kst_dtm"]).dt.total_seconds() / 3600
    missing_by_column = df.isna().sum()
    required = required_weather_columns(kind)
    missing_rows = df[df.isna().any(axis=1)]
    return {
        **_time_summary(df, TIME_COL),
        "expected_grids_per_timestamp": expected_grids,
        "grid_count_distribution": {str(k): int(v) for k, v in grid_counts.value_counts().sort_index().items()},
        "all_timestamps_have_expected_grid_count": bool(grid_counts.eq(expected_grids).all()),
        "duplicate_timestamp_grid_pairs": int(df.duplicated([TIME_COL, "grid_id"]).sum()),
        "missing_required_columns": sorted(required - set(df.columns)),
        "missing_values_by_column": {k: int(v) for k, v in missing_by_column[missing_by_column > 0].items()},
        "lead_time_h": {
            "min": float(lead.min()),
            "max": float(lead.max()),
            "unique": [float(v) for v in sorted(lead.unique())],
        },
        "data_available_relation": "same data_available_kst_dtm repeats for the 24 forecast hours from 01:00 through next-day 00:00",
        "ldaps_test_missing_locations": (
            [{"forecast_kst_dtm": str(r[TIME_COL]), "grid_id": int(r["grid_id"])}
             for _, r in missing_rows[[TIME_COL, "grid_id"]].drop_duplicates().iterrows()]
            if kind == "ldaps" and split == "test" else []
        ),
    }

def data_contract(config=None):
    labels = load_labels(config)
    sample = load_sample_submission(config)
    meta = load_metadata(config)
    weather = {f"{kind}_{split}": _weather_contract(kind, split, config)
               for split in ("train", "test") for kind in ("ldaps", "gfs")}
    label_missing = labels[TARGETS].isna().sum()
    label_counts_by_year = labels.assign(year=labels[LABEL_TIME_COL].dt.year).groupby("year")[TARGETS].count()
    return {
        "weather": weather,
        "labels": {
            **_time_summary(labels, LABEL_TIME_COL),
            "missing_values_by_target": {k: int(v) for k, v in label_missing.items()},
            "non_null_by_calendar_year": {
                str(year): {target: int(value) for target, value in row.items()}
                for year, row in label_counts_by_year.iterrows()
            },
            "timestamp_meaning": "kst_dtm is the end timestamp of the one-hour generation interval and aligns with forecast_kst_dtm.",
        },
        "sample_submission": {
            **_time_summary(sample, TIME_COL),
            "rows_match_expected": bool(len(sample) == 8760),
            "columns": list(sample.columns),
        },
        "metadata": {
            "rows": int(len(meta)),
            "columns": list(meta.columns),
            "groups": sorted(int(v) for v in meta["KPX그룹"].ffill().dropna().unique()),
        },
        "checks": {
            "train_weather_forecast_times": int(weather["ldaps_train"]["unique_timestamps"]),
            "test_weather_forecast_times": int(weather["ldaps_test"]["unique_timestamps"]),
            "ldaps_train_16_grids": bool(weather["ldaps_train"]["all_timestamps_have_expected_grid_count"]),
            "ldaps_test_16_grids": bool(weather["ldaps_test"]["all_timestamps_have_expected_grid_count"]),
            "gfs_train_9_grids": bool(weather["gfs_train"]["all_timestamps_have_expected_grid_count"]),
            "gfs_test_9_grids": bool(weather["gfs_test"]["all_timestamps_have_expected_grid_count"]),
            "sample_submission_8760_rows": bool(len(sample) == 8760),
            "no_train_test_weather_overlap": not bool(
                set(load_ldaps("train", config)[TIME_COL].unique()) & set(load_ldaps("test", config)[TIME_COL].unique())
            ),
        },
    }

def time_semantics(config=None):
    contract = data_contract(config)
    out = {
        "label_timestamp_semantics": contract["labels"]["timestamp_meaning"],
        "forecast_label_join_key": f"{LABEL_TIME_COL} == {TIME_COL}",
        "timezone": "KST represented as timezone-naive timestamps",
        "weather_lead_times_h": {
            name: block["lead_time_h"] for name, block in contract["weather"].items()
        },
        "periods": {
            "train_weather": {
                "start": contract["weather"]["ldaps_train"]["min"],
                "end": contract["weather"]["ldaps_train"]["max"],
            },
            "test_weather": {
                "start": contract["weather"]["ldaps_test"]["min"],
                "end": contract["weather"]["ldaps_test"]["max"],
            },
            "labels": {
                "start": contract["labels"]["min"],
                "end": contract["labels"]["max"],
            },
            "sample_submission": {
                "start": contract["sample_submission"]["min"],
                "end": contract["sample_submission"]["max"],
            },
        },
        "data_available_kst_dtm_relation": (
            "Forecasts initialized for a weather day become available at 13:00 KST and cover "
            "forecast hours 01:00 through next-day 00:00."
        ),
        "ldaps_test_missing_locations": contract["weather"]["ldaps_test"]["ldaps_test_missing_locations"],
    }
    return out
