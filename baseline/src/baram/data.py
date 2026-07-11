from pathlib import Path
import pandas as pd
from .constants import TARGETS, TIME_COL

_CONFIG = None
def configure_data(config):
    global _CONFIG; _CONFIG = config

def _cfg(config=None):
    c = config or _CONFIG
    if c is None: raise ValueError("Data config is required; call configure_data(config) or pass config")
    return c

def _csv(path): return pd.read_csv(path, encoding="utf-8-sig")

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
    missing = {"kst_dtm", *TARGETS} - set(df.columns)
    if missing: raise ValueError(f"labels: missing columns {sorted(missing)}")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    if df["kst_dtm"].duplicated().any(): raise ValueError("labels: duplicate timestamps")
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

