import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from ..constants import TIME_COL, SCHEMA_VERSION
from ..data import load_ldaps, load_gfs, load_metadata
from .weather import summary_features
from .spatial import group_centres, nearest_features

def time_features(ldaps):
    base=ldaps.groupby(TIME_COL,as_index=False)["data_available_kst_dtm"].first(); dt=base[TIME_COL]
    base["hour"]=dt.dt.hour; base["dayofweek"]=dt.dt.dayofweek; base["month"]=dt.dt.month; base["dayofyear"]=dt.dt.dayofyear
    for col,period in [("hour",24),("month",12),("dayofyear",365.25)]:
        base[f"{col}_sin"]=np.sin(2*np.pi*base[col]/period); base[f"{col}_cos"]=np.cos(2*np.pi*base[col]/period)
    base["lead_time_h"]=(base[TIME_COL]-base.pop("data_available_kst_dtm")).dt.total_seconds()/3600
    return base

def _raw_signature(cfg):
    files=[Path(cfg["data"][f"{s}_dir"])/f"{k}_{s}.csv" for s in ("train","test") for k in ("ldaps","gfs")]
    return [{"path":str(p),"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns} for p in files]

def cache_key(cfg):
    payload={"schema":SCHEMA_VERSION,"features":cfg["features"],"raw":_raw_signature(cfg)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16],payload

def _build(split,cfg):
    ld, gf=load_ldaps(split,cfg),load_gfs(split,cfg); f=time_features(ld)
    if cfg["features"].get("weather_summary",True):
        for p in [summary_features(ld,"ldaps",cfg["features"].get("thermodynamic",True)),summary_features(gf,"gfs",cfg["features"].get("thermodynamic",True))]: f=f.merge(p,on=TIME_COL,validate="one_to_one")
    ids={}
    if cfg["features"].get("nearest_grid",True):
        centres=group_centres(load_metadata(cfg))
        for d,k in [(ld,"ldaps"),(gf,"gfs")]:
            p,x=nearest_features(d,k,centres,cfg["features"].get("thermodynamic",True)); f=f.merge(p,on=TIME_COL,validate="one_to_one"); ids[k]=x
    return f.sort_values(TIME_COL).reset_index(drop=True),ids

def build_feature_tables(cfg, force=False):
    key,payload=cache_key(cfg); cache=Path(cfg["cache_dir"]); cache.mkdir(parents=True,exist_ok=True)
    paths={s:cache/f"{s}_features_{key}.parquet" for s in ("train","test")}; meta=cache/f"feature_metadata_{key}.json"
    if not force and all(p.exists() for p in paths.values()): return pd.read_parquet(paths["train"]),pd.read_parquet(paths["test"])
    tr,tr_ids=_build("train",cfg); te,te_ids=_build("test",cfg)
    constant=[c for c in tr.columns if c!=TIME_COL and tr[c].nunique(dropna=False)<=1]
    tr=tr.drop(columns=constant); te=te.drop(columns=[c for c in constant if c in te])
    if list(tr.columns)!=list(te.columns): raise ValueError("train/test feature schemas differ")
    tr.to_parquet(paths["train"],index=False); te.to_parquet(paths["test"],index=False)
    meta.write_text(json.dumps({**payload,"columns":list(tr.columns),"constant_removed":constant,"nearest_grids":tr_ids},ensure_ascii=False,indent=2,default=str))
    return tr,te

