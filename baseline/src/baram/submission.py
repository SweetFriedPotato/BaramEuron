import numpy as np, pandas as pd
from .constants import TARGETS,TIME_COL,CAPACITY_KWH

def postprocess(pred,target,cfg):
    p=np.asarray(pred,dtype=float); p=np.maximum(p,cfg.get("lower_clip",0))
    upper=cfg.get("upper_clip",{})
    if upper.get("enabled",False): p=np.minimum(p,upper.get(target,CAPACITY_KWH[target]))
    return p

def create_submission(sample,predictions,path=None):
    out=sample.copy()
    for t in TARGETS: out[t]=np.asarray(predictions[t],dtype=float)
    if len(out)!=8760 or out[TIME_COL].duplicated().any(): raise ValueError("submission keys invalid")
    if not np.isfinite(out[TARGETS].to_numpy()).all(): raise ValueError("predictions contain NaN/inf")
    if list(out.columns)!=list(sample.columns) or not out[["forecast_id",TIME_COL]].equals(sample[["forecast_id",TIME_COL]]): raise ValueError("submission order changed")
    out[TIME_COL]=out[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    if path: out.to_csv(path,index=False,encoding="utf-8-sig")
    return out

