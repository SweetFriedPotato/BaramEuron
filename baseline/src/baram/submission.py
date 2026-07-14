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
    validate_submission_contract(out, sample)
    out[TIME_COL]=out[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    if path: out.to_csv(path,index=False,encoding="utf-8-sig")
    return out

def validate_submission_contract(submission, sample):
    out = submission.copy()
    ref = sample.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    ref[TIME_COL] = pd.to_datetime(ref[TIME_COL])
    if len(out)!=8760 or out[TIME_COL].duplicated().any(): raise ValueError("submission keys invalid")
    if list(out.columns)!=list(ref.columns): raise ValueError("submission columns changed")
    if not out[["forecast_id",TIME_COL]].equals(ref[["forecast_id",TIME_COL]]): raise ValueError("submission order changed")
    if not np.isfinite(out[TARGETS].to_numpy(dtype=float)).all(): raise ValueError("predictions contain NaN/inf")
    for target in TARGETS:
        if not np.issubdtype(out[target].dtype, np.number):
            raise ValueError(f"{target} prediction is not numeric")
    return True
