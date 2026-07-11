import numpy as np
from sklearn.metrics import mean_absolute_error
from .constants import CAPACITY_KWH

def regression_metrics(y,p,target):
    value=float(mean_absolute_error(y,p)); return {"mae":value,"nmae":value/CAPACITY_KWH[target]}

def detailed_metrics(times,y,p,target):
    import pandas as pd
    d=pd.DataFrame({"time":times,"y":np.asarray(y),"p":np.asarray(p)}); d["ae"]=(d.y-d.p).abs()
    out=regression_metrics(y,p,target)
    out["monthly_mae"]={str(k):float(v) for k,v in d.groupby(d.time.dt.month).ae.mean().items()}
    out["hourly_mae"]={str(k):float(v) for k,v in d.groupby(d.time.dt.hour).ae.mean().items()}
    return out

