import numpy as np, pandas as pd
from ..constants import TIME_COL

def make_sequences(features, target_times=None, sequence_length=24):
    f=features.sort_values(TIME_COL).reset_index(drop=True); times=pd.DatetimeIndex(f[TIME_COL]); values=f.drop(columns=TIME_COL).to_numpy(dtype=np.float32)
    wanted=times if target_times is None else pd.DatetimeIndex(target_times); pos=pd.Series(np.arange(len(times)),index=times)
    xs,kept=[],[]
    for t in wanted:
        if t not in pos: continue
        i=int(pos[t]); start=i-sequence_length+1
        if start<0: continue
        expected=pd.date_range(times[start],t,freq="h")
        if len(expected)!=sequence_length or not expected.equals(times[start:i+1]): continue
        xs.append(values[start:i+1]); kept.append(t)
    return np.asarray(xs,dtype=np.float32),pd.DatetimeIndex(kept)

