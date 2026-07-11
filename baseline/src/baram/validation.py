import pandas as pd

def time_split(times,target):
    # Labels denote interval ends: Jan 1 00:00 closes the previous year's last hour.
    t=pd.DatetimeIndex(times); train_start="2023-01-01 01:00:00" if target=="kpx_group_3" else "2022-01-01 01:00:00"
    train=(t>=pd.Timestamp(train_start)) & (t<=pd.Timestamp("2024-01-01 00:00:00"))
    valid=(t>=pd.Timestamp("2024-01-01 01:00:00")) & (t<=pd.Timestamp("2025-01-01 00:00:00"))
    if train.any() and valid.any() and t[train].max()>=t[valid].min(): raise ValueError("validation must be strictly after train")
    return train,valid
