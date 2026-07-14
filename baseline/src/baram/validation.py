import pandas as pd
from .constants import GROUP_TO_TARGET, TARGETS, TARGET_TO_GROUP, TIME_COL

def time_split(times,target):
    # Labels denote interval ends: Jan 1 00:00 closes the previous year's last hour.
    t=pd.DatetimeIndex(times); train_start="2023-01-01 01:00:00" if target=="kpx_group_3" else "2022-01-01 01:00:00"
    train=(t>=pd.Timestamp(train_start)) & (t<=pd.Timestamp("2024-01-01 00:00:00"))
    valid=(t>=pd.Timestamp("2024-01-01 01:00:00")) & (t<=pd.Timestamp("2025-01-01 00:00:00"))
    if train.any() and valid.any() and t[train].max()>=t[valid].min(): raise ValueError("validation must be strictly after train")
    return train,valid

def time_based_split(table, target, config=None):
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    t = pd.DatetimeIndex(pd.to_datetime(table[TIME_COL]))
    cfg = (config or {}).get("validation", {})
    train_start = cfg.get("group_3_train_start", "2023-01-01 01:00:00") if target == "kpx_group_3" else cfg.get("group_1_2_train_start", "2022-01-01 01:00:00")
    train_end = cfg.get("group_3_train_end", "2024-01-01 00:00:00") if target == "kpx_group_3" else cfg.get("group_1_2_train_end", "2024-01-01 00:00:00")
    valid_start = cfg.get("valid_start", "2024-01-01 01:00:00")
    valid_end = cfg.get("valid_end", "2025-01-01 00:00:00")
    train = (t >= pd.Timestamp(train_start)) & (t <= pd.Timestamp(train_end))
    valid = (t >= pd.Timestamp(valid_start)) & (t <= pd.Timestamp(valid_end))
    if train.any() and valid.any() and t[train].max() >= t[valid].min():
        raise ValueError("validation must be strictly after train")
    return train, valid

def split_labeled_table(labeled_table, target, config=None):
    train, valid = time_based_split(labeled_table, target, config)
    available = labeled_table[target].notna().to_numpy()
    train = train & available
    valid = valid & available
    return train, valid

def validation_split_summary(labeled_table, config=None):
    summary = {}
    for target in TARGETS:
        train, valid = split_labeled_table(labeled_table, target, config)
        train_times = pd.to_datetime(labeled_table.loc[train, TIME_COL])
        valid_times = pd.to_datetime(labeled_table.loc[valid, TIME_COL])
        group = TARGET_TO_GROUP[target]
        summary[target] = {
            "group_id": group,
            "train_rows": int(train.sum()),
            "valid_rows": int(valid.sum()),
            "train_start": None if train_times.empty else str(train_times.min()),
            "train_end": None if train_times.empty else str(train_times.max()),
            "valid_start": None if valid_times.empty else str(valid_times.min()),
            "valid_end": None if valid_times.empty else str(valid_times.max()),
            "validation_after_train": bool(not train_times.empty and not valid_times.empty and train_times.max() < valid_times.min()),
            "excluded_missing_label_rows": int(time_based_split(labeled_table, target, config)[0].sum() - train.sum()
                                             + time_based_split(labeled_table, target, config)[1].sum() - valid.sum()),
        }
        if target == GROUP_TO_TARGET[3]:
            summary[target]["train_has_2022_target"] = bool((train_times.dt.year == 2022).any())
    return summary
