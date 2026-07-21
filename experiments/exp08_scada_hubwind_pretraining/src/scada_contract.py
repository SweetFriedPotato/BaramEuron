"""Auditable SCADA source, mapping, access, and timestamp contracts.

SCADA is deliberately isolated in this module and ``scada_hourly_targets``.
Inference modules must never import a loader from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


GROUP_SOURCES = {1: "vestas", 2: "vestas", 3: "unison"}
GROUP_TURBINES = {
    1: tuple(f"vestas_wtg{index:02d}_ws" for index in range(1, 7)),
    2: tuple(f"vestas_wtg{index:02d}_ws" for index in range(7, 13)),
    3: tuple(f"unison_wtg{index:02d}_ws" for index in range(1, 6)),
}
TIME_COLUMN = "kst_dtm"
TARGET_NAMES = ("hub_ws_median", "hub_ws_mean", "hub_ws_std", "hub_ws_iqr")


def scada_path(data_root: Path, source: str) -> Path:
    if source not in {"vestas", "unison"}:
        raise ValueError(f"unknown SCADA source: {source}")
    return Path(data_root) / "train" / f"scada_{source}_train.csv"


def validate_group_mapping(columns_by_source: dict[str, Iterable[str]] | None = None) -> dict:
    all_columns = [column for columns in GROUP_TURBINES.values() for column in columns]
    if len(all_columns) != 17 or len(set(all_columns)) != 17:
        raise ValueError("SCADA group/turbine mapping must contain 17 unique turbines")
    if columns_by_source is not None:
        for group_id, columns in GROUP_TURBINES.items():
            available = set(columns_by_source[GROUP_SOURCES[group_id]])
            missing = set(columns) - available
            if missing:
                raise ValueError(f"group {group_id} SCADA columns missing: {sorted(missing)}")
    return {
        "group_1": list(GROUP_TURBINES[1]),
        "group_2": list(GROUP_TURBINES[2]),
        "group_3": list(GROUP_TURBINES[3]),
    }


def load_scada_wind(data_root: Path, source: str, *, split: str = "train") -> pd.DataFrame:
    """Read SCADA wind columns for target construction only.

    The explicit split guard makes accidental test-time SCADA access fail fast.
    """
    if split != "train":
        raise RuntimeError("SCADA access is forbidden outside the training target pipeline")
    columns = [TIME_COLUMN]
    for group_id, group_source in GROUP_SOURCES.items():
        if group_source == source:
            columns.extend(GROUP_TURBINES[group_id])
    frame = pd.read_csv(scada_path(data_root, source), encoding="utf-8-sig", usecols=columns)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN])
    if frame[TIME_COLUMN].duplicated().any():
        raise ValueError(f"{source} SCADA contains duplicate timestamps")
    return frame.sort_values(TIME_COLUMN).reset_index(drop=True)


def assert_test_pipeline_has_no_scada(paths: Iterable[str | Path]) -> None:
    matches = [str(path) for path in paths if "scada" in str(path).lower()]
    if matches:
        raise ValueError(f"test pipeline references SCADA: {matches}")


def build_source_contract(data_root: Path) -> dict:
    frames = {
        source: load_scada_wind(data_root, source)
        for source in ("vestas", "unison")
    }
    mapping = validate_group_mapping({name: frame.columns for name, frame in frames.items()})
    sources = {}
    for name, frame in frames.items():
        timestamps = frame[TIME_COLUMN]
        minute_values = sorted(timestamps.dt.minute.unique().tolist())
        sources[name] = {
            "path": str(scada_path(data_root, name)),
            "rows": int(len(frame)),
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
            "minute_values": minute_values,
            "ten_minute_cadence_observed": set(minute_values).issubset({0, 10, 20, 30, 40, 50}),
            "wind_columns": [column for column in frame if column.endswith("_ws")],
        }
    return {
        "timestamp_rule": "right-closed hour ending: timestamp.ceil('h')",
        "group_mapping": mapping,
        "sources": sources,
        "wind_direction_targets_used": False,
        "wind_direction_omission_reason": (
            "VESTAS directions use degree-like [0,360) values while UNISON includes negative values; "
            "a shared convention is not established by the supplied metadata"
        ),
        "scada_is_input": False,
        "test_pipeline_reads_scada": False,
    }


def write_source_contract(data_root: Path, output_path: Path, extra: dict | None = None) -> dict:
    payload = build_source_contract(data_root)
    if extra:
        payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
