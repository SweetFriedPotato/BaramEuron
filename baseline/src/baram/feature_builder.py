import json
from pathlib import Path

import pandas as pd

from .constants import LABEL_TIME_COL, TARGETS, TIME_COL
from .data import load_gfs, load_labels, load_ldaps, load_metadata
from .features.spatial import group_centres, nearest_features
from .features.time import build_time_features
from .features.weather import summary_features
from .metadata import build_feature_metadata


FEATURE_CACHE_FILES = {
    "train_features": "train_features_raw.parquet",
    "test_features": "test_features_raw.parquet",
    "labels": "train_labels.parquet",
    "metadata": "feature_metadata.json",
}


def feature_cache_dir(config):
    return Path(config["cache_dir"]) / "features"


def label_table(config):
    labels = load_labels(config).rename(columns={LABEL_TIME_COL: TIME_COL})
    return labels[[TIME_COL, *TARGETS]].sort_values(TIME_COL).reset_index(drop=True)


def _build_one(split, config):
    ldaps = load_ldaps(split, config)
    gfs = load_gfs(split, config)
    features = build_time_features(ldaps)
    thermodynamic = config.get("features", {}).get("thermodynamic", True)
    nearest_grid_map = {}

    if config.get("features", {}).get("weather_summary", True):
        for part in (
            summary_features(ldaps, "ldaps", thermodynamic),
            summary_features(gfs, "gfs", thermodynamic),
        ):
            features = features.merge(part, on=TIME_COL, how="inner", validate="one_to_one")

    if config.get("features", {}).get("nearest_grid", True):
        centres = group_centres(load_metadata(config))
        for kind, weather in (("ldaps", ldaps), ("gfs", gfs)):
            part, grid_ids = nearest_features(weather, kind, centres, thermodynamic)
            features = features.merge(part, on=TIME_COL, how="inner", validate="one_to_one")
            nearest_grid_map[kind] = {str(group): int(grid) for group, grid in grid_ids.items()}

    return features.sort_values(TIME_COL).reset_index(drop=True), nearest_grid_map


def _constant_columns(train_features, test_features):
    constants = []
    for col in train_features.columns:
        if col == TIME_COL:
            continue
        train_constant = train_features[col].nunique(dropna=False) <= 1
        test_constant = test_features[col].nunique(dropna=False) <= 1
        if train_constant and test_constant:
            train_value = train_features[col].dropna().iloc[0] if train_features[col].notna().any() else None
            test_value = test_features[col].dropna().iloc[0] if test_features[col].notna().any() else None
            if pd.isna(train_value) and pd.isna(test_value):
                constants.append(col)
            elif train_value == test_value:
                constants.append(col)
    return constants


def validate_feature_tables(train_features, test_features):
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("train/test feature schemas differ")
    if not train_features[TIME_COL].is_unique:
        raise ValueError("train feature timestamps are not unique")
    if not test_features[TIME_COL].is_unique:
        raise ValueError("test feature timestamps are not unique")
    if len(test_features) != 8760:
        raise ValueError(f"test features must have 8,760 rows, got {len(test_features)}")
    lowered = [c.lower() for c in train_features.columns]
    if any("scada" in c for c in lowered):
        raise ValueError("SCADA columns are not allowed in shared features")
    if any(target in train_features.columns or target in test_features.columns for target in TARGETS):
        raise ValueError("target columns are not allowed in raw feature tables")
    if train_features.columns[0] != TIME_COL or test_features.columns[0] != TIME_COL:
        raise ValueError(f"{TIME_COL} must be the first feature table column")


def build_feature_tables(config, force=False):
    cache_dir = feature_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: cache_dir / filename for name, filename in FEATURE_CACHE_FILES.items()}

    if not force and paths["train_features"].exists() and paths["test_features"].exists() and paths["labels"].exists():
        train_features = pd.read_parquet(paths["train_features"])
        test_features = pd.read_parquet(paths["test_features"])
        validate_feature_tables(train_features, test_features)
        return train_features, test_features

    train_features, train_nearest = _build_one("train", config)
    test_features, test_nearest = _build_one("test", config)
    constant_cols = _constant_columns(train_features, test_features)
    if constant_cols:
        train_features = train_features.drop(columns=constant_cols)
        test_features = test_features.drop(columns=constant_cols)
    validate_feature_tables(train_features, test_features)

    labels = label_table(config)
    train_features.to_parquet(paths["train_features"], index=False)
    test_features.to_parquet(paths["test_features"], index=False)
    labels.to_parquet(paths["labels"], index=False)

    metadata = build_feature_metadata(
        train_features,
        test_features,
        config,
        removed_constants=constant_cols,
        nearest_grids={"train": train_nearest, "test": test_nearest},
    )
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_features, test_features


def load_raw_feature_artifacts(config):
    cache_dir = feature_cache_dir(config)
    train_features = pd.read_parquet(cache_dir / FEATURE_CACHE_FILES["train_features"])
    test_features = pd.read_parquet(cache_dir / FEATURE_CACHE_FILES["test_features"])
    labels = pd.read_parquet(cache_dir / FEATURE_CACHE_FILES["labels"])
    validate_feature_tables(train_features, test_features)
    return train_features, test_features, labels


def merge_labels(feature_table, labels):
    return feature_table.merge(labels, on=TIME_COL, how="left", validate="one_to_one")


def target_mask(labeled_table, target):
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    return labeled_table[target].notna()


def get_features_for_group(feature_table, group_id):
    group_prefix = f"group_{int(group_id)}__"
    blocked_group_prefixes = [f"group_{i}__" for i in (1, 2, 3) if i != int(group_id)]
    columns = [
        col for col in feature_table.columns
        if col == TIME_COL
        or not col.startswith("group_")
        or col.startswith(group_prefix)
    ]
    columns = [col for col in columns if not any(col.startswith(prefix) for prefix in blocked_group_prefixes)]
    return feature_table[columns].copy()
