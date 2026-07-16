"""Auditable contracts for raw-grid tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .raw_grid_loader import RawGridBundle, channel_manifest


def _source_contract(bundle: RawGridBundle, source_name: str) -> dict:
    source = getattr(bundle, source_name)
    time_diffs = np.diff(source.forecast_times.astype("datetime64[h]").astype(np.int64), axis=1)
    lead = (
        source.forecast_times.astype("datetime64[h]").astype(np.int64)
        - source.issue_times.astype("datetime64[h]").astype(np.int64)[:, None]
    )
    return {
        "shape": list(source.dynamic.shape),
        "issue_blocks": int(source.dynamic.shape[0]),
        "hours_per_issue": int(source.dynamic.shape[1]),
        "grid_count": int(source.dynamic.shape[2]),
        "channel_count": int(source.dynamic.shape[3]),
        "channels": source.channel_names,
        "grid_ids": source.grid_ids.tolist(),
        "all_hours_contiguous": bool(np.all(time_diffs == 1)),
        "lead_time_min_h": int(lead.min()),
        "lead_time_max_h": int(lead.max()),
        "future_issue_mixed": bool(np.any(lead <= 0)),
        "nan_count": int(np.isnan(source.dynamic).sum()),
        "inf_count": int(np.isinf(source.dynamic).sum()),
    }


def validate_raw_contract(train: RawGridBundle, test: RawGridBundle) -> dict:
    if train.split != "train" or test.split != "test":
        raise ValueError("raw contract expects train and test bundles")
    if train.forecast_times.shape != (1096, 24) or test.forecast_times.shape != (365, 24):
        raise ValueError("raw issue-block shape contract failed")
    checks = {}
    for source_name, grid_count in (("ldaps", 16), ("gfs", 9)):
        train_source, test_source = getattr(train, source_name), getattr(test, source_name)
        checks[f"{source_name}_grid_order_equal"] = bool(np.array_equal(train_source.grid_ids, test_source.grid_ids))
        checks[f"{source_name}_channel_schema_equal"] = train_source.channel_names == test_source.channel_names
        checks[f"{source_name}_coordinate_schema_equal"] = bool(
            np.allclose(train_source.latitude, test_source.latitude)
            and np.allclose(train_source.longitude, test_source.longitude)
        )
        checks[f"{source_name}_grid_count"] = int(grid_count)
    checks["timestamp_label_alignment"] = bool(train.targets_cf.shape[:2] == train.forecast_times.shape)
    checks["scada_or_target_in_input"] = bool(channel_manifest(train)["forbidden_input_matches"])
    checks["train_test_timestamp_overlap"] = bool(
        np.intersect1d(train.forecast_times.reshape(-1), test.forecast_times.reshape(-1)).size
    )
    required_true = [key for key in checks if key.endswith("_equal") or key == "timestamp_label_alignment"]
    if not all(checks[key] for key in required_true):
        raise ValueError(f"raw schema contract failed: {checks}")
    if checks["scada_or_target_in_input"] or checks["train_test_timestamp_overlap"]:
        raise ValueError(f"raw leakage contract failed: {checks}")
    return {
        "train": {name: _source_contract(train, name) for name in ("ldaps", "gfs")},
        "test": {name: _source_contract(test, name) for name in ("ldaps", "gfs")},
        "targets_shape": list(train.targets_cf.shape),
        "label_mask_shape": list(train.label_mask.shape),
        "ldaps_group_static_shape": list(train.ldaps_group_static.shape),
        "gfs_group_static_shape": list(train.gfs_group_static.shape),
        "checks": checks,
    }


def write_raw_contract(train: RawGridBundle, test: RawGridBundle, checks_dir: Path) -> dict:
    checks_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_raw_contract(train, test)
    (checks_dir / "raw_grid_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = channel_manifest(train)
    (checks_dir / "raw_channel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shapes = {
        "ldaps_train": list(train.ldaps.dynamic.shape),
        "gfs_train": list(train.gfs.dynamic.shape),
        "ldaps_test": list(test.ldaps.dynamic.shape),
        "gfs_test": list(test.gfs.dynamic.shape),
        "targets": list(train.targets_cf.shape),
        "label_mask": list(train.label_mask.shape),
        "ldaps_static": list(train.ldaps_group_static.shape),
        "gfs_static": list(train.gfs_group_static.shape),
    }
    (checks_dir / "raw_tensor_shapes.json").write_text(
        json.dumps(shapes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return contract
