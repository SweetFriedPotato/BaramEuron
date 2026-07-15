from copy import deepcopy
from pathlib import Path
import yaml
from .constants import PROJECT_ROOT

DEFAULTS = {
    "seed": 42,
    "data": {"root": "open", "train_dir": "open/train", "test_dir": "open/test",
             "metadata": "open/info.xlsx", "sample_submission": "open/sample_submission.csv"},
    "features": {"time": True, "weather_summary": True, "nearest_grid": True,
                 "thermodynamic": True, "correlation_selected_grid": False,
                 "distance_weighted_grid": False, "power_curve_features": False,
                 "weather_lags": False},
    "preprocessing": {"tree": {"imputer_strategy": "median", "scale": False},
                      "neural": {"imputer_strategy": "median", "scale": True}},
    "validation": {"group_1_2_train_start": "2022-01-01 01:00:00",
                   "group_1_2_train_end": "2024-01-01 00:00:00",
                   "group_3_train_start": "2023-01-01 01:00:00",
                   "group_3_train_end": "2024-01-01 00:00:00",
                   "valid_start": "2024-01-01 01:00:00",
                   "valid_end": "2025-01-01 00:00:00"},
    "smoke_test": {"model": {"n_estimators": 120, "max_depth": 14,
                             "min_samples_leaf": 8, "max_features": "sqrt",
                             "random_state": 42, "n_jobs": -1},
                   "clip_predictions": True},
    "postprocess": {"lower_clip": 0, "upper_clip": {"enabled": False}},
    "output_root": "outputs", "cache_dir": "baseline/cache",
}

def _merge(a, b):
    out = deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    for section, keys in {"data": ["root", "train_dir", "test_dir", "metadata", "sample_submission"]}.items():
        for key in keys:
            p = Path(cfg[section][key])
            cfg[section][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    for key in ("output_root", "cache_dir"):
        p = Path(cfg[key]); cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg
