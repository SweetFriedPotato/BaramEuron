import hashlib
import json

from .constants import TIME_COL
from .features.time import time_feature_metadata
from .features.weather import weather_feature_metadata, weather_feature_columns


def config_hash(config):
    relevant = {
        "seed": config.get("seed"),
        "features": config.get("features", {}),
        "preprocessing": config.get("preprocessing", {}),
        "validation": config.get("validation", {}),
        "schema": "preprocessing_v2",
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _feature_scope(name):
    if name == TIME_COL:
        return "key"
    if name.startswith("group_"):
        return "group_nearest"
    return "common"


def _weather_metadata_for_column(name, thermodynamic=True):
    for kind in ("ldaps", "gfs"):
        if name.startswith(f"{kind}__"):
            _, feature, stat = name.split("__")
            base = weather_feature_metadata(kind, thermodynamic)[feature]
            return {
                "source": kind,
                "formula": f"{stat} over grids of {base['formula']}",
                "unit": base["unit"],
            }
        if f"__{kind}_nearest__" in name:
            feature = name.split(f"__{kind}_nearest__", 1)[1]
            base = weather_feature_metadata(kind, thermodynamic)[feature]
            return {
                "source": kind,
                "formula": f"nearest grid value of {base['formula']}",
                "unit": base["unit"],
            }
    return None


def build_feature_metadata(train_features, test_features, config, removed_constants=None, nearest_grids=None):
    h = config_hash(config)
    time_meta = time_feature_metadata()
    thermodynamic = config.get("features", {}).get("thermodynamic", True)
    records = []
    for name in train_features.columns:
        if name == TIME_COL:
            continue
        if name in time_meta:
            source, formula, unit, scope = time_meta[name]
        else:
            weather_meta = _weather_metadata_for_column(name, thermodynamic)
            if weather_meta is None:
                source, formula, unit = "unknown", "unknown", "unknown"
            else:
                source, formula, unit = weather_meta["source"], weather_meta["formula"], weather_meta["unit"]
            scope = _feature_scope(name)
        train_constant = train_features[name].nunique(dropna=False) <= 1
        test_constant = test_features[name].nunique(dropna=False) <= 1
        records.append({
            "name": name,
            "source": source,
            "formula": formula,
            "unit": unit,
            "scope": scope,
            "train_missing_count": int(train_features[name].isna().sum()),
            "test_missing_count": int(test_features[name].isna().sum()),
            "constant": bool(train_constant and test_constant),
            "config_hash": h,
        })
    return {
        "config_hash": h,
        "feature_count": len(records),
        "features": records,
        "removed_constant_columns": removed_constants or [],
        "nearest_grids": nearest_grids or {},
        "notes": {
            "ws50_maxcomp": "sqrt(50MUmax^2 + 50MVmax^2) combines component-wise maxima and is not an observed maximum wind speed.",
            "ws50_mincomp": "sqrt(50MUmin^2 + 50MVmin^2) combines component-wise minima and is not an observed minimum wind speed.",
            "scada": "SCADA is intentionally excluded because no SCADA is available for test.",
            "scaling": "Raw shared feature tables do not include imputation or scaling.",
        },
        "weather_feature_sets": {
            "ldaps": weather_feature_columns("ldaps", thermodynamic),
            "gfs": weather_feature_columns("gfs", thermodynamic),
        },
    }
