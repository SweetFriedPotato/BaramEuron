from __future__ import annotations

from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    base = load_experiment_config(base_path)
    return _merge(base, config)
