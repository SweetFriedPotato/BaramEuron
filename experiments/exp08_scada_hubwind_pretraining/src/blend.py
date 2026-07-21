"""Leakage-safe three-component convex search with coarse/fine grids."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups


ALIGN_KEYS = ["fold", TIME_COL, "target", "group_id"]


def align_components(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not 2 <= len(components) <= 3:
        raise ValueError("blend search supports two or three components")
    output = None
    for name, frame in components.items():
        columns = ALIGN_KEYS + (["y_true_kwh"] if output is None else []) + ["y_pred_kwh"]
        part = frame[columns].rename(columns={"y_pred_kwh": name})
        output = part if output is None else output.merge(part, on=ALIGN_KEYS, validate="one_to_one")
    if output is None or any(len(frame) != len(output) for frame in components.values()):
        raise ValueError("blend component prediction keys differ")
    return output


def _simplex(step: float, count: int) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    return [tuple(value / units for value in values) for values in itertools.product(range(units + 1), repeat=count) if sum(values) == units]


def _evaluate(aligned: pd.DataFrame, names: list[str], weights: tuple[float, ...]) -> tuple[dict, pd.DataFrame]:
    frame = aligned[ALIGN_KEYS + ["y_true_kwh"]].copy()
    frame["y_pred_kwh"] = sum(float(weight) * aligned[name] for name, weight in zip(names, weights))
    frame["model_id"] = "exp08_convex_blend"
    summary, _ = score_available_groups(frame)
    return summary, frame


def search_convex_blend(components: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_components(components)
    names = list(components)
    rows = []
    for weights in _simplex(0.025, len(names)):
        summary, _ = _evaluate(aligned, names, weights)
        rows.append({**{f"weight_{name}": weight for name, weight in zip(names, weights)}, "stage": "coarse", **summary})
    coarse = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    best = np.asarray([coarse.iloc[0][f"weight_{name}"] for name in names], dtype=float)
    fine_weights = []
    for weights in _simplex(0.005, len(names)):
        if np.max(np.abs(np.asarray(weights) - best)) <= 0.025 + 1e-12:
            fine_weights.append(weights)
    for weights in fine_weights:
        summary, _ = _evaluate(aligned, names, weights)
        rows.append({**{f"weight_{name}": weight for name, weight in zip(names, weights)}, "stage": "fine", **summary})
    search = pd.DataFrame(rows).drop_duplicates([f"weight_{name}" for name in names], keep="last")
    search = search.sort_values(["total_score", *[f"weight_{name}" for name in names]], ascending=[False, *([False] * len(names))]).reset_index(drop=True)
    weights = tuple(float(search.iloc[0][f"weight_{name}"]) for name in names)
    _, prediction = _evaluate(aligned, names, weights)
    for name, weight in zip(names, weights):
        prediction[f"weight_{name}"] = weight
    return search, prediction
