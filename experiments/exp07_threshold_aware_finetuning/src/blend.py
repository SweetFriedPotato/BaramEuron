"""OOF-only global blend search across original and fine-tuned components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import TIME_COL

from .evaluate import score_column


KEYS = ["quarter", TIME_COL, "target", "group_id"]


def blend_prediction(exp03: np.ndarray, raw: np.ndarray, raw_weight: float) -> np.ndarray:
    if not 0.0 <= raw_weight <= 1.0:
        raise ValueError("raw weight must be within [0, 1]")
    return (1.0 - float(raw_weight)) * np.asarray(exp03) + float(raw_weight) * np.asarray(raw)


def search_global_blends(
    data: pd.DataFrame,
    component_pairs: dict[str, tuple[str, str]],
    weights: list[float] | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = np.round(np.arange(0.0, 0.8001, 0.025), 3) if weights is None else np.asarray(weights)
    rows = []
    for combination, (exp03_column, raw_column) in component_pairs.items():
        for weight in weights:
            frame = data.copy()
            frame["blend_prediction"] = blend_prediction(
                frame[exp03_column].to_numpy(), frame[raw_column].to_numpy(), float(weight)
            )
            score, _ = score_column(frame, "blend_prediction")
            rows.append({"combination": combination, "raw_weight": float(weight), **score})
    search = pd.DataFrame(rows).sort_values(
        ["total_score", "ficr", "one_minus_nmae", "raw_weight"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_frames = []
    for combination, part in search.groupby("combination", sort=False):
        best = part.iloc[0]
        exp03_column, raw_column = component_pairs[str(combination)]
        frame = data.copy()
        frame["blend_prediction"] = blend_prediction(
            frame[exp03_column].to_numpy(), frame[raw_column].to_numpy(),
            float(best["raw_weight"]),
        )
        frame["combination"] = combination
        frame["raw_weight"] = float(best["raw_weight"])
        best_frames.append(frame)
    return search, pd.concat(best_frames, ignore_index=True)
