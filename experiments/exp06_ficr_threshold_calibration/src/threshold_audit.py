"""Exact official-tier distributions, transitions, and boundary margins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp03_official_score_calibration.src.ficr_surrogate import (
    OFFICIAL_FULL_REWARD_THRESHOLD,
    OFFICIAL_PARTIAL_REWARD_THRESHOLD,
)

from .oof_loader import MODEL_COLUMNS


TIER_ORDER = ["tier_4", "tier_3", "tier_0"]


def tier_from_error(error: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(error, dtype=float)
    return np.select(
        [values <= OFFICIAL_FULL_REWARD_THRESHOLD,
         values <= OFFICIAL_PARTIAL_REWARD_THRESHOLD],
        ["tier_4", "tier_3"], default="tier_0",
    )


def reward_from_error(error: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(error, dtype=float)
    return np.select(
        [values <= OFFICIAL_FULL_REWARD_THRESHOLD,
         values <= OFFICIAL_PARTIAL_REWARD_THRESHOLD],
        [4.0, 3.0], default=0.0,
    )


def tier_frame(data: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    out = data.copy()
    out["normalized_error"] = (
        out[prediction_column].sub(out["y_true_kwh"]).abs() / out["capacity_kwh"]
    )
    out["tier"] = tier_from_error(out["normalized_error"])
    out["reward"] = reward_from_error(out["normalized_error"])
    return out


def tier_distribution(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, column in MODEL_COLUMNS.items():
        frame = tier_frame(data, column).loc[lambda x: x["official_mask"]]
        for keys, part in frame.groupby(["quarter", "target", "group_id"], sort=True):
            counts = part["tier"].value_counts(normalize=True)
            rows.append({
                "model": model, "quarter": keys[0], "target": keys[1], "group_id": keys[2],
                **{f"{tier}_rate": float(counts.get(tier, 0.0)) for tier in TIER_ORDER},
                "mean_normalized_error": float(part["normalized_error"].mean()),
                "median_normalized_error": float(part["normalized_error"].median()),
                "samples": len(part),
            })
    return pd.DataFrame(rows)


def transition_matrix(
    data: pd.DataFrame,
    candidate_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    candidates = candidate_columns or {
        "exp05_ridge": "ridge_prediction",
        "exp05_catboost": "catboost_prediction",
        "exp05_final": "final_prediction",
    }
    base = tier_frame(data, "global_blend_prediction")[[*data.columns, "tier"]].rename(
        columns={"tier": "from_tier"}
    )
    rows = []
    for model, column in candidates.items():
        candidate_tier = tier_from_error(
            data[column].sub(data["y_true_kwh"]).abs() / data["capacity_kwh"]
        )
        frame = base.copy(); frame["to_tier"] = candidate_tier
        frame = frame.loc[frame["official_mask"]]
        for keys, part in frame.groupby(["quarter", "target", "group_id"], sort=True):
            table = part.groupby(["from_tier", "to_tier"]).size()
            for from_tier in TIER_ORDER:
                denominator = int((part["from_tier"] == from_tier).sum())
                for to_tier in TIER_ORDER:
                    cell_mask = part["from_tier"].eq(from_tier) & part["to_tier"].eq(to_tier)
                    count = int(cell_mask.sum())
                    energy = float(part.loc[cell_mask, "y_true_kwh"].sum())
                    reward_value = {"tier_4": 4.0, "tier_3": 3.0, "tier_0": 0.0}
                    rows.append({
                        "candidate": model, "quarter": keys[0], "target": keys[1],
                        "group_id": keys[2], "from_tier": from_tier, "to_tier": to_tier,
                        "count": count, "from_tier_total": denominator,
                        "rate": float(count / denominator) if denominator else 0.0,
                        "actual_energy_kwh": energy,
                        "reward_delta_energy": energy * (reward_value[to_tier]-reward_value[from_tier]),
                    })
    return pd.DataFrame(rows)


def threshold_margin_samples(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for model, column in MODEL_COLUMNS.items():
        frame = tier_frame(data, column).loc[lambda x: x["official_mask"]].copy()
        error = frame["normalized_error"]
        frame["near_6pct"] = error.between(0.055, 0.065)
        frame["near_8pct"] = error.between(0.075, 0.085)
        frame["within_005_below"] = error.between(0.055, 0.06) | error.between(0.075, 0.08)
        frame["within_005_above"] = error.between(0.06, 0.065, inclusive="right") | error.between(0.08, 0.085, inclusive="right")
        selected = frame.loc[frame[["near_6pct", "near_8pct"]].any(axis=1), [
            "quarter", TIME_COL, "target", "group_id", "capacity_kwh", "y_true_kwh",
            column, "normalized_error", "tier", "near_6pct", "near_8pct",
            "within_005_below", "within_005_above",
        ]].rename(columns={column: "prediction"})
        selected.insert(0, "model", model); parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def write_tier_check(data: pd.DataFrame, path: Path) -> dict:
    observed = {}
    for model, column in MODEL_COLUMNS.items():
        frame = tier_frame(data, column)
        group_ficr = []
        for target, part in frame.loc[frame["official_mask"]].groupby("target", sort=True):
            ficr = float((part["y_true_kwh"] * part["reward"]).sum() / (part["y_true_kwh"] * 4.0).sum())
            group_ficr.append(ficr)
        official, _ = score_available_groups(
            data[[TIME_COL, "target", "group_id", "y_true_kwh", column]].rename(columns={column: "y_pred_kwh"})
        )
        observed[model] = {
            "tier_aggregate_ficr": float(np.mean(group_ficr)),
            "official_ficr": official["ficr"],
            "absolute_error": abs(float(np.mean(group_ficr)) - official["ficr"]),
        }
        if observed[model]["absolute_error"] >= 1e-12:
            raise AssertionError(f"tier reward aggregation mismatch for {model}")
    payload = {
        "thresholds": [OFFICIAL_FULL_REWARD_THRESHOLD, OFFICIAL_PARTIAL_REWARD_THRESHOLD],
        "rewards": [4.0, 3.0, 0.0], "models": observed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
