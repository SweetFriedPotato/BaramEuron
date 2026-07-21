"""Component, threshold-transition, slice, and acceptance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp06_ficr_threshold_calibration.src.threshold_audit import tier_from_error


def score_column(data: pd.DataFrame, prediction_column: str) -> tuple[dict, pd.DataFrame]:
    frame = data[[TIME_COL, "target", "group_id", "y_true_kwh", prediction_column]].rename(
        columns={prediction_column: "y_pred_kwh"}
    )
    return score_available_groups(frame)


def threshold_transitions(
    data: pd.DataFrame,
    base_column: str,
    candidate_column: str,
    candidate: str = "candidate",
) -> pd.DataFrame:
    required = {"capacity_kwh", "y_true_kwh", "official_mask", base_column, candidate_column}
    missing = required - set(data)
    if missing:
        raise ValueError(f"transition columns missing: {sorted(missing)}")
    frame = data.loc[data["official_mask"]].copy()
    base_error = (frame[base_column] - frame["y_true_kwh"]).abs() / frame["capacity_kwh"]
    candidate_error = (frame[candidate_column] - frame["y_true_kwh"]).abs() / frame["capacity_kwh"]
    frame["from_tier"] = tier_from_error(base_error)
    frame["to_tier"] = tier_from_error(candidate_error)
    rows = []
    grouping = [name for name in ("quarter", "target", "group_id") if name in frame]
    for keys, part in frame.groupby(grouping, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = dict(zip(grouping, keys))
        for source in ("tier_4", "tier_3", "tier_0"):
            for destination in ("tier_4", "tier_3", "tier_0"):
                selected = part["from_tier"].eq(source) & part["to_tier"].eq(destination)
                rows.append({
                    "candidate": candidate,
                    **values,
                    "from_tier": source,
                    "to_tier": destination,
                    "count": int(selected.sum()),
                    "actual_energy_kwh": float(part.loc[selected, "y_true_kwh"].sum()),
                })
    return pd.DataFrame(rows)


def rescue_gain(transitions: pd.DataFrame, count_column: str = "count") -> float:
    def total(pairs: set[tuple[str, str]]) -> float:
        selected = transitions.apply(
            lambda row: (row["from_tier"], row["to_tier"]) in pairs, axis=1
        )
        return float(transitions.loc[selected, count_column].sum())
    rescue = total({("tier_0", "tier_3"), ("tier_0", "tier_4"), ("tier_3", "tier_4")})
    loss = total({("tier_4", "tier_3"), ("tier_4", "tier_0"), ("tier_3", "tier_0")})
    return rescue - loss


def boundary_region_scores(
    data: pd.DataFrame,
    base_column: str,
    candidate_column: str,
) -> pd.DataFrame:
    out = []
    base_error = (data[base_column] - data["y_true_kwh"]).abs() / data["capacity_kwh"]
    for name, low, high in (("near_6pct", 0.055, 0.065), ("near_8pct", 0.075, 0.085)):
        selected = data["official_mask"] & base_error.between(low, high)
        part = data.loc[selected]
        for model, column in (("base", base_column), ("candidate", candidate_column)):
            error = (part[column] - part["y_true_kwh"]).abs() / part["capacity_kwh"]
            out.append({
                "region": name,
                "model": model,
                "samples": int(len(part)),
                "mean_normalized_error": float(error.mean()) if len(part) else np.nan,
                "tier4_rate": float((error <= 0.06).mean()) if len(part) else np.nan,
                "rewarded_rate": float((error <= 0.08).mean()) if len(part) else np.nan,
            })
    return pd.DataFrame(out)


def summarize_candidate(
    data: pd.DataFrame,
    prediction_column: str,
    model_id: str,
    reference_column: str = "global_blend_prediction",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, groups = score_column(data, prediction_column)
    quarter_rows = []
    for quarter, part in data.groupby("quarter", sort=True):
        score, _ = score_column(part, prediction_column)
        reference, _ = score_column(part, reference_column)
        quarter_rows.append({"model_id": model_id, "quarter": quarter, **score,
                             "reference_score": reference["total_score"],
                             "score_delta": score["total_score"] - reference["total_score"]})
    quarters = pd.DataFrame(quarter_rows)
    group3 = groups.loc[groups["group_id"].eq(3), "score"]
    result = {
        "model_id": model_id,
        **summary,
        "equal_quarter_mean": float(quarters["total_score"].mean()),
        "worst_quarter": float(quarters["total_score"].min()),
        "improved_quarters": int((quarters["score_delta"] >= -1e-12).sum()),
        "group3_score": float(group3.iloc[0]) if len(group3) else np.nan,
    }
    groups = groups.copy(); groups.insert(0, "model_id", model_id)
    return result, quarters, groups


def acceptance(
    candidate: dict,
    incumbent: dict,
    *,
    rescue: float,
    seed_mean_improved: bool,
    config: dict,
) -> dict:
    conditions = {
        "minimum_score": candidate["total_score"] >= float(config["minimum_score"]),
        "minimum_delta": candidate["total_score"] - incumbent["total_score"] >= float(config["minimum_delta"]),
        "improved_quarters": candidate["improved_quarters"] >= int(config["minimum_improved_quarters"]),
        "worst_quarter": candidate["worst_quarter"] >= incumbent["worst_quarter"] - float(config["maximum_worst_quarter_degradation"]),
        "ficr": candidate["ficr"] > incumbent["ficr"],
        "one_minus_nmae": candidate["one_minus_nmae"] >= incumbent["one_minus_nmae"] - float(config["maximum_one_minus_nmae_degradation"]),
        "group3": candidate["group3_score"] >= incumbent["group3_score"],
        "rescue_gain": rescue > 0,
        "seed_mean": bool(seed_mean_improved),
    }
    return {"accepted": all(conditions.values()), "conditions": conditions}

