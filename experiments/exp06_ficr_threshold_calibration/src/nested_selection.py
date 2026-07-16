"""Acceptance and single-rule final selection for nested Exp06 candidates."""

from __future__ import annotations

import pandas as pd

from experiments.exp05_cross_group_transfer.src.evaluate import rolling_metrics


def summarize_candidate(data: pd.DataFrame, prediction_column: str, model: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, quarters, groups = rolling_metrics(data, prediction_column)
    group3 = groups.loc[groups["group_id"].eq(3), "score"]
    summary.update({"model": model, "group3_score": float(group3.iloc[0]) if len(group3) else float("nan")})
    quarters.insert(0, "model", model); groups.insert(0, "model", model)
    return summary, quarters, groups


def piecewise_acceptance(summary: dict, config: dict, change_p95_fraction: float) -> dict:
    conditions = {
        "aggregate": summary["total_score"] >= float(config["minimum_piecewise_score"]),
        "improved_quarters": summary["improved_quarters"] >= int(config["minimum_improved_quarters"]),
        "worst_quarter": summary["worst_quarter"] >= 0.6054628191969988-float(config["maximum_worst_degradation"]),
        "group3": summary["group3_score"] >= float(config["minimum_group3_score"]),
        "ficr": summary["ficr"] > 0.4217273903596575,
        "nmae": summary["one_minus_nmae"] >= 0.8731518084215217-float(config["maximum_nmae_degradation"]),
        "change_p95": change_p95_fraction <= float(config["maximum_change_p95_fraction"]),
    }
    return {"accepted": all(conditions.values()), "conditions": conditions}


def choose_final(candidates: pd.DataFrame) -> dict:
    ordered = candidates.sort_values(
        ["accepted", "total_score", "equal_quarter_mean", "worst_quarter"],
        ascending=False,
    )
    row = ordered.iloc[0]
    return {"selected_model": row["model"],
            "accepted_new_rule": bool(row["accepted"] and row["model"] != "exp04_global"),
            "selection_table": ordered.to_dict("records")}
