"""Diagnostic Exp03/raw advantages and strictly nested deployable oracle headroom."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp05_cross_group_transfer.src.constrained_blend import _single_group_score
from experiments.exp05_cross_group_transfer.src.nested_rolling import ORDERED_QUARTERS, assert_nested_order

from .threshold_audit import reward_from_error


def add_regimes(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["pred_cf"] = np.clip(out["global_blend_prediction"] / out["capacity_kwh"], 0.0, 1.2)
    out["pred_cf_band"] = pd.cut(
        out["pred_cf"], [0.0, 0.30, 0.70, 1.200001], labels=["low", "mid", "high"], include_lowest=True
    ).astype(str)
    out["lead_band"] = pd.cut(
        out["lead_time_h"], [11.999, 20, 28, 36], labels=["12-20", "21-28", "29-35"]
    ).astype(str)
    out["season"] = out["month"].map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
                                         6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
    out["disagreement_fraction"] = (
        out["raw_prediction"].sub(out["exp03_prediction"]).abs() / out["capacity_kwh"]
    )
    out["disagreement_band"] = pd.cut(
        out["disagreement_fraction"], [0, .02, .05, .10, 10],
        labels=["0-.02", ".02-.05", ".05-.10", ">.10"], include_lowest=True,
    ).astype(str)
    out["wind_band"] = "unavailable"
    for quarter, indices in out.groupby("quarter").groups.items():
        values = out.loc[indices, "validation_wind_mps_raw"]
        try:
            out.loc[indices, "wind_band"] = pd.qcut(values, 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
        except ValueError:
            out.loc[indices, "wind_band"] = "mid"
    out["raw_source_gate_band"] = "unavailable"
    out["seed_uncertainty_band"] = "unavailable"
    return out


def regime_advantage(data: pd.DataFrame) -> pd.DataFrame:
    frame = add_regimes(data).loc[lambda x: x["official_mask"]].copy()
    exp_error = frame["exp03_prediction"].sub(frame["y_true_kwh"]).abs() / frame["capacity_kwh"]
    raw_error = frame["raw_prediction"].sub(frame["y_true_kwh"]).abs() / frame["capacity_kwh"]
    exp_reward, raw_reward = reward_from_error(exp_error), reward_from_error(raw_error)
    frame["raw_win"] = raw_error < exp_error
    frame["exp03_win"] = exp_error < raw_error
    frame["tie"] = np.isclose(exp_error, raw_error, atol=1e-12)
    frame["raw_sample_contribution_advantage"] = (
        0.5 * (exp_error - raw_error) + 0.5 * (raw_reward - exp_reward) / 4.0
    )
    dimensions = [
        "pred_cf_band", "lead_band", "season", "wind_band", "disagreement_band",
        "raw_source_gate_band", "seed_uncertainty_band",
    ]
    rows = []
    for dimension in dimensions:
        for keys, part in frame.groupby(["target", "group_id", dimension], observed=True, sort=True):
            quarter_rates = part.groupby("quarter")["raw_win"].mean()
            rows.append({
                "regime_dimension": dimension, "regime_value": keys[2],
                "target": keys[0], "group_id": int(keys[1]), "samples": len(part),
                "exp03_win_rate": float(part["exp03_win"].mean()),
                "raw_win_rate": float(part["raw_win"].mean()),
                "tie_rate": float(part["tie"].mean()),
                "average_score_contribution_difference_raw_minus_exp03": float(
                    part["raw_sample_contribution_advantage"].mean()
                ),
                "quarter_raw_win_rate_std": float(quarter_rates.std(ddof=0)),
                "quarters": int(len(quarter_rates)),
            })
    return pd.DataFrame(rows)


def _official_score(data: pd.DataFrame, column: str) -> tuple[dict, pd.DataFrame]:
    frame = data[[TIME_COL, "target", "group_id", "y_true_kwh", column]].rename(
        columns={column: "y_pred_kwh"}
    )
    return score_available_groups(frame)


def oracle_predictions(data: pd.DataFrame) -> pd.DataFrame:
    out = add_regimes(data)
    exp_error = out["exp03_prediction"].sub(out["y_true_kwh"]).abs() / out["capacity_kwh"]
    raw_error = out["raw_prediction"].sub(out["y_true_kwh"]).abs() / out["capacity_kwh"]
    out["nmae_oracle_prediction"] = np.where(
        raw_error < exp_error, out["raw_prediction"], out["exp03_prediction"]
    )
    exp_reward, raw_reward = reward_from_error(exp_error), reward_from_error(raw_error)
    choose_raw = (raw_reward > exp_reward) | ((raw_reward == exp_reward) & (raw_error < exp_error))
    out["ficr_oracle_prediction"] = np.where(
        choose_raw, out["raw_prediction"], out["exp03_prediction"]
    )
    out["deployable_regime_prediction"] = out["global_blend_prediction"]
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(out["quarter"])]
    cell_columns = ["target", "pred_cf_band", "lead_band"]
    for index, quarter in enumerate(quarters):
        evaluation = out.loc[out["quarter"].eq(quarter)]
        if index == 0:
            continue
        fit_quarters = quarters[:index]; assert_nested_order(fit_quarters, quarter)
        fit = out.loc[out["quarter"].isin(fit_quarters)]
        winners: dict[tuple, bool] = {}
        for keys, part in fit.groupby(cell_columns, observed=True, sort=True):
            if len(part.loc[part["official_mask"]]) < 30:
                continue
            exp_score = _single_group_score(part, part["exp03_prediction"].to_numpy(dtype=float))
            raw_score = _single_group_score(part, part["raw_prediction"].to_numpy(dtype=float))
            winners[keys] = bool(raw_score > exp_score)
        for keys, indices in evaluation.groupby(cell_columns, observed=True).groups.items():
            if keys not in winners:
                continue
            column = "raw_prediction" if winners[keys] else "exp03_prediction"
            out.loc[indices, "deployable_regime_prediction"] = out.loc[indices, column]
    return out


def oracle_headroom(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions = oracle_predictions(data)
    candidates = [
        ("exp04_global", "global_blend_prediction"),
        ("sample_nmae_oracle", "nmae_oracle_prediction"),
        ("sample_ficr_oracle", "ficr_oracle_prediction"),
        ("deployable_regime", "deployable_regime_prediction"),
    ]
    reference, reference_groups = _official_score(predictions, "global_blend_prediction")
    rows, group_rows = [], []
    for model, column in candidates:
        score, groups = _official_score(predictions, column)
        rows.append({"slice": "overall", "model": model, **score,
                     "headroom_vs_exp04": score["total_score"] - reference["total_score"]})
        for row in groups.itertuples():
            base = float(reference_groups.loc[reference_groups["target"].eq(row.target), "score"].iloc[0])
            group_rows.append({"model": model, "target": row.target, "group_id": row.group_id,
                               "score": row.score, "headroom_vs_exp04": row.score-base})
        for quarter, part in predictions.groupby("quarter", sort=True):
            value, _ = _official_score(part, column)
            base, _ = _official_score(part, "global_blend_prediction")
            rows.append({"slice": quarter, "model": model, **value,
                         "headroom_vs_exp04": value["total_score"] - base["total_score"]})
    deployable = next(row for row in rows if row["slice"] == "overall" and row["model"] == "deployable_regime")
    return pd.DataFrame(rows), pd.DataFrame(group_rows), {
        "deployable_headroom": float(deployable["headroom_vs_exp04"]),
        "gate_headroom_sufficient": float(deployable["headroom_vs_exp04"]) >= 0.003,
    }
