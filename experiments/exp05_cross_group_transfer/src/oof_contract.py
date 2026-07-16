"""Load and verify Exp03/Exp04 rolling OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP03_OUTPUT = PROJECT_ROOT / "experiments/exp03_official_score_calibration/outputs"
EXP04_OUTPUT = PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs"
EXPECTED_GLOBAL_SCORE = 0.6474395993905896
KEYS = ["quarter", TIME_COL, "target", "group_id"]


def assert_prediction_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Require identical timestamp/quarter/target ordering for base predictions."""
    missing = set(KEYS) - set(left) | (set(KEYS) - set(right))
    if missing:
        raise ValueError(f"alignment columns missing: {sorted(missing)}")
    left_keys = left[KEYS].reset_index(drop=True)
    right_keys = right[KEYS].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise ValueError("base prediction timestamp/order mismatch")


def issue_timestamp(forecast: pd.Series) -> pd.Series:
    values = pd.to_datetime(forecast)
    return (values - pd.Timedelta(hours=1)).dt.normalize() - pd.Timedelta(hours=11)


def load_oof_contract(
    exp03_root: Path = EXP03_OUTPUT,
    exp04_root: Path = EXP04_OUTPUT,
) -> pd.DataFrame:
    exp03 = pd.read_csv(
        Path(exp03_root) / "predictions/rolling_retrained_predictions.csv",
        parse_dates=[TIME_COL],
    )
    exp03 = exp03.loc[exp03["experiment_id"].eq("ficr_lambda_02")].copy()
    raw = pd.read_csv(
        Path(exp04_root) / "predictions/rolling_oof_predictions.csv",
        parse_dates=[TIME_COL],
    )
    raw = raw.loc[raw["model_id"].eq("raw_hybrid_gated") & raw["stage"].eq("rolling")].copy()
    if exp03.duplicated(KEYS).any() or raw.duplicated(KEYS).any():
        raise ValueError("rolling OOF contains duplicate quarter/timestamp/target keys")
    assert_prediction_alignment(exp03.sort_values(KEYS), raw.sort_values(KEYS))
    left = exp03[KEYS + ["y_true_kwh", "y_pred_kwh", "validation_wind_mps", "high_wind_mask"]].rename(
        columns={"y_pred_kwh": "exp03_prediction", "validation_wind_mps": "validation_wind_mps_exp03"}
    )
    right = raw[KEYS + ["y_true_kwh", "y_pred_kwh", "validation_wind_mps", "high_wind_mask"]].rename(
        columns={"y_true_kwh": "raw_y_true_kwh", "y_pred_kwh": "raw_prediction",
                 "validation_wind_mps": "validation_wind_mps_raw", "high_wind_mask": "raw_high_wind_mask"}
    )
    merged = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Exp03/raw rolling OOF key sets differ")
    if not np.allclose(merged["y_true_kwh"], merged["raw_y_true_kwh"], atol=0.01, rtol=0):
        raise ValueError("Exp03/raw OOF targets differ beyond serialization tolerance")
    merged["issue_kst_dtm"] = issue_timestamp(merged[TIME_COL])
    merged["lead_time_h"] = (
        merged[TIME_COL] - merged["issue_kst_dtm"]
    ).dt.total_seconds() / 3600.0
    merged["hour"] = merged[TIME_COL].dt.hour
    merged["month"] = merged[TIME_COL].dt.month
    merged["dayofyear"] = merged[TIME_COL].dt.dayofyear
    merged["capacity_kwh"] = merged["target"].map(CAPACITY_KWH).astype(float)
    merged["global_blend_prediction"] = (
        0.60 * merged["exp03_prediction"] + 0.40 * merged["raw_prediction"]
    )
    merged["official_mask"] = merged["y_true_kwh"] >= 0.10 * merged["capacity_kwh"]
    merged["high_wind_mask"] = merged["raw_high_wind_mask"].astype(bool)
    forbidden = [column for column in merged if "test" in column.lower() or "full" in column.lower()]
    if forbidden:
        raise ValueError(f"non-OOF columns found: {forbidden}")
    if not merged["lead_time_h"].between(12, 35).all():
        raise ValueError("lead-time contract failed")
    issue_quarter = (merged[TIME_COL] - pd.Timedelta(hours=1)).dt.to_period("Q").astype(str)
    if not issue_quarter.equals(merged["quarter"]):
        raise ValueError("rolling quarter does not match the issue block")
    numeric = merged.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("OOF contract contains NaN/inf")
    return merged.sort_values(KEYS).reset_index(drop=True)


def score_prediction(data: pd.DataFrame, prediction_column: str) -> dict:
    frame = data[[TIME_COL, "target", "group_id", "y_true_kwh", prediction_column]].rename(
        columns={prediction_column: "y_pred_kwh"}
    )
    return score_available_groups(frame)[0]


def write_oof_checks(data: pd.DataFrame, output_root: Path) -> dict:
    checks = Path(output_root) / "checks"; checks.mkdir(parents=True, exist_ok=True)
    score = score_prediction(data, "global_blend_prediction")
    error = abs(score["total_score"] - EXPECTED_GLOBAL_SCORE)
    if error >= 1e-8:
        raise ValueError(f"Exp04 reference reproduction error {error:.3g} >= 1e-8")
    contract = {
        "rows": len(data),
        "quarters": sorted(data["quarter"].unique()),
        "first_forecast": str(data[TIME_COL].min()),
        "last_forecast": str(data[TIME_COL].max()),
        "lead_time_min_h": float(data["lead_time_h"].min()),
        "lead_time_max_h": float(data["lead_time_h"].max()),
        "duplicate_keys": int(data.duplicated(KEYS).sum()),
        "oof_only": True,
        "full_or_test_rows": 0,
        "target_serialization_tolerance_kwh": 0.01,
    }
    (checks / "oof_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    coverage = (
        data.groupby(["quarter", "target", "group_id"], sort=True)
        .agg(rows=(TIME_COL, "size"), first_forecast=(TIME_COL, "min"), last_forecast=(TIME_COL, "max"),
             official_samples=("official_mask", "sum"))
        .reset_index()
    )
    coverage.to_csv(checks / "oof_coverage.csv", index=False)
    reproduction = {
        "expected_total_score": EXPECTED_GLOBAL_SCORE,
        "reproduced": score,
        "absolute_error": error,
        "tolerance": 1e-8,
    }
    (checks / "reference_reproduction.json").write_text(
        json.dumps(reproduction, indent=2), encoding="utf-8"
    )
    return reproduction
