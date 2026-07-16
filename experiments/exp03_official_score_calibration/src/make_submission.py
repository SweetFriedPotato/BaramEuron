"""Finalize OOF ensemble search and create at most three non-duplicate submissions."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TARGETS, TIME_COL
from baram.data import load_sample_submission
from baram.submission import create_submission, validate_submission_contract

from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config

from .backtest import issue_quarter
from .calibration import (
    apply_affine,
    blend_predictions,
    rolling_affine_backtest,
    rolling_seasonal_affine_backtest,
    search_global_blend,
    select_affine_parameters,
)
from .evaluate import evaluate_models, score_available_groups
from .prediction_loader import KEY_COLS, load_exp01_model, load_neural_ensemble


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp03_official_score_calibration"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
EXP02_OUTPUT = PROJECT_ROOT / "experiments/exp02_daily_tcn_scada_aux/outputs"


def ficr_oof_ensemble(output_root: Path) -> pd.DataFrame:
    path = output_root / "predictions/ficr_aware_predictions.csv"
    data = pd.read_csv(path, parse_dates=[TIME_COL])
    data = data.loc[
        data["stage"].eq("full") & data["experiment_id"].eq("ficr_lambda_02")
    ]
    out = (
        data.groupby(KEY_COLS, sort=False)
        .agg(
            y_true_kwh=("y_true_kwh", "first"),
            y_pred_kwh=("y_pred_kwh", "mean"),
            validation_wind_mps=("validation_wind_mps", "first"),
            train_wind_p90_mps=("train_wind_p90_mps", "first"),
            high_wind_mask=("high_wind_mask", "first"),
        )
        .reset_index()
    )
    out["model_id"] = "ficr_lambda_02"
    out["seed"] = -1; out["ensemble"] = True
    return out


def align_three_models(catboost: pd.DataFrame, tcn: pd.DataFrame, ficr: pd.DataFrame) -> pd.DataFrame:
    # TCN/FICR retain full-precision truth; exp01 truth was serialized to three decimals.
    out = tcn[[*KEY_COLS, "y_true_kwh", "y_pred_kwh"]].rename(columns={"y_pred_kwh": "tcn_prediction"})
    out = out.merge(
        catboost[[*KEY_COLS, "y_pred_kwh"]].rename(columns={"y_pred_kwh": "catboost_prediction"}),
        on=KEY_COLS, validate="one_to_one",
    )
    out = out.merge(
        ficr[[*KEY_COLS, "y_pred_kwh"]].rename(columns={"y_pred_kwh": "ficr_prediction"}),
        on=KEY_COLS, validate="one_to_one",
    )
    if len(out) != len(tcn) or len(out) != len(catboost) or len(out) != len(ficr):
        raise ValueError("final ensemble OOF key sets differ")
    return out


def convex_weight_grid(step: float = 0.05):
    units = int(round(1.0 / step))
    for cat_units in range(units + 1):
        for tcn_units in range(units + 1 - cat_units):
            yield cat_units / units, tcn_units / units, (units - cat_units - tcn_units) / units


def apply_three_model_weights(aligned: pd.DataFrame, weights: tuple[float, float, float], model_id: str) -> pd.DataFrame:
    cat_weight, tcn_weight, ficr_weight = weights
    out = aligned[[*KEY_COLS, "y_true_kwh"]].copy()
    out["y_pred_kwh"] = (
        cat_weight * aligned["catboost_prediction"]
        + tcn_weight * aligned["tcn_prediction"]
        + ficr_weight * aligned["ficr_prediction"]
    )
    out["model_id"] = model_id
    return out


def search_final_ensemble(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weights in convex_weight_grid(0.05):
        candidate = apply_three_model_weights(aligned, weights, "candidate")
        summary, _ = score_available_groups(candidate)
        rows.append({"catboost_weight": weights[0], "tcn_aux_005_weight": weights[1],
                     "ficr_lambda_02_weight": weights[2], **summary})
    return pd.DataFrame(rows).sort_values(
        ["total_score", "ficr_lambda_02_weight"], ascending=[False, False]
    )


def group_blend_fold_separation(catboost: pd.DataFrame, tcn: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    fold_a_search = search_global_blend(
        catboost.loc[catboost["fold"].eq("fold_a")], tcn.loc[tcn["fold"].eq("fold_a")],
        np.round(np.arange(0.0, 1.0001, 0.025), 3),
    )
    center = float(fold_a_search.iloc[0]["right_weight"])
    weights = np.round(np.arange(max(0.0, center - 0.15), min(1.0, center + 0.15) + 1e-9, 0.025), 3)
    aligned_a = catboost.loc[catboost["fold"].eq("fold_a")]
    aligned_b = tcn.loc[tcn["fold"].eq("fold_a")]
    searches, selected = [], {}
    for target in sorted(aligned_a["target"].unique()):
        target_search = search_global_blend(
            aligned_a.loc[aligned_a["target"].eq(target)],
            aligned_b.loc[aligned_b["target"].eq(target)], weights,
        )
        target_search["target"] = target; searches.append(target_search)
        selected[target] = float(target_search.iloc[0]["right_weight"])
    selected.setdefault("kpx_group_3", center)
    search = pd.concat(searches, ignore_index=True)
    search["global_center"] = center
    return search, selected


def _read_test_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if TIME_COL not in data:
        raise ValueError(f"test prediction has no {TIME_COL}: {path}")
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    if data[TIME_COL].duplicated().any() or len(data) != 8760:
        raise ValueError(f"test prediction timestamp contract failed: {path}")
    values = data[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"test prediction contains NaN/inf: {path}")
    return data.set_index(TIME_COL)[TARGETS]


def _submission_from_frame(sample: pd.DataFrame, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    ordered = frame.loc[pd.DatetimeIndex(pd.to_datetime(sample[TIME_COL]))]
    submission = create_submission(
        sample, {target: ordered[target].to_numpy(dtype=float) for target in TARGETS}, path
    )
    validate_submission_contract(submission, sample)
    if submission[TARGETS].duplicated().all():
        raise ValueError("all submission target rows are duplicated")
    return submission


def finalize(output_root: Path = DEFAULT_OUTPUT) -> dict:
    for name in ("metrics", "predictions", "calibration", "submissions"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    # This experiment owns only exp03_* files in its ignored submission folder.
    # Remove a prior finalize run so the promised maximum of three is enforced.
    for stale in (output_root / "submissions").glob("exp03_*.csv"):
        stale.unlink()
    catboost = load_exp01_model("catboost_selected")
    tcn = load_neural_ensemble("tcn_aux_005")
    ficr = ficr_oof_ensemble(output_root)
    aligned = align_three_models(catboost, tcn, ficr)

    final_search = search_final_ensemble(aligned)
    final_search.to_csv(output_root / "metrics/final_ensemble_search.csv", index=False)
    best = final_search.iloc[0]
    final_weights = (
        float(best["catboost_weight"]), float(best["tcn_aux_005_weight"]),
        float(best["ficr_lambda_02_weight"]),
    )
    final_oof = apply_three_model_weights(aligned, final_weights, "final_ensemble")
    final_oof.to_csv(output_root / "predictions/final_ensemble_predictions.csv", index=False)

    calibration_search = search_global_blend(catboost, tcn, np.round(np.arange(0, 1.0001, 0.025), 3))
    calibration_weight = float(calibration_search.iloc[0]["right_weight"])
    calibration_base = blend_predictions(catboost, tcn, calibration_weight, "calibration_base")
    affine_parameters, affine_search = select_affine_parameters(calibration_base)
    calibrated_oof = apply_affine(calibration_base, affine_parameters, "calibration_only")
    calibrated_oof.to_csv(output_root / "predictions/calibrated_predictions.csv", index=False)
    affine_search.to_csv(output_root / "calibration/final_affine_search.csv", index=False)
    calibration_search.to_csv(output_root / "calibration/final_global_blend_search.csv", index=False)
    group_search, group_weights = group_blend_fold_separation(catboost, tcn)
    group_search.to_csv(output_root / "calibration/group_blend_search.csv", index=False)
    seasonal_backtest, seasonal_parameters = rolling_seasonal_affine_backtest(calibration_base)
    seasonal_backtest.to_csv(output_root / "calibration/seasonal_affine_backtest.csv", index=False)
    seasonal_parameters.to_csv(output_root / "calibration/seasonal_affine_parameters.csv", index=False)
    final_global_backtest, _ = rolling_affine_backtest(calibration_base)
    final_global_backtest.to_csv(output_root / "calibration/final_global_affine_backtest.csv", index=False)
    global_2024_mean = float(
        final_global_backtest.loc[final_global_backtest["evaluation_quarter"] >= "2024Q1", "calibrated_score"].mean()
    )
    seasonal_2024_mean = float(seasonal_backtest["seasonal_score"].mean())

    quarter_rows = []
    candidates = pd.concat([ficr, final_oof, calibrated_oof], ignore_index=True, sort=False)
    candidates["quarter"] = issue_quarter(candidates[TIME_COL])
    for (model_id, quarter), part in candidates.groupby(["model_id", "quarter"], sort=True):
        summary, _ = score_available_groups(part)
        quarter_rows.append({"model_id": model_id, "quarter": quarter, **summary})
    pd.DataFrame(quarter_rows).to_csv(output_root / "metrics/final_quarterly_scores.csv", index=False)
    score_table, group_table = evaluate_models(
        pd.concat([ficr, final_oof, calibrated_oof], ignore_index=True, sort=False)
    )
    score_table.to_csv(output_root / "metrics/final_candidate_scores.csv", index=False)
    group_table.to_csv(output_root / "metrics/final_candidate_group_scores.csv", index=False)

    cfg = baseline_config(); sample = load_sample_submission(cfg)
    cat_test = _read_test_predictions(EXP02_OUTPUT / "reference/exp01_catboost_selected_test.csv")
    tcn_test = _read_test_predictions(EXP02_OUTPUT / "predictions/tcn_full_ensemble_test.csv")
    ficr_test = _read_test_predictions(output_root / "predictions/ficr_aware_full_ensemble_test.csv")
    if not cat_test.index.equals(tcn_test.index) or not tcn_test.index.equals(ficr_test.index):
        raise ValueError("test prediction timestamp sets differ")

    calibration_test = (1.0 - calibration_weight) * cat_test + calibration_weight * tcn_test
    for target, (scale, offset) in affine_parameters.items():
        calibration_test[target] = np.maximum(calibration_test[target] * scale + offset, 0.0)
    final_test = (
        final_weights[0] * cat_test + final_weights[1] * tcn_test + final_weights[2] * ficr_test
    ).clip(lower=0.0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submissions = []
    calibration_path = output_root / f"submissions/exp03_calibration_only_{stamp}.csv"
    _submission_from_frame(sample, calibration_test, calibration_path); submissions.append(calibration_path)
    ficr_path = output_root / f"submissions/exp03_ficr_aware_{stamp}.csv"
    _submission_from_frame(sample, ficr_test, ficr_path); submissions.append(ficr_path)
    if not np.allclose(final_test.to_numpy(), ficr_test.to_numpy(), rtol=0.0, atol=1e-12):
        ensemble_path = output_root / f"submissions/exp03_best_ensemble_{stamp}.csv"
        _submission_from_frame(sample, final_test, ensemble_path); submissions.append(ensemble_path)

    summary = {
        "final_weights": {"catboost": final_weights[0], "tcn_aux_005": final_weights[1],
                          "ficr_lambda_02": final_weights[2]},
        "final_oof_score": float(best["total_score"]),
        "calibration_tcn_weight": calibration_weight,
        "affine_parameters": affine_parameters,
        "group_weights_selected_on_fold_a": group_weights,
        "seasonal_calibration_improved_quarters": int(seasonal_backtest["improved"].sum()),
        "seasonal_calibration_retained": bool(seasonal_backtest["improved"].mean() > 0.5),
        "global_affine_2024_mean_score": global_2024_mean,
        "seasonal_affine_2024_mean_score": seasonal_2024_mean,
        "best_calibration_mode": "global_affine" if global_2024_mean >= seasonal_2024_mean else "seasonal_affine",
        "calibration_oof_score": score_available_groups(calibrated_oof)[0],
        "submissions": [str(path) for path in submissions],
        "submission_count": len(submissions),
        "best_ensemble_is_duplicate_of_ficr_model": bool(len(submissions) == 2),
    }
    (output_root / "final_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_root.resolve()), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
