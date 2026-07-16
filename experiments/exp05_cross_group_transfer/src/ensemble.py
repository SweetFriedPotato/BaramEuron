"""Run cheap nested stages and the conditionally enabled A100 Stage D."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baram.constants import TIME_COL
from baram.constants import TARGETS
from baram.data import load_sample_submission
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config
from experiments.exp04_raw_grid_spatiotemporal.src.evaluate import prediction_frame
from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_loader import load_raw_grid_bundle
from experiments.exp04_raw_grid_spatiotemporal.src.run_experiment import (
    CONFIG_DIR as EXP04_CONFIG_DIR,
    _prepare_engineered_folds,
    _prepared_cache,
    load_variant_config,
)
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import train_raw_model

from .constrained_blend import apply_group_weights, nested_constrained_blend, penalty_grid
from .cross_group_attention import build_cross_group_model
from .evaluate import convex_search, rolling_metrics, slice_metrics, stage_d_decision
from .make_submission import make_submission, validate_submission_limit
from .oof_contract import load_oof_contract, score_prediction, write_oof_checks
from .residual_stacker import apply_final_stacker, nested_catboost_stacker, nested_ridge_stacker
from .stacker_features import build_stacker_features, load_weather_summaries, write_schema


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp05_cross_group_transfer"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"


def _read_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _directories(output: Path) -> None:
    for name in ("checks", "metrics", "predictions", "figures", "checkpoints", "submissions"):
        (output / name).mkdir(parents=True, exist_ok=True)


def _score_stage(
    data: pd.DataFrame,
    prediction_column: str,
    stage: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, quarters, groups = rolling_metrics(data, prediction_column)
    group3 = groups.loc[groups["group_id"].eq(3), "score"]
    summary["group3_score"] = float(group3.iloc[0]) if len(group3) else np.nan
    summary["stage"] = stage
    quarters.insert(0, "stage", stage); groups.insert(0, "stage", stage)
    return summary, quarters, groups


def run_cheap_stages(output: Path) -> dict:
    _directories(output)
    contract = load_oof_contract(); write_oof_checks(contract, output)
    contract.assign(y_pred_kwh=contract["exp03_prediction"])[
        ["quarter", TIME_COL, "target", "group_id", "y_true_kwh", "y_pred_kwh"]
    ].to_csv(output / "predictions/exp03_oof.csv", index=False)
    contract.assign(y_pred_kwh=contract["raw_prediction"])[
        ["quarter", TIME_COL, "target", "group_id", "y_true_kwh", "y_pred_kwh"]
    ].to_csv(output / "predictions/raw_oof.csv", index=False)

    blend_config = _read_yaml("constrained_group_blend.yaml")
    constrained, weights, searches, blend_summary = nested_constrained_blend(
        contract, penalty_grid(blend_config)
    )
    constrained["base_prediction"] = constrained["constrained_prediction"]
    constrained.to_csv(output / "predictions/constrained_blend_oof.csv", index=False)
    weights.to_csv(output / "metrics/group_weight_stability.csv", index=False)
    searches.to_csv(output / "metrics/constrained_group_weight_search.csv", index=False)
    _write_json(output / "metrics/constrained_group_summary.json", blend_summary)

    train_weather, test_weather = load_weather_summaries()
    featured, feature_columns = build_stacker_features(constrained, train_weather)
    write_schema(output / "checks/stacker_schema.json", feature_columns)
    _write_json(output / "checks/leakage_audit.json", {
        "rolling_oof_only": True,
        "outer_evaluation_target_used_for_selection": False,
        "target_input_features": [], "target_lag_features": [], "scada_input_features": [],
        "source_gate_omitted": "per-quarter gate artifacts unavailable; full-model gate would leak",
    })

    ridge, ridge_details, ridge_model = nested_ridge_stacker(
        featured, feature_columns, "base_prediction", _read_yaml("ridge_residual_stacker.yaml")
    )
    catboost, cat_details, cat_model = nested_catboost_stacker(
        featured, feature_columns, "base_prediction", _read_yaml("catboost_residual_stacker.yaml")
    )
    ridge.to_csv(output / "predictions/ridge_stacker_oof.csv", index=False)
    catboost.to_csv(output / "predictions/catboost_stacker_oof.csv", index=False)

    stage_specs = [
        (contract, "global_blend_prediction", "exp04_global"),
        (constrained, "constrained_prediction", "constrained"),
        (ridge, "ridge_prediction", "ridge"),
        (catboost, "catboost_prediction", "catboost"),
    ]
    summaries, quarter_tables, group_tables = [], [], []
    for values, column, stage in stage_specs:
        summary, quarters, groups = _score_stage(values, column, stage)
        summaries.append(summary); quarter_tables.append(quarters); group_tables.append(groups)
    candidates = pd.DataFrame(summaries).sort_values("total_score", ascending=False)
    candidates.to_csv(output / "metrics/final_candidate_scores.csv", index=False)
    pd.concat(quarter_tables, ignore_index=True).to_csv(
        output / "metrics/nested_quarter_scores.csv", index=False
    )
    pd.concat(group_tables, ignore_index=True).to_csv(output / "metrics/group_scores.csv", index=False)
    pd.concat(group_tables, ignore_index=True).loc[lambda x: x["group_id"].eq(3)].to_csv(
        output / "metrics/group3_scores.csv", index=False
    )
    ridge_details.to_csv(output / "metrics/ridge_stacker_scores.csv", index=False)
    cat_details.to_csv(output / "metrics/catboost_stacker_scores.csv", index=False)
    importance_rows = []
    for target in TARGETS:
        ridge_estimator = ridge_model.models[target][0].named_steps["ridge"]
        for feature, coefficient in zip(feature_columns, ridge_estimator.coef_):
            importance_rows.append({
                "model": "ridge", "target": target, "feature": feature,
                "importance": abs(float(coefficient)), "signed_coefficient": float(coefficient),
            })
        for feature, importance in zip(
            feature_columns, cat_model.models[target][0].get_feature_importance()
        ):
            importance_rows.append({
                "model": "catboost", "target": target, "feature": feature,
                "importance": float(importance), "signed_coefficient": np.nan,
            })
    pd.DataFrame(importance_rows).to_csv(output / "metrics/stacker_feature_importance.csv", index=False)
    january, high_wind = [], []
    for values, column, stage in stage_specs:
        j, h = slice_metrics(values, column); j.insert(0, "stage", stage); h.insert(0, "stage", stage)
        january.append(j); high_wind.append(h)
    pd.concat(january, ignore_index=True).to_csv(output / "metrics/january_scores.csv", index=False)
    pd.concat(high_wind, ignore_index=True).to_csv(output / "metrics/high_wind_scores.csv", index=False)
    cheap_best = candidates.loc[candidates["stage"].ne("exp04_global")].iloc[0].to_dict()
    decision = stage_d_decision(cheap_best)
    decision["best_cheap_stage"] = cheap_best["stage"]
    decision["best_cheap_summary"] = cheap_best
    _write_json(output / "stage_d_decision.json", decision)
    pd.DataFrame(columns=["phase", "fold", "seed", "total_score"]).to_csv(
        output / "metrics/cross_group_attention_scores.csv", index=False
    )
    pd.DataFrame(columns=[
        "quarter", TIME_COL, "target", "group_id", "y_true_kwh", "y_pred_kwh", "status"
    ]).to_csv(output / "predictions/cross_group_raw_oof.csv", index=False)
    # Final convex ensemble uses only nested OOF candidate predictions.
    ensemble_frame = contract.copy()
    ensemble_frame["constrained_prediction"] = constrained["constrained_prediction"].to_numpy()
    ensemble_frame["ridge_prediction"] = ridge["ridge_prediction"].to_numpy()
    ensemble_frame["catboost_prediction"] = catboost["catboost_prediction"].to_numpy()
    best_residual_column = (
        "ridge_prediction"
        if float(candidates.loc[candidates["stage"].eq("ridge"), "total_score"].iloc[0])
        >= float(candidates.loc[candidates["stage"].eq("catboost"), "total_score"].iloc[0])
        else "catboost_prediction"
    )
    search_columns = ["global_blend_prediction", "constrained_prediction", best_residual_column]
    convex = convex_search(ensemble_frame, search_columns)
    convex.to_csv(output / "metrics/final_ensemble_weight_search.csv", index=False)
    selected = convex.iloc[0]
    ensemble_frame["final_prediction"] = sum(
        float(selected[f"weight_{column}"]) * ensemble_frame[column] for column in search_columns
    )
    ensemble_frame.to_csv(output / "predictions/final_candidate_oof.csv", index=False)
    final_summary, final_quarters, final_groups = _score_stage(
        ensemble_frame, "final_prediction", "final_ensemble"
    )
    pd.concat([candidates, pd.DataFrame([final_summary])], ignore_index=True).to_csv(
        output / "metrics/final_candidate_scores.csv", index=False
    )
    pd.concat([pd.concat(quarter_tables, ignore_index=True), final_quarters], ignore_index=True).to_csv(
        output / "metrics/nested_quarter_scores.csv", index=False
    )
    all_groups = pd.concat([pd.concat(group_tables, ignore_index=True), final_groups], ignore_index=True)
    all_groups.to_csv(output / "metrics/group_scores.csv", index=False)
    all_groups.loc[all_groups["group_id"].eq(3)].to_csv(output / "metrics/group3_scores.csv", index=False)
    final_january, final_high_wind = slice_metrics(ensemble_frame, "final_prediction")
    final_january.insert(0, "stage", "final_ensemble")
    final_high_wind.insert(0, "stage", "final_ensemble")
    pd.concat([*january, final_january], ignore_index=True).to_csv(
        output / "metrics/january_scores.csv", index=False
    )
    pd.concat([*high_wind, final_high_wind], ignore_index=True).to_csv(
        output / "metrics/high_wind_scores.csv", index=False
    )

    # Apply models fitted only on all rolling OOF residuals to aligned full-test base predictions.
    exp03_test = pd.read_csv(
        PROJECT_ROOT / "experiments/exp03_official_score_calibration/outputs/predictions/ficr_aware_full_ensemble_test.csv",
        parse_dates=[TIME_COL],
    )
    raw_test = pd.read_csv(
        PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs/predictions/raw_ensemble_predictions.csv",
        parse_dates=[TIME_COL],
    )
    if not exp03_test[TIME_COL].equals(raw_test[TIME_COL]):
        raise ValueError("Exp03/Exp04 full-test timestamps differ")
    test_rows = []
    for group_id, target in enumerate(TARGETS, 1):
        test_rows.append(pd.DataFrame({
            TIME_COL: exp03_test[TIME_COL], "target": target, "group_id": group_id,
            "exp03_prediction": exp03_test[target], "raw_prediction": raw_test[target],
        }))
    test_long = pd.concat(test_rows, ignore_index=True).sort_values(
        [TIME_COL, "target", "group_id"]
    ).reset_index(drop=True)
    test_long = apply_group_weights(
        test_long, blend_summary["final_weights"], "constrained_prediction"
    )
    test_long["base_prediction"] = test_long["constrained_prediction"]
    test_featured, test_columns = build_stacker_features(test_long, test_weather)
    if test_columns != feature_columns:
        raise ValueError("OOF/test stacker feature schema mismatch")
    ridge_test = apply_final_stacker(ridge_model, test_featured, "ridge_prediction")
    catboost_test = apply_final_stacker(cat_model, test_featured, "catboost_prediction")
    residual_test = ridge_test if best_residual_column == "ridge_prediction" else catboost_test
    test_predictions = test_featured.copy()
    test_predictions["ridge_prediction"] = ridge_test["ridge_prediction"]
    test_predictions["catboost_prediction"] = catboost_test["catboost_prediction"]
    test_predictions["global_blend_prediction"] = (
        0.6 * test_predictions["exp03_prediction"] + 0.4 * test_predictions["raw_prediction"]
    )
    test_predictions["final_prediction"] = sum(
        float(selected[f"weight_{column}"]) * (
            residual_test[best_residual_column] if column == best_residual_column else test_predictions[column]
        )
        for column in search_columns
    )
    test_predictions.to_csv(output / "predictions/final_test_predictions.csv", index=False)
    sample = load_sample_submission(baseline_config())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submissions = [
        output / "submissions" / f"exp05_constrained_group_blend_{stamp}.csv",
        output / "submissions" / f"exp05_{best_residual_column.replace('_prediction', '')}_stacker_{stamp}.csv",
        output / "submissions" / f"exp05_final_ensemble_{stamp}.csv",
    ]
    make_submission(sample, test_predictions, submissions[0], "constrained_prediction")
    make_submission(sample, residual_test, submissions[1], best_residual_column)
    make_submission(sample, test_predictions, submissions[2], "final_prediction")
    validate_submission_limit(submissions)
    _write_json(output / "submission_selection.json", {
        "paths": [str(path) for path in submissions],
        "best_residual": best_residual_column,
        "final_columns": search_columns,
        "final_weights": {column: float(selected[f"weight_{column}"]) for column in search_columns},
        "auto_submitted": False,
    })
    return {
        "candidates": candidates.to_dict("records"), "stage_d_decision": decision,
        "final_ensemble": final_summary, "submissions": [str(path) for path in submissions],
    }


def _cross_group_fold_b(output: Path, phase: str, seeds: list[int]) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage D is GPU-only and no CUDA device is available")
    _directories(output)
    cross = _read_yaml("raw_cross_group_attention.yaml")
    base = load_variant_config(EXP04_CONFIG_DIR / "raw_hybrid_gated.yaml")
    base["experiment_id"] = "raw_cross_group_attention"
    raw = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    engineered, _ = _prepare_engineered_folds(output, ["fold_b"])
    data = _prepared_cache(raw, engineered, output)[("fold_b", True)]
    rows, frames = [], []
    epochs = int(cross["smoke_epochs"]) if phase == "smoke" else None
    for seed in seeds:
        model = build_cross_group_model(
            base, cross,
            data.train_inputs.ldaps.shape[-1], data.train_inputs.gfs.shape[-1],
            data.ldaps_static, data.gfs_static, data.common_dim, data.group_dims,
        )
        checkpoint = output / "checkpoints" / f"cross_group_{phase}_fold_b_seed_{seed}.pt"
        result = train_raw_model(
            model, data.train_inputs, data.train_y, data.train_mask,
            data.valid_inputs, data.valid_y, data.valid_mask, base, seed, checkpoint,
            data.train_aux, data.train_aux_mask, max_epochs_override=epochs,
        )
        row = {
            "phase": phase, "fold": "fold_b", "seed": seed,
            "best_epoch": result.best_epoch, "total_score": result.best_total_score,
            "one_minus_nmae": result.best_one_minus_nmae, "ficr": result.best_ficr,
            "device": result.device, "gpu": torch.cuda.get_device_name(0),
            "training_seconds": result.training_seconds,
            "raw_seed42_reference": 0.644135,
            "delta_vs_raw_seed42": result.best_total_score - 0.644135,
        }
        rows.append(row)
        if seed == 42:
            attention_path = output / "predictions" / f"cross_group_attention_{phase}_seed42.npz"
            np.savez_compressed(
                attention_path,
                **{name: value for name, value in result.diagnostics.items() if value is not None},
            )
        frame = prediction_frame(
            data.valid_timestamps, data.valid_y, data.valid_mask, result.prediction_cf,
            "raw_cross_group_attention", "fold_b", seed,
            data.validation_wind, data.high_wind_threshold,
        )
        frame["phase"] = phase; frames.append(frame)
    metrics_path = output / "metrics/cross_group_attention_scores.csv"
    old = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    metrics = pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False)
    metrics = metrics.drop_duplicates(["phase", "fold", "seed"], keep="last")
    metrics.to_csv(metrics_path, index=False)
    pred_path = output / "predictions/cross_group_fold_b_predictions.csv"
    old_pred = pd.read_csv(pred_path, parse_dates=[TIME_COL]) if pred_path.exists() else pd.DataFrame()
    predictions = pd.concat([old_pred, *frames], ignore_index=True, sort=False)
    predictions = predictions.drop_duplicates(["phase", "fold", "seed", TIME_COL, "target"], keep="last")
    predictions.to_csv(pred_path, index=False)
    return pd.DataFrame(rows)


def write_manifest(output: Path) -> None:
    _write_json(output / "run_manifest.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
                                     check=True, capture_output=True, text=True).stdout.strip(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                     check=True, capture_output=True, text=True).stdout.strip(),
        "public_scores_used_for_selection": False,
        "stage_a_to_c_device": "cpu",
        "stage_d_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["cheap", "cross-smoke", "cross-full", "cross-seeds"], default="cheap")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); output = args.output_root.resolve()
    if args.phase == "cheap":
        result = run_cheap_stages(output)
    elif args.phase == "cross-smoke":
        result = _cross_group_fold_b(output, "smoke", [42]).to_dict("records")
    elif args.phase == "cross-full":
        result = _cross_group_fold_b(output, "full", [42]).to_dict("records")
    else:
        metrics = pd.read_csv(output / "metrics/cross_group_attention_scores.csv")
        seed42 = metrics.loc[(metrics["phase"] == "full") & (metrics["seed"] == 42)]
        if seed42.empty or float(seed42.iloc[-1]["total_score"]) <= 0.644135:
            raise RuntimeError("seed42 did not beat raw_hybrid_gated; seeds 52/62 are forbidden")
        result = _cross_group_fold_b(output, "full", [52, 62]).to_dict("records")
    write_manifest(output); print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
