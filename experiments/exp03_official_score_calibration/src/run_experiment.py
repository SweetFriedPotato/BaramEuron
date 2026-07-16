"""Run official rescoring and leakage-safe prediction-only calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import issue_quarter, quarterly_scores
from .calibration import (
    apply_affine,
    blend_predictions,
    rolling_affine_backtest,
    search_global_blend,
    select_affine_parameters,
)
from .evaluate import (
    add_official_components,
    evaluate_models,
    evaluation_mask_summary,
    ficr_threshold_diagnostics,
    score_available_groups,
    slice_scores,
)
from .prediction_loader import (
    EXP02_PREDICTIONS,
    aggregate_seeds,
    load_best_blend,
    load_existing_predictions,
    load_exp01_model,
    load_neural_seeds,
    load_neural_ensemble,
    prediction_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp03_official_score_calibration"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
OFFICIAL_NOTEBOOK = PROJECT_ROOT / "official/dacon_baram_metric/metric.ipynb"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def _save_figures(scores: pd.DataFrame, quarters: pd.DataFrame, blend: pd.DataFrame,
                  diagnostics_source: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fold_b = scores.loc[scores["fold"].eq("fold_b")].sort_values("total_score")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fold_b["model_id"], fold_b["total_score"])
    ax.set_xlabel("Official Score"); ax.set_title("Fold B official score comparison")
    fig.tight_layout(); fig.savefig(figures / "official_score_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(scores["one_minus_nmae"], scores["ficr"])
    for row in scores.itertuples():
        ax.annotate(f"{row.model_id}/{row.fold}", (row.one_minus_nmae, row.ficr), fontsize=7)
    ax.set_xlabel("1-NMAE"); ax.set_ylabel("FICR"); ax.set_title("Accuracy and settlement trade-off")
    fig.tight_layout(); fig.savefig(figures / "nmae_vs_ficr.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_id, part in quarters.groupby("model_id"):
        ax.plot(part["quarter"], part["total_score"], marker="o", label=model_id)
    ax.tick_params(axis="x", rotation=45); ax.set_ylabel("Score"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(figures / "quarterly_score_stability.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = blend.sort_values("right_weight")
    ax.plot(ordered["right_weight"], ordered["total_score"])
    ax.set_xlabel("TCN weight"); ax.set_ylabel("Score"); ax.set_title("Global blend search")
    fig.tight_layout(); fig.savefig(figures / "calibration_surface.png", dpi=160); plt.close(fig)

    data = add_official_components(diagnostics_source)
    data = data.loc[data["official_mask"]]
    distance = np.minimum(np.abs(data["error_rate"] - 0.06), np.abs(data["error_rate"] - 0.08))
    fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(distance.clip(upper=0.04), bins=60)
    ax.set_xlabel("Absolute distance to nearest FICR threshold")
    fig.tight_layout(); fig.savefig(figures / "error_distance_to_ficr_threshold.png", dpi=160); plt.close(fig)

    for month, name in ((1, "january_score_comparison.png"),):
        subset = diagnostics_source.loc[pd.to_datetime(diagnostics_source["forecast_kst_dtm"]).dt.month.eq(month)]
        rows = []
        for model_id, part in subset.groupby("model_id"):
            summary, _ = score_available_groups(part); rows.append((model_id, summary["total_score"]))
        fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([x[0] for x in rows], [x[1] for x in rows])
        ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(figures / name, dpi=160); plt.close(fig)

    high = diagnostics_source.loc[diagnostics_source.get("high_wind_mask", False).astype(bool)]
    rows = []
    for model_id, part in high.groupby("model_id"):
        summary, _ = score_available_groups(part); rows.append((model_id, summary["total_score"]))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([x[0] for x in rows], [x[1] for x in rows])
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    fig.savefig(figures / "high_wind_score_comparison.png", dpi=160); plt.close(fig)


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    for name in ("checks", "metrics", "predictions", "calibration", "figures", "checkpoints", "submissions"):
        (output / name).mkdir(parents=True, exist_ok=True)

    existing = load_existing_predictions()
    prediction_inventory(existing).to_csv(output / "checks/prediction_inventory.csv", index=False)
    scores, groups = evaluate_models(existing)
    scores.to_csv(output / "metrics/existing_models_official_scores.csv", index=False)
    scores.to_csv(output / "metrics/metric_alignment.csv", index=False)
    scores.to_csv(output / "checks/metric_alignment.csv", index=False)
    groups.to_csv(output / "metrics/group_scores.csv", index=False)
    evaluation_mask_summary(existing).to_csv(output / "checks/evaluation_mask_summary.csv", index=False)

    enriched = add_official_components(existing)
    enriched["month"] = enriched["forecast_kst_dtm"].dt.month
    monthly = slice_scores(existing, "month", enriched["month"])
    monthly.to_csv(output / "metrics/monthly_scores.csv", index=False)
    ficr_threshold_diagnostics(existing).to_csv(
        output / "metrics/ficr_threshold_diagnostics.csv", index=False
    )
    quarters = quarterly_scores(existing)
    quarters.to_csv(output / "metrics/quarterly_backtest_scores.csv", index=False)

    catboost = load_exp01_model("catboost_selected")
    tcn = load_neural_ensemble("tcn_aux_005")
    blend_search_parts = []
    for fold in ("fold_a", "fold_b"):
        search = search_global_blend(
            catboost.loc[catboost["fold"].eq(fold)], tcn.loc[tcn["fold"].eq(fold)],
            np.round(np.arange(0.0, 1.0001, 0.025), 3),
        )
        search["fold"] = fold; blend_search_parts.append(search)
    blend_search = pd.concat(blend_search_parts, ignore_index=True)
    blend_search.to_csv(output / "metrics/blend_search_official_score.csv", index=False)
    best_weight = float(
        blend_search.loc[blend_search["fold"].eq("fold_a")]
        .sort_values(["total_score", "right_weight"], ascending=[False, True]).iloc[0]["right_weight"]
    )
    calibrated_blend = blend_predictions(catboost, tcn, best_weight, "calibration_global_blend")

    seed_rows = []
    for model_id in ("mlp_pointwise", "tcn_plain", "tcn_aux_005", "tcn_aux_015"):
        seeds = load_neural_seeds(model_id)
        for method in ("mean", "median", "trimmed_mean"):
            aggregated = aggregate_seeds(seeds, method)
            for fold, part in aggregated.groupby("fold"):
                summary, _ = score_available_groups(part)
                seed_rows.append({"model_id": model_id, "aggregation": method, "fold": fold, **summary})
    pd.DataFrame(seed_rows).to_csv(output / "calibration/seed_aggregation.csv", index=False)

    rolling, calibration_search = rolling_affine_backtest(calibrated_blend)
    rolling.to_csv(output / "calibration/rolling_affine_backtest.csv", index=False)
    calibration_search.to_csv(output / "metrics/calibration_search.csv", index=False)
    calibrated_blend.assign(quarter=issue_quarter(calibrated_blend["forecast_kst_dtm"])).to_csv(
        output / "predictions/rolling_oof_predictions.csv", index=False
    )

    fit = calibrated_blend.loc[calibrated_blend["fold"].eq("fold_a")]
    parameters, _ = select_affine_parameters(fit)
    for target in calibrated_blend["target"].unique():
        parameters.setdefault(target, (1.0, 0.0))
    calibrated = apply_affine(calibrated_blend, parameters, "calibration_affine")
    calibrated.to_csv(output / "predictions/calibrated_predictions.csv", index=False)

    _save_figures(scores, quarters, blend_search.loc[blend_search["fold"].eq("fold_b")], existing, output)
    notebook_hash = hashlib.sha256(OFFICIAL_NOTEBOOK.read_bytes()).hexdigest()
    scorer_version = {
        "source_page": "https://dacon.io/competitions/official/236727/codeshare/14035",
        "source_object": "https://dacon.s3.ap-northeast-2.amazonaws.com/codeshare/236727/14035/md_file.ipynb",
        "sha256": notebook_hash,
        "expected_sha256": "0a3ab5a57dba0705dbdbda73cd723be37ef39cce388fcb22b1a220ce523a70f9",
    }
    _write_json(output / "scorer_version.json", scorer_version)
    manifest = {
        "run_at": datetime.now().astimezone().isoformat(),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "official_scorer": scorer_version,
        "existing_prediction_rows": len(existing),
        "best_global_tcn_weight_selected_on_fold_a": best_weight,
        "affine_parameters_selected_on_fold_a": parameters,
        "public_scores_used_for_tuning": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
