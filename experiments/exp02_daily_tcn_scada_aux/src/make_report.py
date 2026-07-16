"""Figures and concise report generation for exp02."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baram.constants import TIME_COL


def _save(path: Path, title: str | None = None) -> None:
    if title:
        plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def make_figures(
    history: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    january_metrics: pd.DataFrame,
    high_wind_metrics: pd.DataFrame,
    blend_search: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    best_tcn: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for keys, part in history.groupby(["experiment_id", "fold", "seed"]):
        plt.plot(part["epoch"], part["valid_macro_nmae"], alpha=.55, label="/".join(map(str, keys)))
    plt.xlabel("Epoch"); plt.ylabel("Validation macro nMAE"); plt.legend(fontsize=6, ncol=3)
    _save(output_dir / "training_curves.png", "Training curves")

    plt.figure(figsize=(8, 4.5))
    fold_b_seed = seed_metrics[seed_metrics.fold.eq("fold_b")]
    for experiment, part in fold_b_seed.groupby("experiment_id"):
        plt.scatter([experiment] * len(part), part["macro_nmae"], label=experiment)
    plt.xticks(rotation=25); plt.ylabel("Macro nMAE")
    _save(output_dir / "seed_stability.png", "Fold B seed stability")

    plt.figure(figsize=(8, 4.5))
    view = ensemble_metrics[ensemble_metrics.fold.eq("fold_b")].sort_values("macro_nmae")
    plt.bar(view["experiment_id"], view["macro_nmae"]); plt.xticks(rotation=25); plt.ylabel("Macro nMAE")
    _save(output_dir / "model_nmae_comparison.png", "Model ensemble comparison")

    group_view = group_metrics[(group_metrics.fold == "fold_b") & (group_metrics.ensemble == True)]
    group_view.pivot(index="experiment_id", columns="group_id", values="nmae").plot(kind="bar", figsize=(9, 5))
    plt.ylabel("nMAE"); plt.xticks(rotation=25); plt.legend(title="Group")
    _save(output_dir / "group_nmae_comparison.png", "Fold B group comparison")

    chosen = monthly_metrics[(monthly_metrics.fold == "fold_b") & (monthly_metrics.ensemble == True)
                             & monthly_metrics.experiment_id.isin(["catboost_reference", best_tcn, "best_blend"])]
    monthly = chosen.groupby(["experiment_id", "month"])["nmae"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    for experiment, part in monthly.groupby("experiment_id"):
        plt.plot(part.month, part.nmae, marker="o", label=experiment)
    plt.xticks(range(1, 13)); plt.xlabel("Month"); plt.ylabel("Macro nMAE"); plt.legend()
    _save(output_dir / "monthly_error_comparison.png", "Monthly error")

    january = validation_predictions[(validation_predictions.ensemble == True)
                                     & (pd.to_datetime(validation_predictions[TIME_COL]).dt.month == 1)
                                     & validation_predictions.experiment_id.isin(["catboost_reference", best_tcn, "best_blend"])]
    plt.figure(figsize=(11, 5))
    for experiment, part in january[january.group_id.eq(3)].groupby("experiment_id"):
        part = part.sort_values(TIME_COL)
        plt.plot(pd.to_datetime(part[TIME_COL]), part.y_pred_kwh, linewidth=.7, label=experiment)
    if not january.empty:
        actual = january[january.group_id.eq(3)].drop_duplicates(TIME_COL).sort_values(TIME_COL)
        plt.plot(pd.to_datetime(actual[TIME_COL]), actual.y_true_kwh, color="black", linewidth=.8, label="actual")
    plt.ylabel("Group 3 kWh"); plt.legend()
    _save(output_dir / "january_timeseries.png", "January group 3")

    high = high_wind_metrics[(high_wind_metrics.fold == "fold_b") & (high_wind_metrics.ensemble == True)
                             & high_wind_metrics.experiment_id.isin(["catboost_reference", best_tcn, "best_blend"])]
    if not high.empty:
        high.pivot(index="experiment_id", columns="group_id", values="nmae").plot(kind="bar", figsize=(9, 5))
    else:
        plt.figure(figsize=(9, 5)); plt.text(.5, .5, "No high-wind rows", ha="center")
    plt.ylabel("High-wind nMAE"); plt.xticks(rotation=20)
    _save(output_dir / "high_wind_error_comparison.png", "High-wind error")

    ref = validation_predictions[validation_predictions.experiment_id.eq("catboost_reference") & (validation_predictions.ensemble == True)]
    tcn = validation_predictions[validation_predictions.experiment_id.eq(best_tcn) & (validation_predictions.ensemble == True)]
    if not ref.empty and not tcn.empty:
        aligned = ref[[TIME_COL, "target", "y_true_kwh", "y_pred_kwh"]].merge(
            tcn[[TIME_COL, "target", "y_pred_kwh"]], on=[TIME_COL, "target"], suffixes=("_cat", "_tcn")
        )
        x = aligned.y_pred_kwh_cat - aligned.y_true_kwh
        y = aligned.y_pred_kwh_tcn - aligned.y_true_kwh
        sample = np.linspace(0, len(aligned) - 1, min(10000, len(aligned))).astype(int)
        plt.figure(figsize=(6, 6)); plt.scatter(x.iloc[sample], y.iloc[sample], s=3, alpha=.2)
        plt.xlabel("CatBoost residual"); plt.ylabel("TCN residual")
    else:
        plt.figure(figsize=(6, 6)); plt.text(.5, .5, "Reference unavailable", ha="center")
    _save(output_dir / "residual_correlation.png", "Residual correlation")

    plt.figure(figsize=(8, 4.5))
    for fold, part in blend_search.groupby("fold"):
        plt.plot(part.tcn_weight, part.macro_nmae, marker="o", label=fold)
    plt.xlabel("TCN weight"); plt.ylabel("Macro nMAE"); plt.legend()
    _save(output_dir / "blend_weight_search.png", "Blend weight search")


def write_report(path: Path, manifest: dict, ensemble_metrics: pd.DataFrame, group_metrics: pd.DataFrame) -> None:
    selected = manifest.get("best_tcn_config")
    best_weight = manifest.get("best_blend_weight")
    view = ensemble_metrics[["experiment_id", "fold", "macro_nmae"]].copy()
    table = view.to_markdown(index=False)
    groups = group_metrics[(group_metrics.experiment_id == "best_blend") & (group_metrics.fold == "fold_b")]
    group_lines = "\n".join(f"- group {int(row.group_id)}: `{row.nmae:.6f}`" for row in groups.itertuples()) or "- unavailable"
    text = f"""# exp02 daily TCN with SCADA auxiliary report

## Result

- A100 used: `{manifest.get('gpu_used')}`
- Feature count: `{manifest.get('feature_count')}`
- Best TCN: `{selected}`
- Auxiliary retained: `{manifest.get('auxiliary_retained')}`
- Best common TCN blend weight: `{best_weight}`
- Fold B best blend macro nMAE: `{manifest.get('best_blend_fold_b_macro_nmae')}`
- CatBoost reference improvement: `{manifest.get('catboost_improvement')}`
- Submission: `{manifest.get('submission_path')}`
- Drive artifact: `{manifest.get('drive_artifact_path')}`

## Ensemble metrics

{table}

## Best blend groups

{group_lines}

## Diagnostics

```json
{json.dumps(manifest.get('residual_diagnostics', {}), ensure_ascii=False, indent=2)}
```

The first next change should be: {manifest.get('next_experiment_change', 'review temporal model capacity')}.
"""
    path.write_text(text, encoding="utf-8")
