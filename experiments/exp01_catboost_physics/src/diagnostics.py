"""Metrics, slice diagnostics, and figures for exp01."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metrics(y: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray, capacity: float) -> dict[str, float]:
    y_array = np.asarray(y, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    mae = float(np.mean(np.abs(y_array - prediction_array)))
    return {"mae": mae, "nmae": mae / float(capacity), "rows": int(len(y_array))}


def build_metric_tables(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all requested diagnostics from one long validation-prediction table."""
    group_rows: list[dict] = []
    month_rows: list[dict] = []
    hour_rows: list[dict] = []
    region_rows: list[dict] = []
    wind_rows: list[dict] = []

    keys = ["experiment_id", "ablation_label", "fold", "target", "group_id", "capacity_kwh"]
    for values, part in predictions.groupby(keys, sort=False, dropna=False):
        context = dict(zip(keys, values, strict=True))
        capacity = float(context["capacity_kwh"])
        raw = _metrics(part["y_true_kwh"], part["y_pred_raw_kwh"], capacity)
        lower = _metrics(part["y_true_kwh"], part["y_pred_kwh"], capacity)
        upper = _metrics(part["y_true_kwh"], part["y_pred_capacity_clipped_kwh"], capacity)
        group_rows.append(
            {
                **context,
                "mae": lower["mae"],
                "nmae": lower["nmae"],
                "raw_mae": raw["mae"],
                "raw_nmae": raw["nmae"],
                "capacity_clipped_mae": upper["mae"],
                "capacity_clipped_nmae": upper["nmae"],
                "rows": lower["rows"],
                "feature_count": int(part["feature_count"].iloc[0]),
                "training_seconds": float(part["training_seconds"].iloc[0]),
                "best_iteration": int(part["best_iteration"].iloc[0]),
            }
        )

        timed = part.assign(
            month=pd.to_datetime(part["forecast_kst_dtm"]).dt.month,
            hour=pd.to_datetime(part["forecast_kst_dtm"]).dt.hour,
        )
        for month, chunk in timed.groupby("month"):
            month_rows.append({**context, "month": int(month), **_metrics(chunk["y_true_kwh"], chunk["y_pred_kwh"], capacity)})
        for hour, chunk in timed.groupby("hour"):
            hour_rows.append({**context, "hour": int(hour), **_metrics(chunk["y_true_kwh"], chunk["y_pred_kwh"], capacity)})

        cf = part["y_true_kwh"] / capacity
        regions = {
            "target_zero": cf == 0,
            "cf_gt0_lt10pct": (cf > 0) & (cf < 0.10),
            "cf_10_30pct": (cf >= 0.10) & (cf < 0.30),
            "cf_30_80pct": (cf >= 0.30) & (cf < 0.80),
            "cf_ge80pct": cf >= 0.80,
        }
        for region, mask in regions.items():
            if mask.any():
                chunk = part.loc[mask]
                region_rows.append({**context, "capacity_region": region, **_metrics(chunk["y_true_kwh"], chunk["y_pred_kwh"], capacity)})

        high = part["high_wind_mask"].astype(bool)
        if high.any():
            chunk = part.loc[high]
            wind_rows.append(
                {
                    **context,
                    "wind_feature": str(part["high_wind_feature"].iloc[0]),
                    "train_wind_p90_mps": float(part["train_wind_p90_mps"].iloc[0]),
                    **_metrics(chunk["y_true_kwh"], chunk["y_pred_kwh"], capacity),
                }
            )

    by_group = pd.DataFrame(group_rows)
    macro = (
        by_group.groupby(["experiment_id", "ablation_label", "fold"], sort=False)
        .agg(
            macro_mae=("mae", "mean"),
            macro_nmae=("nmae", "mean"),
            macro_raw_mae=("raw_mae", "mean"),
            macro_raw_nmae=("raw_nmae", "mean"),
            macro_capacity_clipped_mae=("capacity_clipped_mae", "mean"),
            macro_capacity_clipped_nmae=("capacity_clipped_nmae", "mean"),
            feature_count_mean=("feature_count", "mean"),
            training_seconds=("training_seconds", "sum"),
            evaluated_groups=("group_id", "nunique"),
        )
        .reset_index()
    )
    return {
        "ablation": macro,
        "group": by_group,
        "month": pd.DataFrame(month_rows),
        "hour": pd.DataFrame(hour_rows),
        "capacity_region": pd.DataFrame(region_rows),
        "high_wind": pd.DataFrame(wind_rows),
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_figures(tables: dict[str, pd.DataFrame], predictions: pd.DataFrame, output_dir: Path, best_experiment: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation = tables["ablation"]
    group = tables["group"]
    month = tables["month"]
    high_wind = tables["high_wind"]

    fold_b = ablation.query("fold == 'fold_b'").sort_values("ablation_label")
    plt.figure(figsize=(8, 4.5))
    plt.bar(fold_b["ablation_label"], fold_b["macro_nmae"], color="#4472C4")
    plt.ylabel("Macro nMAE")
    plt.xlabel("Ablation")
    plt.title("Fold B macro nMAE")
    _save_figure(output_dir / "ablation_macro_nmae.png")

    group_b = group.query("fold == 'fold_b'")
    pivot = group_b.pivot(index="ablation_label", columns="group_id", values="nmae").sort_index()
    pivot.plot(kind="bar", figsize=(9, 5), color=["#4472C4", "#ED7D31", "#70AD47"])
    plt.ylabel("nMAE")
    plt.xlabel("Ablation")
    plt.title("Fold B group nMAE")
    plt.legend(title="Group")
    _save_figure(output_dir / "group_nmae_comparison.png")

    selected_ids = [item for item in ["rf_reference", "catboost_basic", best_experiment] if item in set(month["experiment_id"])]
    monthly = month[(month["fold"] == "fold_b") & month["experiment_id"].isin(dict.fromkeys(selected_ids))]
    monthly_macro = monthly.groupby(["experiment_id", "month"], sort=False)["nmae"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    for experiment, chunk in monthly_macro.groupby("experiment_id", sort=False):
        plt.plot(chunk["month"], chunk["nmae"], marker="o", label=experiment)
    plt.xticks(range(1, 13))
    plt.ylabel("Macro nMAE")
    plt.xlabel("Month")
    plt.title("Monthly validation error")
    plt.legend()
    _save_figure(output_dir / "monthly_error_comparison.png")

    wind = high_wind[high_wind["fold"] == "fold_b"]
    pivot = wind.pivot(index="ablation_label", columns="group_id", values="nmae").sort_index()
    pivot.plot(kind="bar", figsize=(9, 5), color=["#4472C4", "#ED7D31", "#70AD47"])
    plt.ylabel("High-wind nMAE")
    plt.xlabel("Ablation")
    plt.title("Fold B high-wind error")
    plt.legend(title="Group")
    _save_figure(output_dir / "high_wind_error_comparison.png")

    best = predictions[(predictions["fold"] == "fold_b") & (predictions["experiment_id"] == best_experiment)].copy()
    if best.empty:
        best = predictions[predictions["fold"] == "fold_b"].copy()
    plt.figure(figsize=(6, 6))
    for group_id, chunk in best.groupby("group_id"):
        plt.scatter(chunk["y_true_kwh"], chunk["y_pred_kwh"], s=3, alpha=0.2, label=f"group {group_id}")
    limit = float(max(best["y_true_kwh"].max(), best["y_pred_kwh"].max()))
    plt.plot([0, limit], [0, limit], "k--", linewidth=1)
    plt.xlabel("Actual kWh")
    plt.ylabel("Predicted kWh")
    plt.title(f"Prediction scatter: {best_experiment}")
    plt.legend()
    _save_figure(output_dir / "prediction_scatter_best_model.png")

    plt.figure(figsize=(11, 5))
    for group_id, chunk in best.groupby("group_id"):
        chunk = chunk.sort_values("forecast_kst_dtm")
        residual = chunk["y_pred_kwh"] - chunk["y_true_kwh"]
        rolling = residual.rolling(24 * 7, min_periods=24).mean()
        plt.plot(pd.to_datetime(chunk["forecast_kst_dtm"]), rolling, linewidth=0.8, label=f"group {group_id}")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("7-day rolling residual (kWh)")
    plt.xlabel("Validation time")
    plt.title(f"Residual time series: {best_experiment}")
    plt.legend()
    _save_figure(output_dir / "residual_timeseries_best_model.png")
