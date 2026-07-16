"""Create exp04 figures and a result-driven Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def make_figures(output_root: Path) -> None:
    sns.set_theme(style="whitegrid")
    figures = output_root / "figures"
    ablation = _read(output_root / "metrics/ablation_scores.csv")
    if not ablation.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); order = ablation.sort_values("total_score")
        ax.barh(order["model_id"], order["total_score"]); ax.set_xlabel("Fold B official Score")
        _save(fig, figures / "ablation_official_score.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ablation["one_minus_nmae"], ablation["ficr"])
        for row in ablation.itertuples(): ax.annotate(row.model_id, (row.one_minus_nmae, row.ficr), fontsize=8)
        ax.set(xlabel="1-NMAE", ylabel="FICR")
        _save(fig, figures / "nmae_ficr_tradeoff.png")
    rolling = _read(output_root / "metrics/rolling_quarter_scores.csv")
    exp03_rolling_path = output_root / "predictions/exp03_reference_predictions.csv"
    if not rolling.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for model_id, part in rolling.groupby("model_id"):
            ax.plot(part["quarter"], part["total_score"], marker="o", label=model_id)
        ax.tick_params(axis="x", rotation=45); ax.legend()
        _save(fig, figures / "rolling_quarter_comparison.png")
    group = _read(output_root / "metrics/group_scores.csv")
    if not group.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=group, x="group_id", y="score", hue="model_id", ax=ax)
        _save(fig, figures / "group_score_comparison.png")
    for metric_file, figure_name in (
        ("january_scores.csv", "january_comparison.png"),
        ("high_wind_scores.csv", "high_wind_comparison.png"),
    ):
        table = _read(output_root / "metrics" / metric_file)
        if not table.empty:
            fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=table, x="model_id", y="total_score", hue="fold", ax=ax)
            ax.tick_params(axis="x", rotation=30); _save(fig, figures / figure_name)
    residual = _read(output_root / "metrics/residual_correlations.csv")
    if not residual.empty:
        fig, ax = plt.subplots(figsize=(7, 4)); view = residual.loc[residual["slice"].isin(["overall", "group"])]
        ax.bar(np.arange(len(view)), view["residual_pearson"]); ax.set_ylabel("Residual Pearson")
        _save(fig, figures / "residual_correlation.png")
    blend = _read(output_root / "metrics/blend_search.csv")
    if not blend.empty:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(blend["raw_weight"], blend["total_score"])
        ax.set(xlabel="Raw model weight", ylabel="Rolling official Score")
        _save(fig, figures / "blend_search.png")
    for source in ("ldaps", "gfs"):
        attention = _read(output_root / f"attention/{source}_attention_by_group.csv")
        if not attention.empty:
            pivot = attention.pivot(index="group_id", columns="grid_id", values="mean_attention")
            fig, ax = plt.subplots(figsize=(10, 3)); sns.heatmap(pivot, cmap="viridis", ax=ax)
            _save(fig, figures / f"{source}_attention_heatmap.png")
    gate = _read(output_root / "attention/source_gate_by_group.csv")
    if not gate.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for group_id, part in gate.groupby("group_id"):
            ax.plot(part["lead_time_h"], part["ldaps_gate_mean"], label=f"group {group_id}")
        ax.set(xlabel="Lead time (h)", ylabel="LDAPS gate"); ax.legend()
        _save(fig, figures / "source_gate_by_lead_time.png")


def write_report(output_root: Path = DEFAULT_OUTPUT) -> Path:
    ablation = _read(output_root / "metrics/ablation_scores.csv")
    seed = _read(output_root / "metrics/seed_scores.csv")
    fold = _read(output_root / "metrics/fold_scores.csv")
    training = _read(output_root / "metrics/training_runs.csv")
    group_scores = _read(output_root / "metrics/group_scores.csv")
    january = _read(output_root / "metrics/january_scores.csv")
    high_wind = _read(output_root / "metrics/high_wind_scores.csv")
    rolling = _read(output_root / "metrics/rolling_quarter_scores.csv")
    blend = _read(output_root / "metrics/blend_search.csv")
    residual = _read(output_root / "metrics/residual_correlations.csv")
    attention_ldaps = _read(output_root / "attention/ldaps_attention_by_group.csv")
    attention_gfs = _read(output_root / "attention/gfs_attention_by_group.csv")
    gate = _read(output_root / "attention/source_gate_by_group.csv")
    selection = json.loads((output_root / "architecture_selection.json").read_text())
    full = json.loads((output_root / "full_training_summary.json").read_text())
    best = selection["selected_architecture"]
    ablation_lines = "\n".join(
        f"- `{row.model_id}`: Score {row.total_score:.6f}, 1-NMAE {row.one_minus_nmae:.6f}, FICR {row.ficr:.6f}"
        for row in ablation.sort_values("total_score", ascending=False).itertuples()
    )
    fold_lines = "\n".join(
        f"- {row.fold}: Score {row.total_score:.6f}, 1-NMAE {row.one_minus_nmae:.6f}, FICR {row.ficr:.6f}"
        for row in fold.loc[fold["model_id"].eq(best)].sort_values("fold").itertuples()
    )
    comparison_ids = ["exp03_ficr_aware", best, "exp03_raw_blend"]
    fold_b_lines = "\n".join(
        f"- `{row.model_id}`: Score {row.total_score:.6f}, 1-NMAE {row.one_minus_nmae:.6f}, FICR {row.ficr:.6f}"
        for row in fold.loc[fold["fold"].eq("fold_b") & fold["model_id"].isin(comparison_ids)]
        .sort_values("total_score", ascending=False).itertuples()
    )
    selected_seeds = training.loc[
        training["stage"].eq("full") & training["fold"].eq("fold_b")
        & training["model_id"].eq(best) & training["seed"].isin([42, 52, 62]), "total_score"
    ]
    seed_mean, seed_std = float(selected_seeds.mean()), float(selected_seeds.std())
    raw_rolling = rolling.loc[rolling["model_id"].eq(best)]
    rolling_mean = float(raw_rolling["total_score"].mean())
    rolling_worst = float(raw_rolling["total_score"].min())
    rolling_pivot = rolling.pivot(index="quarter", columns="model_id", values="total_score")
    raw_improved = int((rolling_pivot[best] > rolling_pivot["exp03_ficr_aware"]).sum())
    blend_improved = int((rolling_pivot["exp03_raw_blend"] > rolling_pivot["exp03_ficr_aware"]).sum())
    blend_quarter_mean = float(rolling_pivot["exp03_raw_blend"].mean())
    blend_quarter_worst = float(rolling_pivot["exp03_raw_blend"].min())
    best_blend = blend.sort_values("total_score", ascending=False).iloc[0]
    exp03_only = blend.loc[blend["raw_weight"].eq(0.0)].iloc[0]
    overall_corr = residual.loc[residual["slice"].eq("overall"), "residual_pearson"].iloc[0]
    def top_grids(table: pd.DataFrame) -> str:
        if table.empty: return "not available"
        return ", ".join(
            f"G{int(row.group_id)}→grid {int(row.grid_id)} ({row.mean_attention:.3f})"
            for row in table.sort_values(["group_id", "mean_attention"], ascending=[True, False]).groupby("group_id").head(1).itertuples()
        )
    gate_text = (
        "not used by the selected architecture" if gate.empty
        else (
            f"mean LDAPS gate {gate['ldaps_gate_mean'].mean():.3f}; "
            f"lead 12-19h {gate.loc[gate['lead_time_h'].between(12, 19), 'ldaps_gate_mean'].mean():.3f}, "
            f"lead 20-27h {gate.loc[gate['lead_time_h'].between(20, 27), 'ldaps_gate_mean'].mean():.3f}, "
            f"lead 28-35h {gate.loc[gate['lead_time_h'].between(28, 35), 'ldaps_gate_mean'].mean():.3f}"
        )
    )
    grid13 = []
    if not attention_ldaps.empty:
        for group_id, part in attention_ldaps.groupby("group_id"):
            ranked = part.sort_values("mean_attention", ascending=False).reset_index(drop=True)
            row = ranked.loc[ranked["grid_id"].eq(13)].iloc[0]
            rank = int(ranked.index[ranked["grid_id"].eq(13)][0] + 1)
            grid13.append(f"G{int(group_id)} rank {rank} ({row.mean_attention:.3f})")
    def slice_lines(table: pd.DataFrame, label: str) -> str:
        view = table.loc[table["fold"].eq("fold_b") & table["model_id"].isin(comparison_ids)]
        return "\n".join(
            f"- {label} `{row.model_id}`: {row.total_score:.6f}"
            for row in view.sort_values("total_score", ascending=False).itertuples()
        )
    group_view = group_scores.loc[
        group_scores["fold"].eq("fold_b") & group_scores["model_id"].isin(comparison_ids)
    ]
    group_lines = "\n".join(
        f"- group {int(row.group_id)} `{row.model_id}`: {row.score:.6f}"
        for row in group_view.sort_values(["group_id", "score"], ascending=[True, False]).itertuples()
    )
    report = f"""# exp04 raw-grid spatiotemporal report

## Contract

LDAPS dynamic tensor is `[1096, 24, 16, 16]` and GFS is `[1096, 24, 9, 26]` before variant channel selection. Static group tensors are `[3, 16, 11]` and `[3, 9, 11]`. Dynamic imputation, clipping, and scaling were fit on each fold's training blocks only. SCADA remained an auxiliary target and was never included in model input.

## B-F ablation

{ablation_lines}

Selected architecture: `{best}`.

## Official validation

{fold_lines}

Fold B comparison at the rolling-selected raw weight:

{fold_b_lines}

The selected raw seed Score mean/std is {seed_mean:.6f}/{seed_std:.6f}. Exp03's Public Score 0.631535, 1-NMAE 0.865998, and FICR 0.397072 were report context only and were not used for selection.

## Rolling and ensemble

Raw rolling mean Score is {rolling_mean:.6f}; worst quarter is {rolling_worst:.6f}; it improves {raw_improved}/8 quarters. Residual Pearson versus Exp03 is {overall_corr:.6f}. The best global blend uses raw weight {best_blend.raw_weight:.3f} and reaches rolling aggregate Score {best_blend.total_score:.6f}, a {best_blend.total_score-exp03_only.total_score:+.6f} gain over Exp03-only. Its equal-quarter mean/worst are {blend_quarter_mean:.6f}/{blend_quarter_worst:.6f}, and it improves {blend_improved}/8 individual quarters.

## Slice results

{group_lines}

{slice_lines(january, 'January')}

{slice_lines(high_wind, 'high-wind')}

## Attention

- LDAPS: {top_grids(attention_ldaps)}
- GFS: {top_grids(attention_gfs)}
- LDAPS grid 13: {', '.join(grid13)}
- Source gate: {gate_text}

Attention is interpretation-only and was not used as a selection metric.

## Full train and submissions

Full training used epochs={full['epochs']} and seeds={full['seeds']} on devices={full['devices']}. Generated submissions:

""" + "\n".join(f"- `{Path(path).name}`" for path in full["submissions"]) + """

Submission priority is the Exp03/raw blend when its leakage-safe rolling gain is positive, followed by the raw-only model. No submission was sent automatically.
"""
    path = output_root / "report.md"; path.write_text(report, encoding="utf-8")
    make_figures(output_root)
    return path


if __name__ == "__main__":
    print(write_report())
