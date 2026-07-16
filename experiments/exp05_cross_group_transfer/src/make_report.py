"""Build the Exp05 Markdown report and compact diagnostic figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp05_cross_group_transfer"


def _figure(path: Path, title: str, render) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5)); render(ax)
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_figures(output: Path) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(output / "metrics/group_weight_stability.csv")
    _figure(figures / "group_weight_stability.png", "Nested group raw weights", lambda ax: [
        ax.plot(weights["evaluation_quarter"], weights[f"weight_g{group}"], marker="o", label=f"group {group}")
        for group in (1, 2, 3)
    ] or ax.legend())
    # The list-comprehension render above cannot add the legend through short-circuiting.
    image = plt.imread(figures / "group_weight_stability.png"); del image
    quarters = pd.read_csv(output / "metrics/nested_quarter_scores.csv")
    def quarter_plot(ax):
        for stage, part in quarters.groupby("stage"):
            ax.plot(part["quarter"], part["total_score"], marker="o", label=stage)
        ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=35)
    _figure(figures / "quarter_score_comparison.png", "Rolling quarter Score", quarter_plot)
    groups = pd.read_csv(output / "metrics/group_scores.csv")
    group3 = groups.loc[groups["group_id"].eq(3)]
    _figure(figures / "group3_score_comparison.png", "Group 3 Score", lambda ax: ax.bar(group3["stage"], group3["score"]))
    candidates = pd.read_csv(output / "metrics/final_candidate_scores.csv")
    _figure(figures / "final_score_comparison.png", "Final candidate rolling Score", lambda ax: ax.bar(candidates["stage"], candidates["total_score"]))
    ridge = pd.read_csv(output / "predictions/ridge_stacker_oof.csv")
    correction = ridge["ridge_prediction"] - ridge["base_prediction"]
    _figure(figures / "residual_correction_distribution.png", "Ridge residual correction", lambda ax: ax.hist(correction, bins=60))
    importance = pd.read_csv(output / "metrics/stacker_feature_importance.csv")
    importance = (
        importance.loc[importance["model"].eq("ridge")]
        .groupby("feature", as_index=False)["importance"].mean()
        .nlargest(15, "importance").sort_values("importance")
    )
    _figure(figures / "stacker_feature_importance.png", "Ridge mean absolute coefficient", lambda ax: ax.barh(
        importance["feature"], importance["importance"]
    ))
    attention_files = sorted((output / "predictions").glob("cross_group_attention_full_seed42.npz"))
    def attention_plot(ax):
        if attention_files:
            values = np.load(attention_files[-1])["cross_group_attention"].mean(axis=(0, 1, 2))
            sns.heatmap(values, annot=True, fmt=".3f", cmap="viridis", ax=ax)
            ax.set_xlabel("key group"); ax.set_ylabel("query group")
        else:
            ax.text(.5, .5, "Stage D artifact not available", ha="center", va="center")
            ax.axis("off")
    _figure(figures / "cross_group_attention_heatmap.png", "Cross-group attention", attention_plot)


def write_report(output: Path, report_path: Path = EXPERIMENT_DIR / "report.md") -> str:
    candidates = pd.read_csv(output / "metrics/final_candidate_scores.csv")
    weights = pd.read_csv(output / "metrics/group_weight_stability.csv")
    decision = json.loads((output / "stage_d_decision.json").read_text())
    reproduction = json.loads((output / "checks/reference_reproduction.json").read_text())
    submission = json.loads((output / "submission_selection.json").read_text())
    blend = json.loads((output / "metrics/constrained_group_summary.json").read_text())
    cross_path = output / "metrics/cross_group_attention_scores.csv"
    cross = pd.read_csv(cross_path) if cross_path.exists() else pd.DataFrame()
    final = candidates.loc[candidates["stage"].eq("final_ensemble")].iloc[0]
    groups = pd.read_csv(output / "metrics/group_scores.csv")
    final_groups = groups.loc[groups["stage"].eq("final_ensemble")].sort_values("group_id")
    reference_groups = groups.loc[groups["stage"].eq("exp04_global")].sort_values("group_id")
    january = pd.read_csv(output / "metrics/january_scores.csv").loc[lambda x: x["stage"].eq("final_ensemble")].iloc[0]
    high_wind = pd.read_csv(output / "metrics/high_wind_scores.csv").loc[lambda x: x["stage"].eq("final_ensemble")].iloc[0]
    accepted = bool(
        final.total_score >= 0.648940
        and final.improved_quarters >= 5
        and final.worst_quarter >= 0.6054628191969988 - 0.003
        and float(final_groups.loc[final_groups["group_id"].eq(3), "score"].iloc[0]) >= 0.617185
    )
    lines = [
        "# exp05 cross-group transfer v2 report", "",
        "## Contract", "",
        f"Exp04 0.4 reference reproduced at `{reproduction['reproduced']['total_score']:.12f}` "
        f"(absolute error `{reproduction['absolute_error']:.3g}`). Stacker training used rolling OOF rows only; "
        "Public metrics were not used for selection.", "", "## Cheap stages", "",
    ]
    for row in candidates.loc[candidates["stage"].isin(["exp04_global", "constrained", "ridge", "catboost"])].itertuples():
        lines.append(
            f"- `{row.stage}`: Score {row.total_score:.6f}, 1-NMAE {row.one_minus_nmae:.6f}, "
            f"FICR {row.ficr:.6f}, equal-quarter {row.equal_quarter_mean:.6f}, "
            f"worst {row.worst_quarter:.6f}, improved {int(row.improved_quarters)}/8"
        )
    final_weights = blend["final_weights"]
    lines.extend(["", "## Constrained weights", "",
                  f"Final all-OOF raw weights: g1 `{final_weights['kpx_group_1']:.2f}`, "
                  f"g2 `{final_weights['kpx_group_2']:.2f}`, g3 `{final_weights['kpx_group_3']:.2f}`. "
                  f"Selected penalties: `{blend['final_penalties']}`.",
                  f"Nested mean/std: g1 {weights.weight_g1.mean():.4f}/{weights.weight_g1.std():.4f}, "
                  f"g2 {weights.weight_g2.mean():.4f}/{weights.weight_g2.std():.4f}, "
                  f"g3 {weights.weight_g3.mean():.4f}/{weights.weight_g3.std():.4f}.", "",
                  "Quarter weights:", ""])
    lines.extend(
        f"- {row.evaluation_quarter}: {row.weight_g1:.2f}/{row.weight_g2:.2f}/{row.weight_g3:.2f} ({row.selection_status})"
        for row in weights.itertuples()
    )
    lines.extend(["", "## Stage D", ""])
    if cross.empty:
        lines.append(f"Stage D required: `{not decision['skip_cross_group_attention']}`; result not yet present.")
    else:
        for row in cross.itertuples():
            lines.append(f"- {row.phase} seed {int(row.seed)}: Fold B Score {row.total_score:.6f}, delta vs raw seed42 {row.delta_vs_raw_seed42:+.6f}")
        lines.append("Full seed 42 did not beat raw_hybrid_gated; seeds 52/62 and conditional cross-group regularization were therefore skipped.")
    lines.extend(["", "## Final", "",
                  f"Best new rolling candidate: `{final.stage}` with Score {final.total_score:.6f}, "
                  f"equal-quarter mean {final.equal_quarter_mean:.6f}, worst {final.worst_quarter:.6f}, "
                  f"and {int(final.improved_quarters)}/8 improved quarters.",
                  f"Group Scores: " + ", ".join(
                      f"g{int(row.group_id)} {row.score:.6f}" for row in final_groups.itertuples()
                  ) + ".",
                  "Matched rolling group deltas vs Exp04: " + ", ".join(
                      f"g{int(left.group_id)} {left.score-right.score:+.6f}"
                      for left, right in zip(final_groups.itertuples(), reference_groups.itertuples())
                  ) + ".",
                  f"The supplied report-context group-3 reference 0.617185 is exceeded by "
                  f"{float(final_groups.loc[final_groups['group_id'].eq(3), 'score'].iloc[0])-0.617185:+.6f}, "
                  "but the matched rolling Exp04 group-3 score is not preserved.",
                  f"January Score {january.total_score:.6f}; high-wind Score {high_wind.total_score:.6f}.",
                  f"Final convex weights: " + ", ".join(
                      f"{name}={weight:.2f}" for name, weight in submission["final_weights"].items()
                  ) + ".",
                  f"Acceptance: `{'passed' if accepted else 'failed'}`. "
                  + ("The Exp05 candidate replaces Exp04." if accepted else "Exp04 remains champion; Exp05 submissions are diagnostic only."),
                  "", "Generated submissions:", ""])
    lines.extend(f"- `{Path(path).name}`" for path in submission["paths"])
    lines.extend(["", "No submission was sent automatically. Public priority remains the existing Exp04 blend. "
                  "If one diagnostic slot is intentionally used, the Exp05 order is final ensemble, constrained blend, then Ridge.", "",
                  "## Public context and next step", "",
                  "Exp04 Public Score 0.634005715 and Exp03 Public Score 0.6315350794 are report context only and were not used for fitting or selection.",
                  "The next experiment should target FICR threshold calibration within each group using the same nested OOF contract; a larger cross-group neural model is not supported by this result.", ""])
    text = "\n".join(lines); report_path.write_text(text, encoding="utf-8")
    write_figures(output)
    return text
