"""Generate Exp06 figures and a decision-oriented Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp06_ficr_threshold_calibration"


def _optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _figure(path: Path, title: str, render) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8)); render(ax); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _empty(ax, text: str) -> None:
    ax.text(.5, .5, text, ha="center", va="center"); ax.axis("off")


def write_figures(output: Path) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    transitions = pd.read_csv(output / "metrics/tier_transition_matrix.csv")
    ridge = transitions.loc[transitions["candidate"].eq("exp05_ridge")].groupby(
        ["from_tier", "to_tier"], as_index=False
    )["count"].sum()
    pivot = ridge.pivot(index="from_tier", columns="to_tier", values="count").fillna(0)
    _figure(figures / "tier_transition_heatmap.png", "Exp04 → Ridge tier transitions", lambda ax: sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="magma", ax=ax
    ))
    margins = pd.read_csv(output / "metrics/threshold_margin_samples.csv")
    _figure(figures / "threshold_margin_histogram.png", "Official threshold margins", lambda ax: [
        ax.hist(part["normalized_error"], bins=50, alpha=.4, label=model)
        for model, part in margins.loc[margins["model"].isin(["exp04_global", "exp05_ridge"])].groupby("model")
    ] and ax.legend())
    advantage = pd.read_csv(output / "metrics/regime_model_advantage.csv")
    pred_cf = advantage.loc[advantage["regime_dimension"].eq("pred_cf_band")]
    pred_plot = pred_cf.groupby("regime_value", as_index=False)["raw_win_rate"].mean()
    _figure(figures / "model_advantage_by_pred_cf.png", "Raw win rate by predicted CF", lambda ax: ax.bar(
        pred_plot["regime_value"], pred_plot["raw_win_rate"]
    ))
    oracle = pd.read_csv(output / "metrics/oracle_headroom_by_group.csv")
    oracle_plot = oracle.loc[oracle["model"].ne("exp04_global")]
    _figure(figures / "oracle_headroom_by_group.png", "Oracle headroom by group", lambda ax: sns.barplot(
        data=oracle_plot, x="group_id", y="headroom_vs_exp04", hue="model", ax=ax
    ))
    scores = pd.read_csv(output / "metrics/piecewise_nested_scores.csv")
    def parameter_plot(ax):
        values = []
        for row in scores.itertuples():
            for parameter in json.loads(row.parameters):
                values.append({"quarter": row.evaluation_quarter, **parameter})
        frame = pd.DataFrame(values)
        if frame.empty: return _empty(ax, "No piecewise parameters")
        for keys, part in frame.groupby(["target", "bin"]):
            ax.plot(part["quarter"], part["scale"], marker="o", label=f"g{keys[0][-1]} b{keys[1]}")
        ax.legend(fontsize=6, ncol=3); ax.tick_params(axis="x", rotation=35)
    _figure(figures / "piecewise_parameter_stability.png", "Nested piecewise scales", parameter_plot)
    gate = _optional_csv(output / "metrics/gate_weight_stability.csv")
    def gate_group(ax):
        part = gate.loc[gate.get("slice", pd.Series(dtype=str)).eq("group")] if not gate.empty else gate
        if part.empty: return _empty(ax, "Gate skipped: deployable headroom < 0.003")
        sns.lineplot(data=part, x="quarter", y="mean_raw_weight", hue="target", marker="o", ax=ax)
    _figure(figures / "gate_weight_by_group.png", "Gate weight by group", gate_group)
    def gate_lead(ax):
        part = gate.loc[gate.get("slice", pd.Series(dtype=str)).eq("lead_time")] if not gate.empty else gate
        if part.empty: return _empty(ax, "Gate skipped: deployable headroom < 0.003")
        sns.lineplot(data=part, x="slice_value", y="mean_raw_weight", hue="quarter", ax=ax)
    _figure(figures / "gate_weight_by_lead_time.png", "Gate weight by lead time", gate_lead)
    quarters = pd.read_csv(output / "metrics/quarter_scores.csv")
    def quarter_plot(ax):
        for model, part in quarters.groupby("model"):
            ax.plot(part["quarter"], part["total_score"], marker="o", label=model)
        ax.legend(); ax.tick_params(axis="x", rotation=35)
    _figure(figures / "quarter_score_comparison.png", "Nested rolling quarter Score", quarter_plot)
    candidates = pd.read_csv(output / "metrics/final_candidate_scores.csv")
    _figure(figures / "final_score_comparison.png", "Final single-rule candidates", lambda ax: ax.bar(
        candidates["model"], candidates["total_score"]
    ))


def _transition_summary(transitions: pd.DataFrame, candidate: str) -> dict:
    part = transitions.loc[transitions["candidate"].eq(candidate)]
    counts = part.groupby(["from_tier", "to_tier"])["count"].sum()
    result = {f"{left}->{right}": int(counts.get((left, right), 0))
              for left in ("tier_4", "tier_3", "tier_0") for right in ("tier_4", "tier_3", "tier_0")}
    result["reward_delta_energy"] = float(part["reward_delta_energy"].sum())
    return result


def write_report(output: Path, report_path: Path = EXPERIMENT_DIR / "report.md") -> str:
    candidates = pd.read_csv(output / "metrics/final_candidate_scores.csv")
    transitions = pd.read_csv(output / "metrics/tier_transition_matrix.csv")
    margins = pd.read_csv(output / "metrics/threshold_margin_samples.csv")
    oracle = pd.read_csv(output / "metrics/oracle_headroom.csv")
    advantage = pd.read_csv(output / "metrics/regime_model_advantage.csv")
    model = json.loads((output / "checks/final_piecewise_model.json").read_text())
    selection = json.loads((output / "final_selection.json").read_text())
    acceptance = json.loads((output / "acceptance.json").read_text())
    submissions = json.loads((output / "submission_manifest.json").read_text())
    piecewise = candidates.loc[candidates["model"].eq("piecewise")].iloc[0]
    reference = candidates.loc[candidates["model"].eq("exp04_global")].iloc[0]
    piecewise_groups = pd.read_csv(output / "metrics/group_scores.csv").loc[lambda x: x["model"].eq("piecewise")]
    reference_groups = pd.read_csv(output / "metrics/group_scores.csv").loc[lambda x: x["model"].eq("exp04_global")]
    nested_predictions = pd.read_csv(output / "predictions/piecewise_nested_oof.csv")
    change_p95 = float((
        nested_predictions["piecewise_prediction"]-nested_predictions["global_blend_prediction"]
    ).abs().div(nested_predictions["capacity_kwh"]).quantile(.95))
    parameter_rows = []
    for row in pd.read_csv(output / "metrics/piecewise_nested_scores.csv").itertuples():
        for parameter in json.loads(row.parameters):
            parameter_rows.append({"quarter": row.evaluation_quarter, **parameter})
    parameter_frame = pd.DataFrame(parameter_rows)
    parameter_stability = parameter_frame.groupby(["target", "bin"])[["scale", "offset_fraction"]].std(ddof=0)
    january = pd.read_csv(output / "metrics/january_scores.csv").loc[lambda x: x["model"].eq("piecewise")].iloc[0]
    high_wind = pd.read_csv(output / "metrics/high_wind_scores.csv").loc[lambda x: x["model"].eq("piecewise")].iloc[0]
    margin_reference = margins.loc[margins["model"].eq("exp04_global")]
    oracle_overall = oracle.loc[oracle["slice"].eq("overall")]
    stable = advantage.loc[
        advantage["samples"].ge(100)
        & advantage["quarter_raw_win_rate_std"].le(.10)
        & ((advantage["raw_win_rate"] >= .60) | (advantage["exp03_win_rate"] >= .60))
    ]
    lines = [
        "# exp06 FICR threshold calibration report", "", "## Contract", "",
        "Exp04 global blend was reproduced at `0.647439599391` with error below `1e-8`. "
        "The official scorer hash and exact 6%/8% reward aggregation matched. Public results were not used.",
        "", "## Threshold audit", "",
    ]
    for candidate in ("exp05_ridge", "exp05_catboost", "exp05_final"):
        value = _transition_summary(transitions, candidate)
        lines.append(
            f"- `{candidate}`: 4→3 {value['tier_4->tier_3']}, 4→0 {value['tier_4->tier_0']}, "
            f"3→4 {value['tier_3->tier_4']}, 0→4 {value['tier_0->tier_4']}, "
            f"energy-weighted reward delta {value['reward_delta_energy']:+.0f}"
        )
    lines.extend(["", f"Exp04 boundary samples: 6% ±0.5pp `{int(margin_reference.near_6pct.sum())}`, "
                  f"8% ±0.5pp `{int(margin_reference.near_8pct.sum())}`.", "",
                  "## Oracle and regimes", ""])
    for row in oracle_overall.itertuples():
        lines.append(f"- `{row.model}`: Score {row.total_score:.6f}, headroom {row.headroom_vs_exp04:+.6f}")
    lines.extend(["", f"Stable model-win regimes meeting the diagnostic support/stability rule: `{len(stable)}`.", "",
                  "## Piecewise affine", "",
                  f"Selected band scheme: `{model['scheme']}`; penalty `{model['penalty']}`.",
                  f"Final predicted-CF boundaries: `{model['boundaries']}`."])
    for parameter in model["parameters"]:
        lines.append(
            f"- {parameter['target']} bin {parameter['bin']}: scale {parameter['scale']:.4f}, "
            f"offset {parameter['offset_fraction']:+.4f} capacity"
        )
    lines.extend(["", f"Score {piecewise.total_score:.6f} ({piecewise.total_score-reference.total_score:+.6f}), "
                  f"1-NMAE {piecewise.one_minus_nmae:.6f}, FICR {piecewise.ficr:.6f}, "
                  f"equal-quarter {piecewise.equal_quarter_mean:.6f}, worst {piecewise.worst_quarter:.6f}, "
                  f"improved {int(piecewise.improved_quarters)}/8.",
                  "Group Scores: " + ", ".join(
                      f"g{int(row.group_id)} {row.score:.6f}" for row in piecewise_groups.itertuples()
                  ) + ".",
                  f"Matched group-3 delta vs Exp04: "
                  f"{float(piecewise_groups.loc[piecewise_groups['group_id'].eq(3),'score'].iloc[0]-reference_groups.loc[reference_groups['group_id'].eq(3),'score'].iloc[0]):+.6f}.",
                  f"FICR delta {piecewise.ficr-reference.ficr:+.6f}; 1-NMAE delta "
                  f"{piecewise.one_minus_nmae-reference.one_minus_nmae:+.6f}. Change p95 `{change_p95:.6f}` capacity.",
                  f"Nested parameter std mean/max: scale {parameter_stability.scale.mean():.4f}/{parameter_stability.scale.max():.4f}, "
                  f"offset {parameter_stability.offset_fraction.mean():.4f}/{parameter_stability.offset_fraction.max():.4f}.",
                  f"January Score {january.total_score:.6f}; high-wind Score {high_wind.total_score:.6f}.",
                  f"Piecewise acceptance conditions: `{acceptance['piecewise']['conditions']}`.", "",
                  "## Gate and final decision", "",
                  f"Gate executed: `{not _optional_csv(output / 'metrics/gate_nested_scores.csv').empty}`. "
                  f"Gate acceptance: `{acceptance['gate']}`.",
                  f"Selected deployable rule: `{selection['selected_model']}`; accepted new rule: `{selection['accepted_new_rule']}`.",
                  "", "Submissions:", ""])
    lines.extend(f"- `{Path(path).name}`" for path in submissions["submissions"])
    lines.extend(["", f"Diagnostic only: `{submissions['diagnostic_only']}`. No submission was sent automatically.", "",
                  "## Next direction", "",
                  "If threshold calibration does not clear acceptance, retain Exp04 and focus on training-time FICR threshold robustness rather than another residual stacker or larger cross-group model.", ""])
    text = "\n".join(lines); report_path.write_text(text, encoding="utf-8")
    write_figures(output); return text
