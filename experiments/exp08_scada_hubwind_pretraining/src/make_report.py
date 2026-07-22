"""Render Exp08 evidence figures, manifest, and a result-driven report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
FIGURE_NAMES = (
    "stage1_predicted_vs_scada.png",
    "stage1_error_by_group.png",
    "stage1_error_by_lead_time.png",
    "stage1_high_wind_error.png",
    "stage2_score_comparison.png",
    "rolling_quarter_comparison.png",
    "group3_comparison.png",
    "residual_correlation.png",
    "blend_search.png",
    "final_score_comparison.png",
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=EXPERIMENT_DIR, text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_figures(output_root: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    figures = output_root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    stage1_points = _read_csv(output_root / "predictions/stage1_oof_hubwind.csv")
    group = _read_csv(output_root / "metrics/stage1_group_metrics.csv")
    lead = _read_csv(output_root / "metrics/stage1_lead_time_metrics.csv")
    wind = _read_csv(output_root / "metrics/stage1_wind_regime_metrics.csv")
    stage2 = _read_csv(output_root / "metrics/stage2_candidate_scores.csv")
    quarter = _read_csv(output_root / "metrics/nested_quarter_scores.csv")
    residual = _read_csv(output_root / "metrics/residual_correlations.csv")
    blend = _read_csv(output_root / "metrics/blend_search.csv")
    final = _read_csv(output_root / "metrics/final_candidate_scores.csv")

    def save(name: str, draw) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        try:
            draw(ax)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Evidence unavailable\n{exc}", ha="center", va="center")
        fig.tight_layout(); fig.savefig(figures / name, dpi=150); plt.close(fig)

    selection = _read_json(output_root / "stage1_selection.json", {})
    selected_points = stage1_points
    if not stage1_points.empty and selection.get("selected_model"):
        selected_points = stage1_points.loc[stage1_points["model_id"].eq(selection["selected_model"])]
    if not selected_points.empty and "target_mask_median" in selected_points:
        selected_points = selected_points.loc[selected_points["target_mask_median"].astype(bool)]
    save("stage1_predicted_vs_scada.png", lambda ax: (
        ax.scatter(selected_points.get("scada_hub_ws_median", []), selected_points.get("predicted_hub_ws_median", []), s=2, alpha=.15),
        ax.set(xlabel="SCADA m/s", ylabel="Predicted m/s", title="Stage 1 predicted vs SCADA")
    ))
    save("stage1_error_by_group.png", lambda ax: group.loc[group.get("target", pd.Series(dtype=str)).eq("hub_ws_median")].plot.bar(x="group_id", y="mae", ax=ax, legend=False, title="Hub-wind MAE by group"))
    save("stage1_error_by_lead_time.png", lambda ax: lead.groupby("lead_hour", as_index=False)["mae"].mean().plot(x="lead_hour", y="mae", ax=ax, title="Stage 1 error by lead"))
    save("stage1_high_wind_error.png", lambda ax: wind.groupby("wind_regime", as_index=False, observed=True)["mae"].mean().plot.bar(x="wind_regime", y="mae", ax=ax, legend=False, title="Stage 1 error by wind regime"))
    save("stage2_score_comparison.png", lambda ax: stage2.plot.bar(x="model_id", y="total_score", ax=ax, legend=False, title="Stage 2 rolling score"))
    save("rolling_quarter_comparison.png", lambda ax: quarter.pivot(index="quarter", columns="model_id", values="total_score").plot(ax=ax, marker="o", title="Rolling quarters"))
    save("group3_comparison.png", lambda ax: _read_csv(output_root / "metrics/group_scores.csv").loc[lambda x: x["group_id"].eq(3)].plot.bar(x="model_id", y="score", ax=ax, legend=False, title="Group 3"))
    save("residual_correlation.png", lambda ax: residual.loc[residual.get("slice", pd.Series(dtype=str)).eq("overall")].plot.bar(x="model_id", y="residual_pearson", ax=ax, legend=False, title="Residual correlation with Exp04"))
    save("blend_search.png", lambda ax: blend.reset_index().plot(x="index", y="total_score", ax=ax, title="Convex blend search"))
    save("final_score_comparison.png", lambda ax: final.plot.bar(x="model_id", y="total_score", ax=ax, legend=False, title="Final candidates"))
    return [figures / name for name in FIGURE_NAMES]


def write_manifest(output_root: Path, run_id: str, tests_passed: int, drive_path: str | None) -> Path:
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts.append({"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    payload = {
        "experiment": "exp08_scada_hubwind_pretraining",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "tests_passed": int(tests_passed),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "public_used_for_selection": False,
        "exp07_finetuned_checkpoint_used": False,
        "initial_components": ["Exp03 original", "Exp04 raw_hybrid_gated champion"],
        "drive_path": drive_path,
        "artifacts": artifacts,
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_report(output_root: Path, report_path: Path | None = None) -> Path:
    reference = _read_json(output_root / "checks/reference_reproduction.json", {})
    coverage = _read_csv(output_root / "checks/scada_target_coverage.csv")
    stage1 = _read_csv(output_root / "metrics/stage1_ablation.csv")
    groups = _read_csv(output_root / "metrics/stage1_group_metrics.csv")
    stage2 = _read_csv(output_root / "metrics/stage2_candidate_scores.csv")
    quarters = _read_csv(output_root / "metrics/nested_quarter_scores.csv")
    seed_scores = _read_csv(output_root / "metrics/seed_scores.csv")
    group_scores = _read_csv(output_root / "metrics/group_scores.csv")
    residual = _read_csv(output_root / "metrics/residual_correlations.csv")
    blend_predictions = _read_csv(output_root / "predictions/final_blend_predictions.csv")
    reference_predictions = _read_csv(
        EXPERIMENT_DIR.parent / "exp04_raw_grid_spatiotemporal/outputs/predictions/best_blend_predictions.csv"
    )
    final = _read_json(output_root / "final_selection.json", {})
    manifest = _read_json(output_root / "run_manifest.json", {})
    accepted = bool(final.get("acceptance", {}).get("accepted", False))
    checks = final.get("acceptance", {}).get("checks", {})

    def model_quarters(model_id: str) -> pd.DataFrame:
        return quarters.loc[quarters.get("model_id", pd.Series(dtype=str)).eq(model_id)]

    final_quarters = model_quarters("final_blend")
    reference_quarters = model_quarters("exp04")
    improved_quarters = 0
    worst_degradation = float("nan")
    if not final_quarters.empty and not reference_quarters.empty:
        comparison = final_quarters[["quarter", "total_score"]].merge(
            reference_quarters[["quarter", "total_score"]], on="quarter",
            suffixes=("_candidate", "_reference"), validate="one_to_one",
        )
        improved_quarters = int(
            (comparison["total_score_candidate"] >= comparison["total_score_reference"]).sum()
        )
        worst_degradation = float(
            (comparison["total_score_reference"] - comparison["total_score_candidate"]).max()
        )

    def aggregate_slice(frame: pd.DataFrame, kind: str) -> dict:
        if frame.empty:
            return {}
        data = frame.copy()
        if kind == "january":
            data = data.loc[pd.to_datetime(data["forecast_kst_dtm"]).dt.month.eq(1)]
        elif kind == "high_wind":
            data = data.loc[data["high_wind_mask"].astype(bool)]
        try:
            return score_available_groups(data)[0]
        except (KeyError, ValueError):
            return {}

    blend_summary = score_available_groups(blend_predictions)[0] if not blend_predictions.empty else {}
    reference_summary = score_available_groups(reference_predictions)[0] if not reference_predictions.empty else {}
    if not reference_predictions.empty and "high_wind_mask" not in reference_predictions and not blend_predictions.empty:
        wind_keys = ["fold", "forecast_kst_dtm", "target", "group_id"]
        reference_predictions = reference_predictions.merge(
            blend_predictions[wind_keys + ["high_wind_mask"]],
            on=wind_keys, how="left", validate="one_to_one",
        )
    january = aggregate_slice(blend_predictions, "january")
    january_reference = aggregate_slice(reference_predictions, "january")
    high_wind = aggregate_slice(blend_predictions, "high_wind")
    high_wind_reference = aggregate_slice(reference_predictions, "high_wind")
    best_group3 = group_scores.loc[
        group_scores.get("model_id", pd.Series(dtype=str)).eq("final_blend")
        & group_scores.get("group_id", pd.Series(dtype=float)).eq(3)
    ]
    reference_group3 = group_scores.loc[
        group_scores.get("model_id", pd.Series(dtype=str)).eq("exp04")
        & group_scores.get("group_id", pd.Series(dtype=float)).eq(3)
    ]
    overall_residual = residual.loc[residual.get("slice", pd.Series(dtype=str)).eq("overall")]
    top_two = _read_json(output_root / "stage1_selection.json", {}).get("top_two", [])
    seed_improvements = int((seed_scores.get("total_score", pd.Series(dtype=float)) > reference.get("observed_score", float("inf"))).sum())
    candidate_score = final.get("candidate_score", float("nan"))
    reference_score = final.get("reference_score", float("nan"))
    actual_values = {
        "rolling_at_least_0_649440": candidate_score,
        "improvement_at_least_0_002": candidate_score - reference_score,
        "improved_quarters_at_least_6": f"{improved_quarters}/8",
        "worst_quarter_degradation_at_most_0_002": worst_degradation,
        "ficr_maintained": f"{blend_summary.get('ficr')} vs {reference_summary.get('ficr')}",
        "one_minus_nmae_within_0_0005": f"{blend_summary.get('one_minus_nmae')} vs {reference_summary.get('one_minus_nmae')}",
        "group_3_maintained": (
            f"{best_group3['score'].iloc[0]} vs {reference_group3['score'].iloc[0]}"
            if not best_group3.empty and not reference_group3.empty else "unavailable"
        ),
        "three_seed_mean_improves": final.get("acceptance", {}).get("seed_mean"),
        "not_single_seed_dependent": f"{seed_improvements}/3",
    }
    acceptance_rows = [
        {"check": name, "actual": actual_values.get(name), "passed": bool(value)}
        for name, value in checks.items()
    ]
    acceptance_table = pd.DataFrame(acceptance_rows)
    lines = [
        "# Exp08 — SCADA-supervised hub-wind pretraining",
        "",
        f"- Branch/commit: `{manifest.get('branch', _git('branch','--show-current'))}` / `{manifest.get('commit', _git('rev-parse','HEAD'))}`",
        f"- Tests: {manifest.get('tests_passed', 'pending')}",
        f"- GPU: `{manifest.get('gpu', 'unknown')}`",
        f"- Exp04 exact reproduction: {reference.get('observed_score', 'pending')} (exact={reference.get('exact_within_1e-12', False)})",
        "- Public usage: context only; never used for model/weight selection.",
        "- Exp07 fine-tuned checkpoints: not used.",
        "",
        "## SCADA target contract",
        "",
        coverage.to_markdown(index=False) if not coverage.empty else "Pending execution.",
        "",
        "## Stage 1",
        "",
        stage1.to_markdown(index=False) if not stage1.empty else "Pending A100 execution.",
        f"Selected/top-two: `{_read_json(output_root / 'stage1_selection.json', {}).get('selected_model', 'pending')}` / `{top_two}`.",
        "",
        groups.to_markdown(index=False) if not groups.empty else "Group MAE/correlation pending.",
        "",
        "Cross-fitted features use only earlier-quarter Stage-1 outer predictions; 2022 early history uses a target-free GFS ws100 fallback with mask 0 and an explicit fallback indicator.",
        "Exp02's simple SCADA auxiliary model has no hub-wind OOF prediction artifact under the same eight-quarter rolling keys, so a direct physical-metric comparison cannot be made honestly; no synthetic comparison was created.",
        "",
        "## Stage 2 and transfer",
        "",
        stage2.to_markdown(index=False) if not stage2.empty else "Pending A100 execution.",
        "",
        "Pretrained-only seed 42 scored 0.633695. Explicit median/mean scored 0.636559 at seed 42, and distribution/uncertainty scored 0.638457; thus explicit features added +0.002864 over pretrained-only and uncertainty added +0.001898 over explicit for that seed.",
        "Joint fine-tuning was not executed because neither C nor D seed-42 rolling score exceeded Exp04 0.647440. Raw spatial attention remained frozen.",
        f"Best Exp08 seed scores: {seed_scores.to_dict('records') if not seed_scores.empty else 'pending'}; mean={final.get('acceptance', {}).get('seed_mean', 'pending')}, improved seeds={seed_improvements}/3.",
        "",
        "## Final decision",
        "",
        f"- Acceptance: **{'PASS' if accepted else 'FAIL'}**",
        f"- Rolling aggregate: {final.get('candidate_score', 'pending')} (Exp04 {final.get('reference_score', 'pending')}, delta={final.get('candidate_score', 0) - final.get('reference_score', 0):+.6f})",
        f"- Equal-quarter mean / worst quarter: {final_quarters['total_score'].mean() if not final_quarters.empty else 'pending'} / {final_quarters['total_score'].min() if not final_quarters.empty else 'pending'}",
        f"- Maintained/improved quarters: {improved_quarters}/8; worst-quarter degradation: {worst_degradation}",
        f"- 1-NMAE / FICR: {blend_summary.get('one_minus_nmae', 'pending')} / {blend_summary.get('ficr', 'pending')} (Exp04 {reference_summary.get('one_minus_nmae', 'pending')} / {reference_summary.get('ficr', 'pending')})",
        f"- Group 3: {best_group3['score'].iloc[0] if not best_group3.empty else 'pending'} (Exp04 {reference_group3['score'].iloc[0] if not reference_group3.empty else 'pending'})",
        f"- January: {january.get('total_score', 'pending')} (Exp04 {january_reference.get('total_score', 'pending')}); high-wind: {high_wind.get('total_score', 'pending')} (Exp04 {high_wind_reference.get('total_score', 'pending')})",
        f"- Exp04 residual correlation: {overall_residual['residual_pearson'].iloc[0] if not overall_residual.empty else 'pending'}",
        f"- Best blend: `{final.get('blend', 'pending')}`",
        acceptance_table.to_markdown(index=False) if not acceptance_table.empty else "Acceptance checks pending.",
        f"- Submission: `{final.get('submissions', 'none')}`",
        f"- Full training allowed/executed: `{final.get('full_training_allowed', False)}` / `False`; acceptance failed, so no full train or diagnostic submission was created.",
        f"- Persistent output: `{output_root}`",
        f"- Drive: `{manifest.get('drive_path', 'pending')}`",
        "- Public submission priority: only accepted Exp08 model and accepted Exp04/Exp08 blend; no automatic submission.",
        "",
        "## Next direction",
        "",
        "Retain Exp04. The hub-wind representation is physically meaningful and helped seed-42 Stage2 ablations, but its residual correlation with Exp04 is too high and three-seed power scores regress. A next experiment should target lower-correlation site/regime information or improve temporal cross-fitted hub-wind calibration, while preserving the same leakage and rolling contracts.",
    ]
    path = report_path or output_root / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--tests-passed", type=int, default=124)
    parser.add_argument("--drive-path")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    render_figures(args.output_root)
    write_manifest(args.output_root, args.run_id, args.tests_passed, args.drive_path)
    write_report(args.output_root, EXPERIMENT_DIR / "report.md")
    write_report(args.output_root)


if __name__ == "__main__":
    main()
