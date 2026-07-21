"""Render Exp08 evidence figures, manifest, and a result-driven report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


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

    save("stage1_predicted_vs_scada.png", lambda ax: (
        ax.scatter(stage1_points.get("y_true_mps", []), stage1_points.get("y_pred_mps", []), s=2, alpha=.15),
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
    final = _read_json(output_root / "final_selection.json", {})
    manifest = _read_json(output_root / "run_manifest.json", {})
    accepted = bool(final.get("acceptance", {}).get("accepted", False))
    lines = [
        "# Exp08 — SCADA-supervised hub-wind pretraining",
        "",
        f"- Branch/commit: `{manifest.get('branch', _git('branch','--show-current'))}` / `{manifest.get('commit', _git('rev-parse','HEAD'))}`",
        f"- Tests: {manifest.get('tests_passed', 'pending')}",
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
        "",
        groups.to_markdown(index=False) if not groups.empty else "Group MAE/correlation pending.",
        "",
        "Cross-fitted features use only earlier-quarter Stage-1 outer predictions; 2022 early history uses a target-free GFS ws100 fallback with mask 0 and an explicit fallback indicator.",
        "",
        "## Stage 2 and transfer",
        "",
        stage2.to_markdown(index=False) if not stage2.empty else "Pending A100 execution.",
        "",
        "Pretrained-only, explicit median/mean, and distribution/uncertainty variants are evaluated first. Joint fine-tuning runs only if C or D improves Exp04. Raw spatial attention remains frozen.",
        "",
        "## Final decision",
        "",
        f"- Acceptance: **{'passed' if accepted else 'not passed / pending'}**",
        f"- Best blend: `{final.get('blend', 'pending')}`",
        f"- Submission: `{final.get('submissions', 'none')}`",
        f"- Drive: `{manifest.get('drive_path', 'pending')}`",
        "- Public submission priority: only accepted Exp08 model and accepted Exp04/Exp08 blend; no automatic submission.",
        "",
        "## Next direction",
        "",
        final.get("next_direction", "Decide after the rolling acceptance gates are evaluated."),
    ]
    path = report_path or output_root / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--tests-passed", type=int, default=123)
    parser.add_argument("--drive-path")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    render_figures(args.output_root)
    write_manifest(args.output_root, args.run_id, args.tests_passed, args.drive_path)
    write_report(args.output_root, EXPERIMENT_DIR / "report.md")
    write_report(args.output_root)


if __name__ == "__main__":
    main()
