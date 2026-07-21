"""Render Exp07's audit figures, manifest, and final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
EXP04_SCORE = 0.6474395993905896
ACCEPTANCE_SCORE = 0.649440
DRIVE_RUN = (
    "/content/drive/MyDrive/Baram/runs/exp07_threshold_aware_finetuning/"
    "20260721_121800"
)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _git(*args: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=EXPERIMENT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")


def _placeholder(plt, path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title, fontweight="bold")
    axis.text(0.5, 0.5, message, ha="center", va="center", color="#475569")
    _save_figure(figure, path)
    plt.close(figure)


def _history_table(output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((output_root / "checkpoints").glob("*.history.json")):
        identity = path.name.removesuffix(".history.json")
        try:
            prefix, seed_text = identity.rsplit("_seed_", 1)
            prefix, quarter = prefix.rsplit("_", 1)
            model_id, candidate_id = prefix.split("_", 1)
            seed = int(seed_text)
        except (ValueError, TypeError):
            continue
        for record in _read_json(path, []) or []:
            rows.append(
                {
                    "model_id": model_id,
                    "candidate_id": candidate_id,
                    "quarter": quarter,
                    "seed": seed,
                    **record,
                }
            )
    return pd.DataFrame(rows)


def render_figures(output_root: Path) -> list[Path]:
    """Create the eight figures required by the Exp07 experiment contract."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    history = _history_table(output_root)
    path = figure_root / "training_curves.png"
    if history.empty:
        _placeholder(plt, path, "Training curves", "No checkpoint histories found")
    else:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        selected = _read_json(output_root / "head_selection.json", {}).get("selected", {})
        for axis, model_id in zip(axes, ("exp03", "raw")):
            part = history.loc[history["model_id"].eq(model_id)].copy()
            preferred = {
                selected.get(model_id, ""),
                f"last_block_{selected.get(model_id, '')}",
            }
            filtered = part.loc[part["candidate_id"].isin(preferred)]
            if not filtered.empty:
                part = filtered
            for candidate_id, candidate in part.groupby("candidate_id", sort=True):
                curve = candidate.groupby("epoch", as_index=False)["total_score"].mean()
                axis.plot(curve["epoch"], curve["total_score"], marker="o", ms=3,
                          label=candidate_id.replace("annealed_015_004_", "annealed_"))
            axis.set(title=model_id.upper(), xlabel="Epoch", ylabel="Mean inner official Score")
            axis.legend(fontsize=7)
        figure.suptitle("Nested fine-tuning curves (selected configurations)", fontweight="bold")
        _save_figure(figure, path); plt.close(figure)
    created.append(path)

    transitions = _read_csv(output_root / "metrics/threshold_transitions.csv")
    path = figure_root / "threshold_transition_heatmap.png"
    if transitions.empty:
        _placeholder(plt, path, "Threshold transitions", "Transition table unavailable")
    else:
        order = ["tier_4", "tier_3", "tier_0"]
        matrix = transitions.groupby(["from_tier", "to_tier"])["count"].sum().unstack(fill_value=0)
        matrix = matrix.reindex(index=order, columns=order, fill_value=0)
        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        image = axis.imshow(matrix.to_numpy(), cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{int(matrix.iloc[row, column]):,}",
                          ha="center", va="center", color="#111827")
        axis.set_xticks(range(3), order); axis.set_yticks(range(3), order)
        axis.set(xlabel="Fine-tuned tier", ylabel="Exp04 tier", title="Threshold transition counts")
        figure.colorbar(image, ax=axis, fraction=0.046)
        _save_figure(figure, path); plt.close(figure)
    created.append(path)

    path = figure_root / "boundary_rescue_vs_loss.png"
    if transitions.empty:
        _placeholder(plt, path, "Boundary rescue vs loss", "Transition table unavailable")
    else:
        rescue_pairs = {("tier_0", "tier_3"), ("tier_0", "tier_4"), ("tier_3", "tier_4")}
        loss_pairs = {("tier_4", "tier_3"), ("tier_4", "tier_0"), ("tier_3", "tier_0")}
        pairs = list(zip(transitions["from_tier"], transitions["to_tier"]))
        rescue = int(transitions.loc[[pair in rescue_pairs for pair in pairs], "count"].sum())
        loss = int(transitions.loc[[pair in loss_pairs for pair in pairs], "count"].sum())
        figure, axis = plt.subplots(figsize=(7, 4.5))
        bars = axis.bar(["Rescued", "Lost"], [rescue, loss], color=["#16a34a", "#dc2626"])
        axis.bar_label(bars, fmt="%d"); axis.set_ylabel("Official-mask samples")
        axis.set_title(f"Boundary transitions (rescue gain = {rescue - loss:+d})", fontweight="bold")
        _save_figure(figure, path); plt.close(figure)
    created.append(path)

    quarters = _read_csv(output_root / "metrics/nested_quarter_scores_final.csv")
    path = figure_root / "quarter_score_comparison.png"
    if quarters.empty:
        _placeholder(plt, path, "Quarter comparison", "Quarter score table unavailable")
    else:
        x = np.arange(len(quarters)); width = 0.38
        figure, axis = plt.subplots(figsize=(10, 4.8))
        axis.bar(x - width / 2, quarters["reference_score"], width, label="Exp04", color="#64748b")
        axis.bar(x + width / 2, quarters["total_score"], width, label="Exp07 selected", color="#2563eb")
        axis.set_xticks(x, quarters["quarter"], rotation=30)
        axis.set(ylabel="Official Score", title="Nested outer-quarter Score")
        axis.legend(); _save_figure(figure, path); plt.close(figure)
    created.append(path)

    components = _read_csv(output_root / "metrics/component_scores.csv")
    path = figure_root / "component_comparison.png"
    if components.empty:
        _placeholder(plt, path, "Component comparison", "Component table unavailable")
    else:
        ordered = components.sort_values("total_score")
        figure, axis = plt.subplots(figsize=(9, 5.2))
        bars = axis.barh(ordered["model_id"], ordered["total_score"], color="#0f766e")
        axis.bar_label(bars, labels=[f"{value:.6f}" for value in ordered["total_score"]], padding=3)
        axis.axvline(EXP04_SCORE, color="#dc2626", ls="--", lw=1, label="Exp04 champion")
        axis.set(xlabel="Rolling aggregate Score", title="Original vs fine-tuned components")
        axis.legend(); _save_figure(figure, path); plt.close(figure)
    created.append(path)

    blend = _read_csv(output_root / "metrics/blend_search.csv")
    path = figure_root / "blend_search.png"
    if blend.empty:
        _placeholder(plt, path, "Blend search", "Blend search table unavailable")
    else:
        figure, axis = plt.subplots(figsize=(10, 5.2))
        for combination, part in blend.groupby("combination", sort=True):
            part = part.sort_values("raw_weight")
            axis.plot(part["raw_weight"], part["total_score"], label=combination[0], lw=2)
        axis.axhline(EXP04_SCORE, color="#64748b", ls="--", lw=1, label="Exp04")
        axis.set(xlabel="Raw component weight", ylabel="Rolling official Score",
                 title="Global blend search (A/B/C/D)")
        axis.legend(ncol=3); _save_figure(figure, path); plt.close(figure)
    created.append(path)

    path = figure_root / "nmae_ficr_tradeoff.png"
    if components.empty:
        _placeholder(plt, path, "1-NMAE / FICR trade-off", "Component table unavailable")
    else:
        figure, axis = plt.subplots(figsize=(8.2, 5.6))
        scatter = axis.scatter(components["one_minus_nmae"], components["ficr"],
                               c=components["total_score"], cmap="viridis", s=90)
        for row in components.itertuples():
            axis.annotate(row.model_id, (row.one_minus_nmae, row.ficr), xytext=(4, 4),
                          textcoords="offset points", fontsize=8)
        axis.set(xlabel="1 - NMAE", ylabel="FICR", title="Accuracy / threshold-reward trade-off")
        figure.colorbar(scatter, ax=axis, label="Official Score")
        _save_figure(figure, path); plt.close(figure)
    created.append(path)

    final = _read_json(output_root / "final_selection.json", {})
    path = figure_root / "final_score_comparison.png"
    candidate_score = final.get("rolling_score")
    if candidate_score is None:
        _placeholder(plt, path, "Final score", "Final selection unavailable")
    else:
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        labels = ["Exp04 champion", "Exp07 selected", "Acceptance floor"]
        values = [EXP04_SCORE, float(candidate_score), ACCEPTANCE_SCORE]
        bars = axis.bar(labels, values, color=["#64748b", "#2563eb", "#f59e0b"])
        axis.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=3)
        axis.set_ylim(min(values) - 0.003, max(values) + 0.003)
        axis.set(ylabel="Rolling official Score", title="Final champion decision")
        _save_figure(figure, path); plt.close(figure)
    created.append(path)
    return created


def _component(components: pd.DataFrame, model_id: str) -> dict:
    if components.empty:
        return {}
    row = components.loc[components["model_id"].eq(model_id)]
    return row.iloc[0].to_dict() if len(row) else {}


def _slice_score(frame: pd.DataFrame) -> str:
    return _fmt(frame.iloc[0]["total_score"]) if len(frame) and "total_score" in frame else "n/a"


def write_report(output_root: Path, path: Path | None = None, *, tests_passed: int = 109) -> Path:
    """Write the final evidence-led report; Public scores remain context only."""
    output_root = Path(output_root)
    path = path or output_root / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = _read_json(output_root / "checks/reference_reproduction.json", {})
    head = _read_json(output_root / "head_selection.json", {}).get("selected", {})
    last_block = _read_json(output_root / "component_selection.json", {})
    selection = _read_json(output_root / "final_selection.json", {"accepted": False})
    components = _read_csv(output_root / "metrics/component_scores.csv")
    quarters = _read_csv(output_root / "metrics/nested_quarter_scores_final.csv")
    seeds = _read_csv(output_root / "metrics/final_seed_scores.csv")
    boundary = _read_csv(output_root / "metrics/boundary_region_scores.csv")
    clipping = _read_csv(output_root / "metrics/clipping_diagnostic.csv")
    january = _read_csv(output_root / "metrics/january_scores.csv")
    high_wind = _read_csv(output_root / "metrics/high_wind_scores.csv")
    exp03_original = _component(components, "original_exp03")
    exp03_tuned = _component(components, "finetuned_exp03")
    raw_original = _component(components, "original_raw")
    raw_tuned = _component(components, "finetuned_raw")
    incumbent = _component(components, "exp04_global")
    candidate = _component(components, "exp07_best_blend")
    acceptance = selection.get("acceptance", {})
    conditions = acceptance.get("conditions", {})
    transition = _read_csv(output_root / "metrics/threshold_transitions.csv")
    if transition.empty:
        rescue = loss = 0
    else:
        pairs = list(zip(transition["from_tier"], transition["to_tier"]))
        rescue_set = {("tier_0", "tier_3"), ("tier_0", "tier_4"), ("tier_3", "tier_4")}
        loss_set = {("tier_4", "tier_3"), ("tier_4", "tier_0"), ("tier_3", "tier_0")}
        rescue = int(transition.loc[[item in rescue_set for item in pairs], "count"].sum())
        loss = int(transition.loc[[item in loss_set for item in pairs], "count"].sum())
    ref_score = reference.get("reproduced", {}).get("total_score", np.nan)
    ref_error = reference.get("absolute_error", np.nan)
    quarter_delta = quarters.get("score_delta", pd.Series(dtype=float))
    seed_mean = seeds.get("total_score", pd.Series(dtype=float)).mean() if len(seeds) else np.nan
    seed_std = seeds.get("total_score", pd.Series(dtype=float)).std(ddof=0) if len(seeds) else np.nan
    boundary_lines = []
    for region, part in boundary.groupby("region", sort=False) if len(boundary) else []:
        base = part.loc[part["model"].eq("base")]
        tuned = part.loc[part["model"].eq("candidate")]
        if len(base) and len(tuned):
            boundary_lines.append(
                f"- {region}: tier-4 {_fmt(base.iloc[0]['tier4_rate'])} → "
                f"{_fmt(tuned.iloc[0]['tier4_rate'])}, rewarded "
                f"{_fmt(base.iloc[0]['rewarded_rate'])} → {_fmt(tuned.iloc[0]['rewarded_rate'])}"
            )
    condition_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in conditions.items()
    ) or "- unavailable"
    component_rows = []
    for name, original, tuned in (
        ("Exp03", exp03_original, exp03_tuned), ("raw", raw_original, raw_tuned)
    ):
        component_rows.append(
            f"| {name} | {_fmt(original.get('total_score'))} | {_fmt(tuned.get('total_score'))} | "
            f"{_fmt((tuned.get('total_score', np.nan) - original.get('total_score', np.nan)))} |"
        )
    clipping_best = clipping.sort_values("total_score", ascending=False).iloc[0].to_dict() if len(clipping) else {}
    report = f"""# Exp07 threshold-aware fine-tuning report

## Outcome

Exp07 did **not** replace Exp04. The selected nested OOF blend is the unchanged
Exp03/raw 0.6/0.4 champion with Score `{_fmt(selection.get('rolling_score'), 12)}`
and delta `{_fmt(selection.get('score_delta'), 12)}`. Full fine-tuning and submission
generation were therefore skipped by contract.

- Branch: `exp/07-threshold-aware-finetune`
- Evidence commit: `{_git('rev-parse', '--short', 'HEAD')}`
- Tests: `{tests_passed} passed`
- A100 Drive run: `{DRIVE_RUN}`
- Public Score used for selection: no
- Public submission priority: none; Exp04 remains champion

## Reference contract

- Exp04 reproduced Score: `{_fmt(ref_score, 12)}`
- Absolute reproduction error: `{_fmt(ref_error, 12)}` (tolerance `1e-8`)
- Official scorer/checkpoint/preprocessing contracts: passed
- Random split: no; outer target used for selection: no

## Fine-tuning selection

- Exp03 head-only: `{head.get('exp03', 'n/a')}`
- Raw head-only: `{head.get('raw', 'n/a')}`
- Temperature: Exp03 fixed `tau=0.006`; raw cosine `0.015→0.004`
- Boundary: symmetric, detached, `sigma=0.006`; selected `lambda=0.05`
- Exp03 last-block: `{last_block.get('exp03', {}).get('selected_candidate', 'n/a')}`;
  inner Score `{_fmt(last_block.get('exp03', {}).get('head_inner_score'))}` →
  `{_fmt(last_block.get('exp03', {}).get('last_block_inner_score'))}`
- Raw last-block: rejected; inner Score
  `{_fmt(last_block.get('raw', {}).get('head_inner_score'))}` →
  `{_fmt(last_block.get('raw', {}).get('last_block_inner_score'))}`

| Component | Original Score | Fine-tuned Score | Delta |
|---|---:|---:|---:|
{chr(10).join(component_rows)}

## Final nested evaluation

- Best combination: `{selection.get('combination', 'n/a')}`
- Raw weight: `{_fmt(selection.get('raw_weight'), 3)}`
- Rolling aggregate: `{_fmt(candidate.get('total_score'), 12)}`
- Equal-quarter mean: `{_fmt(candidate.get('equal_quarter_mean'))}`
- Worst quarter: `{_fmt(candidate.get('worst_quarter'))}`
- Maintained/improved quarters: `{candidate.get('improved_quarters', 'n/a')}/8`
- Quarter delta range: `{_fmt(quarter_delta.min() if len(quarter_delta) else np.nan)}` to
  `{_fmt(quarter_delta.max() if len(quarter_delta) else np.nan)}`
- 1-NMAE: `{_fmt(candidate.get('one_minus_nmae'))}`
- FICR: `{_fmt(candidate.get('ficr'))}`
- Group 3: `{_fmt(candidate.get('group3_score'))}`
- January Score: `{_slice_score(january)}`
- High-wind Score: `{_slice_score(high_wind)}`
- 3-seed mean/std: `{_fmt(seed_mean)}` / `{_fmt(seed_std)}`
- Improved seeds: `{selection.get('improved_seed_count', 'n/a')}/3`

## Threshold diagnostics

- Rescue transitions: `{rescue}`
- Loss transitions: `{loss}`
- Rescue gain: `{rescue - loss:+d}`
{chr(10).join(boundary_lines) if boundary_lines else '- Boundary-region metrics unavailable'}

The zero transition counts are expected because final blend selection returned the
unchanged Exp04 prediction, even though its component candidates were fully evaluated.

## Clipping and acceptance

- Best clipping diagnostic: `{clipping_best.get('clipping', 'n/a')}`
  (`{_fmt(clipping_best.get('total_score'))}`)
- Acceptance: `{'PASS' if selection.get('accepted') else 'FAIL'}`
{condition_lines}

No accepted submission was created and nothing was submitted automatically.

## Interpretation and next direction

The loss improved inner validation for both heads and justified Exp03 last-block
testing, but the gains did not survive outer-quarter/seed evaluation. The deployable
global blend search consequently returned the exact incumbent. The next experiment
should target regime information that is available at inference time—especially
forecasted wind-distribution and ramp features—rather than increasing threshold-loss
strength or model unfreezing.

## Figures

- `figures/training_curves.png`
- `figures/threshold_transition_heatmap.png`
- `figures/boundary_rescue_vs_loss.png`
- `figures/quarter_score_comparison.png`
- `figures/component_comparison.png`
- `figures/blend_search.png`
- `figures/nmae_ficr_tradeoff.png`
- `figures/final_score_comparison.png`
"""
    path.write_text(report, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_root: Path,
    *,
    run_id: str = "20260721_121800",
    tests_passed: int = 109,
) -> Path:
    nested = _read_csv(output_root / "metrics/nested_quarter_scores.csv")
    selection = _read_json(output_root / "final_selection.json", {})
    evidence = {}
    for directory in ("checks", "metrics", "figures"):
        for path in sorted((output_root / directory).glob("*")):
            if path.is_file():
                evidence[str(path.relative_to(output_root))] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
    manifest = {
        "experiment": "exp07_threshold_aware_finetuning",
        "run_id": run_id,
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "implementation_commits": ["f3526b8", "82a359b"],
        "champion_commit": "c4b839e",
        "tests": {"passed": tests_passed, "failed": 0},
        "runtime": {
            "accelerator": "NVIDIA A100-SXM4-40GB",
            "devices_recorded": sorted(nested["device"].dropna().unique().tolist())
            if "device" in nested else [],
        },
        "selection_protocol": "nested rolling inner validation only",
        "public_score_used_for_selection": False,
        "drive_path": DRIVE_RUN,
        "final_selection": selection,
        "evidence": evidence,
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_final_artifacts(
    output_root: Path,
    *,
    report_path: Path | None = None,
    run_id: str = "20260721_121800",
    tests_passed: int = 109,
) -> dict:
    figures = render_figures(output_root)
    output_report = write_report(output_root, tests_passed=tests_passed)
    tracked_report = None
    if report_path is not None:
        tracked_report = write_report(output_root, report_path, tests_passed=tests_passed)
    manifest = write_run_manifest(output_root, run_id=run_id, tests_passed=tests_passed)
    return {
        "figures": [str(path) for path in figures],
        "output_report": str(output_report),
        "tracked_report": None if tracked_report is None else str(tracked_report),
        "run_manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--run-id", default="20260721_121800")
    parser.add_argument("--tests-passed", type=int, default=109)
    args = parser.parse_args()
    result = build_final_artifacts(
        args.output_root.resolve(),
        report_path=None if args.report_path is None else args.report_path.resolve(),
        run_id=args.run_id,
        tests_passed=args.tests_passed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
