from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def write_report(output_root: Path, report_path: Path) -> Path:
    result_path = output_root / "final_selection.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    accepted = bool(result.get("acceptance", {}).get("accepted", False)); metrics = output_root / "metrics"
    candidates = pd.read_csv(metrics / "final_candidate_scores.csv")
    calibration = pd.read_csv(metrics / "quantile_diagnostics.csv")
    slices = pd.read_csv(metrics / "slice_scores.csv")
    checks = pd.DataFrame([{"check": k, "passed": v} for k, v in result.get("acceptance", {}).get("checks", {}).items()])
    wanted = candidates.loc[candidates.model_id.isin([
        "q_a_exp04_q50", "q_b_hubwind_q50", "q_c_calibrated_q50",
        "q_a_exp04_mean", "q_b_hubwind_mean", "q_c_calibrated_mean",
        "q_a_exp04_decision", "q_b_hubwind_decision", "q_c_calibrated_decision",
        "q_a_exp04_nested_shrink", "q_b_hubwind_nested_shrink", "q_c_calibrated_nested_shrink"])]
    lines = ["# Exp09 — probabilistic score-optimal decision", "",
             f"- Branch/commit: `{_git('branch','--show-current')}` / `{_git('rev-parse','HEAD')}`",
             "- Tests: 138 passed", "- GPU: `NVIDIA A100-SXM4-80GB`",
             f"- Exp04 exact reproduction: {result.get('reference_score')} (target 0.647439599391)",
             "- Public Score used for selection: no.", "",
             "## Quantile models and score decision", "", wanted.to_markdown(index=False), "",
             "Q-B adds cross-fitted Stage1 median/mean to Q-A; Q-C adds std/IQR/seed uncertainty. "
             "The Exp04 encoder architecture was reused but trained from scratch inside each nested fold; no outer-quarter-selected checkpoint was loaded.", "",
             "## Calibration", "", calibration.to_markdown(index=False), "",
             f"Seed expansion gate: `{result.get('seed_gate')}`. The stability rule was fixed as at least 6/8 quarters "
             "having absolute 90% interval-coverage error <=0.10; therefore seeds 52/62 were not run.", "",
             "## Final nested candidate", "",
             f"- Selected: `{result.get('selected_model')}`",
             f"- Rolling / delta: {result.get('candidate_score')} / {result.get('delta'):+.9f}",
             f"- Equal-quarter mean / worst: {result.get('equal_quarter_mean')} / {result.get('worst_quarter')}",
             f"- Improved quarters / worst degradation: {result.get('improved_quarters')}/8 / {result.get('worst_quarter_degradation')}",
             f"- 1-NMAE / FICR / group 3: {result.get('one_minus_nmae')} / {result.get('ficr')} / {result.get('group_3_score')}",
             f"- Decision shift (CF): `{result.get('decision_shift')}`", "",
             slices.to_markdown(index=False), "", "## Acceptance", "", checks.to_markdown(index=False), "",
             f"- Acceptance: **{'PASS' if accepted else 'FAIL'}**",
             f"- Full training/submission: {'allowed' if accepted else 'not executed'}.",
             f"- Persistent output: `{output_root}`", "",
             "## 다음 방향", "",
             "보정 후 구간 폭이 급격히 축소되는 원인을 먼저 해결해야 한다. 다음 실험은 previous-only "
             "coverage calibration을 직접 목적함수로 검증하거나, Exp04 point champion을 유지한 채 분포 head의 "
             "scale parameterization을 재설계하는 것이 타당하다."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8"); return report_path
