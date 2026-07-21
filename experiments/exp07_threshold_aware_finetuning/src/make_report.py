"""Render Exp07's final report from generated evidence tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report(output_root: Path, path: Path | None = None) -> Path:
    output_root = Path(output_root); path = path or output_root / "report.md"
    reference = json.loads((output_root / "checks/reference_reproduction.json").read_text())
    selection_path = output_root / "final_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {
        "accepted": False, "reason": "neural stages incomplete"
    }
    candidates_path = output_root / "metrics/final_candidate_scores.csv"
    candidates = pd.read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()
    best = candidates.sort_values("total_score", ascending=False).iloc[0].to_dict() if len(candidates) else {}
    text = f"""# exp07 threshold-aware fine-tuning report

## Reference contract

- Exp04 reproduced Score: {reference['reproduced']['total_score']:.12f}
- absolute error: {reference['absolute_error']:.3g}
- Public score used for selection: no

## Selection

- acceptance: {selection.get('accepted', False)}
- selection detail: `{json.dumps(selection, ensure_ascii=False)}`
- best rolling candidate: `{json.dumps(best, ensure_ascii=False, default=str)}`

Full fine-tuning and submission generation are permitted only when all acceptance
conditions pass. No submission is sent automatically.
"""
    path.write_text(text, encoding="utf-8")
    return path

