"""Run the complete CPU-first Exp06 audit, nested selection, and full OOF fit."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baram.constants import TARGETS, TIME_COL
from baram.data import load_sample_submission
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config
from experiments.exp05_cross_group_transfer.src.evaluate import slice_metrics

from .ficr_gate import apply_gate, build_gate_features, nested_gate_selection
from .make_submission import write_diagnostic_submission
from .nested_selection import choose_final, piecewise_acceptance, summarize_candidate
from .oof_loader import load_exp06_oof, write_contract_checks
from .oracle_analysis import oracle_headroom, regime_advantage
from .piecewise_calibration import apply_piecewise, nested_piecewise_selection
from .threshold_audit import (
    threshold_margin_samples,
    tier_distribution,
    transition_matrix,
    write_tier_check,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp06_ficr_threshold_calibration"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"


def read_config(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def directories(output: Path) -> None:
    for name in ("checks", "metrics", "predictions", "figures", "submissions"):
        (output / name).mkdir(parents=True, exist_ok=True)


def _test_long() -> pd.DataFrame:
    exp03 = pd.read_csv(
        PROJECT_ROOT / "experiments/exp03_official_score_calibration/outputs/predictions/ficr_aware_full_ensemble_test.csv",
        parse_dates=[TIME_COL],
    )
    raw = pd.read_csv(
        PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs/predictions/raw_ensemble_predictions.csv",
        parse_dates=[TIME_COL],
    )
    if not exp03[TIME_COL].equals(raw[TIME_COL]):
        raise ValueError("Exp03/raw test timestamps differ")
    parts = []
    for group_id, target in enumerate(TARGETS, 1):
        parts.append(pd.DataFrame({
            TIME_COL: exp03[TIME_COL], "target": target, "group_id": group_id,
            "capacity_kwh": 21600.0 if group_id < 3 else 21000.0,
            "exp03_prediction": exp03[target], "raw_prediction": raw[target],
        }))
    out = pd.concat(parts, ignore_index=True).sort_values([TIME_COL, "target"]).reset_index(drop=True)
    out["global_blend_prediction"] = 0.6*out["exp03_prediction"] + 0.4*out["raw_prediction"]
    forecast = pd.to_datetime(out[TIME_COL])
    out["issue_kst_dtm"] = (forecast-pd.Timedelta(hours=1)).dt.normalize()-pd.Timedelta(hours=11)
    out["lead_time_h"] = (forecast-out["issue_kst_dtm"]).dt.total_seconds()/3600
    out["hour"] = forecast.dt.hour; out["month"] = forecast.dt.month; out["dayofyear"] = forecast.dt.dayofyear
    return out


def _gate_stability(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in data.groupby(["quarter", "target", "group_id"], sort=True):
        rows.append({
            "slice": "group", "quarter": keys[0], "target": keys[1], "group_id": keys[2],
            "slice_value": keys[1], "mean_raw_weight": part["gate_raw_weight"].mean(),
            "std_raw_weight": part["gate_raw_weight"].std(), "samples": len(part),
        })
    for keys, part in data.groupby(["quarter", "lead_time_h"], sort=True):
        rows.append({
            "slice": "lead_time", "quarter": keys[0], "target": "all", "group_id": 0,
            "slice_value": keys[1], "mean_raw_weight": part["gate_raw_weight"].mean(),
            "std_raw_weight": part["gate_raw_weight"].std(), "samples": len(part),
        })
    return pd.DataFrame(rows)


def run(output: Path) -> dict:
    output = output.resolve(); directories(output)
    data = load_exp06_oof(); reproduction = write_contract_checks(data, output)
    write_tier_check(data, output / "checks/scorer_tier_check.json")
    print("reference reproduced", reproduction["reproduced"]["total_score"], flush=True)

    distribution = tier_distribution(data); distribution.to_csv(output / "metrics/tier_distribution.csv", index=False)
    transitions = transition_matrix(data); transitions.to_csv(output / "metrics/tier_transition_matrix.csv", index=False)
    margins = threshold_margin_samples(data); margins.to_csv(output / "metrics/threshold_margin_samples.csv", index=False)
    advantage = regime_advantage(data); advantage.to_csv(output / "metrics/regime_model_advantage.csv", index=False)
    oracle, oracle_groups, oracle_decision = oracle_headroom(data)
    oracle.to_csv(output / "metrics/oracle_headroom.csv", index=False)
    oracle_groups.to_csv(output / "metrics/oracle_headroom_by_group.csv", index=False)
    write_json(output / "oracle_decision.json", oracle_decision)
    print("oracle", oracle_decision, flush=True)

    piecewise_config = read_config("piecewise_affine.yaml")
    piecewise, piecewise_scores, piecewise_search, final_piecewise = nested_piecewise_selection(
        data, piecewise_config
    )
    piecewise.to_csv(output / "predictions/piecewise_nested_oof.csv", index=False)
    piecewise_scores.to_csv(output / "metrics/piecewise_nested_scores.csv", index=False)
    piecewise_search.to_csv(output / "metrics/piecewise_search.csv", index=False)
    write_json(output / "checks/final_piecewise_model.json", {
        "scheme": final_piecewise.scheme, "boundaries": final_piecewise.boundaries,
        "penalty": final_piecewise.penalty.__dict__,
        "parameters": final_piecewise.parameters.to_dict("records"),
        "fit_source": "all rolling OOF only",
    })
    piecewise_summary, piecewise_quarters, piecewise_groups = summarize_candidate(
        piecewise, "piecewise_prediction", "piecewise"
    )
    piecewise_change = (
        piecewise["piecewise_prediction"]-piecewise["global_blend_prediction"]
    ).abs()/piecewise["capacity_kwh"]
    selection_config = read_config("final_selection.yaml")
    piecewise_gate = piecewise_acceptance(piecewise_summary, selection_config, float(piecewise_change.quantile(.95)))
    print("piecewise", piecewise_summary, piecewise_gate, flush=True)

    gate_config = read_config("ficr_blend_gate.yaml")
    gate_ran = bool(oracle_decision["gate_headroom_sufficient"])
    gate = None; final_gate = None; gate_summary = None; gate_acceptance = {"accepted": False, "reason": "oracle headroom below 0.003"}
    gate_scores = pd.DataFrame(columns=["evaluation_quarter", "status", "total_score", "mean_raw_weight"])
    gate_stability = pd.DataFrame(columns=[
        "slice", "quarter", "target", "group_id", "slice_value", "mean_raw_weight", "std_raw_weight", "samples"
    ])
    if gate_ran:
        featured, gate_columns = build_gate_features(data)
        write_json(output / "checks/gate_schema.json", {
            "feature_columns": gate_columns, "oof_test_schema_equal": True,
            "target_scada_lag_features": [], "source_gate_available": False, "seed_std_available": False,
        })
        gate, gate_scores, final_gate = nested_gate_selection(featured, gate_columns, gate_config, "linear")
        gate_summary, gate_quarters, gate_groups = summarize_candidate(gate, "gate_prediction", "linear_gate")
        gate_stability = _gate_stability(gate)
        conditions = {
            "aggregate": gate_summary["total_score"] >= max(
                piecewise_summary["total_score"]+0.001, 0.6494395993905896
            ),
            "improved_quarters": gate_summary["improved_quarters"] >= 6,
            "worst": gate_summary["worst_quarter"] >= 0.6034628191969988,
            "group3": gate_summary["group3_score"] >= selection_config["minimum_group3_score"],
            "not_collapsed": gate["gate_raw_weight"].mean() > 0.05 and gate["gate_raw_weight"].mean() < 0.95,
            "stable": gate.groupby("quarter")["gate_raw_weight"].mean().std() < 0.15,
        }
        gate_acceptance = {"accepted": all(conditions.values()), "conditions": conditions}
        gate.to_csv(output / "predictions/gate_nested_oof.csv", index=False)
    else:
        write_json(output / "checks/gate_schema.json", {
            "feature_columns": [], "gate_executed": False,
            "reason": "nested deployable regime headroom below 0.003",
            "target_scada_lag_features": [],
        })
        pd.DataFrame(columns=[*data.columns, "gate_raw_weight", "gate_prediction"]).to_csv(
            output / "predictions/gate_nested_oof.csv", index=False
        )
    gate_scores.to_csv(output / "metrics/gate_nested_scores.csv", index=False)
    gate_stability.to_csv(output / "metrics/gate_weight_stability.csv", index=False)

    reference_summary, reference_quarters, reference_groups = summarize_candidate(
        data, "global_blend_prediction", "exp04_global"
    )
    candidate_rows = [
        {**reference_summary, "accepted": True, "acceptance_reason": "incumbent champion"},
        {**piecewise_summary, "accepted": piecewise_gate["accepted"],
         "acceptance_reason": json.dumps(piecewise_gate)},
    ]
    quarter_tables, group_tables = [reference_quarters, piecewise_quarters], [reference_groups, piecewise_groups]
    if gate_summary is not None:
        candidate_rows.append({**gate_summary, "accepted": gate_acceptance["accepted"],
                               "acceptance_reason": json.dumps(gate_acceptance)})
        quarter_tables.append(gate_quarters); group_tables.append(gate_groups)
    candidates = pd.DataFrame(candidate_rows)
    final_selection = choose_final(candidates)
    candidates.to_csv(output / "metrics/final_candidate_scores.csv", index=False)
    pd.concat(group_tables, ignore_index=True).to_csv(output / "metrics/group_scores.csv", index=False)
    pd.concat(quarter_tables, ignore_index=True).to_csv(output / "metrics/quarter_scores.csv", index=False)
    write_json(output / "final_selection.json", final_selection)
    write_json(output / "acceptance.json", {"piecewise": piecewise_gate, "gate": gate_acceptance})

    january_rows, high_rows = [], []
    for values, column, model in [(data, "global_blend_prediction", "exp04_global"),
                                   (piecewise, "piecewise_prediction", "piecewise")]:
        january, high = slice_metrics(values, column); january.insert(0, "model", model); high.insert(0, "model", model)
        january_rows.append(january); high_rows.append(high)
    if gate_summary is not None:
        january, high = slice_metrics(gate, "gate_prediction"); january.insert(0, "model", "linear_gate"); high.insert(0, "model", "linear_gate")
        january_rows.append(january); high_rows.append(high)
    pd.concat(january_rows, ignore_index=True).to_csv(output / "metrics/january_scores.csv", index=False)
    pd.concat(high_rows, ignore_index=True).to_csv(output / "metrics/high_wind_scores.csv", index=False)

    test = _test_long(); test_piecewise = apply_piecewise(test, final_piecewise, "piecewise_prediction")
    test_piecewise.to_csv(output / "predictions/piecewise_test_predictions.csv", index=False)
    sample = load_sample_submission(baseline_config()); stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submissions = []
    piecewise_path = output / "submissions" / f"exp06_piecewise_{stamp}.csv"
    write_diagnostic_submission(sample, test_piecewise, piecewise_path, "piecewise_prediction", piecewise_gate["accepted"])
    submissions.append(str(piecewise_path))
    if final_gate is not None:
        test_featured, test_columns = build_gate_features(test)
        if test_columns != final_gate.feature_columns:
            raise ValueError("OOF/test gate schemas differ")
        test_gate = apply_gate(final_gate, test_featured)
        test_gate.to_csv(output / "predictions/gate_test_predictions.csv", index=False)
        gate_path = output / "submissions" / f"exp06_gate_{stamp}.csv"
        write_diagnostic_submission(sample, test_gate, gate_path, "gate_prediction", gate_acceptance["accepted"])
        submissions.append(str(gate_path))
    write_json(output / "submission_manifest.json", {
        "submissions": submissions, "maximum": 2, "auto_submitted": False,
        "diagnostic_only": not final_selection["accepted_new_rule"],
    })
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
                                 check=True, capture_output=True, text=True).stdout.strip(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                 check=True, capture_output=True, text=True).stdout.strip(),
        "device": "cpu", "public_used_for_selection": False,
        "gate_executed": gate_ran,
    }
    write_json(output / "run_manifest.json", manifest)
    from .make_report import write_report
    write_report(output)
    return {"reference": reference_summary, "piecewise": piecewise_summary,
            "gate": gate_summary, "selection": final_selection, "submissions": submissions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); print(json.dumps(run(args.output_root), indent=2, default=str))


if __name__ == "__main__":
    main()
