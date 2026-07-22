"""Build the final, leakage-audited Exp09 evidence and acceptance decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp04_raw_grid_spatiotemporal.src.blend import residual_correlations
from . import QUANTILE_LEVELS
from .finalize import DEFAULT_OUTPUT, EXP04, QUARTERS
from .quantile_calibration import GroupConformalOffsets

REFERENCE_SCORE = 0.647439599391
SELECTED_MODEL = "q_c_calibrated_nested_shrink"


def _score(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    return score_available_groups(frame)


def _calibration_evidence(output: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, group_rows, pit_rows = [], [], []
    history_q, history_y, history_m = [], [], []
    levels = np.asarray(QUANTILE_LEVELS)
    for quarter in QUARTERS:
        data = np.load(output / f"predictions/q_c_calibrated/{seed}/{quarter}.npz")
        raw, y, mask = data["quantiles"], data["target"], data["mask"].astype(bool)
        calibrated = (GroupConformalOffsets().fit(
            np.concatenate(history_q), np.concatenate(history_y), np.concatenate(history_m)
        ).transform(raw) if history_q else raw)
        pinball = np.maximum(
            levels * (y[..., None] - calibrated),
            (levels - 1.0) * (y[..., None] - calibrated),
        )
        for group in range(3):
            valid = mask[..., group]
            if not valid.any():
                continue
            empirical = [float((y[..., group][valid] <= calibrated[..., group, j][valid]).mean()) for j in range(len(levels))]
            group_rows.extend({"seed": seed, "quarter": quarter, "group_id": group + 1,
                               "level": float(level), "empirical_coverage": observed,
                               "coverage_error": observed - float(level)}
                              for level, observed in zip(levels, empirical))
        valid4 = np.broadcast_to(mask[..., None], calibrated.shape)
        empirical = [float((y[mask] <= calibrated[..., j][mask]).mean()) for j in range(len(levels))]
        pit = np.mean(y[..., None] > calibrated, axis=-1)[mask]
        hist, edges = np.histogram(pit, bins=np.linspace(0, 1, 11))
        pit_rows.extend({"seed": seed, "quarter": quarter, "bin_left": edges[i],
                         "bin_right": edges[i + 1], "count": int(hist[i])} for i in range(10))
        rows.append({"seed": seed, "quarter": quarter, "history_quarters": len(history_q),
                     "pinball": float(pinball[valid4].mean()),
                     "approximate_crps": float(2.0 * pinball[valid4].mean()),
                     "interval_90_coverage": float(((y >= calibrated[..., 0]) &
                                                     (y <= calibrated[..., -1]) & mask).sum() / mask.sum()),
                     "mean_absolute_coverage_error": float(np.mean(np.abs(np.asarray(empirical) - levels))),
                     **{f"coverage_q{int(level*100):02d}": value for level, value in zip(levels, empirical)}})
        history_q.append(raw); history_y.append(y); history_m.append(mask)
    return pd.DataFrame(rows), pd.DataFrame(group_rows), pd.DataFrame(pit_rows)


def _attach_high_wind(output: Path, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    flags = []
    for quarter in QUARTERS:
        data = np.load(output / f"predictions/q_c_calibrated/{seed}/{quarter}.npz")
        times, wind, threshold = data["timestamps"], data["validation_wind"], float(data["high_wind_threshold"])
        flags.append(pd.DataFrame({"fold": quarter, TIME_COL: pd.to_datetime(times.reshape(-1)),
                                   "high_wind_mask": (wind.reshape(-1) >= threshold)}).drop_duplicates())
    result = frame.copy(); result[TIME_COL] = pd.to_datetime(result[TIME_COL])
    return result.merge(pd.concat(flags), on=["fold", TIME_COL], how="left", validate="many_to_one")


def _slice_row(model: str, kind: str, frame: pd.DataFrame) -> dict:
    try:
        summary, _ = _score(frame)
        return {"model_id": model, "slice": kind, **summary}
    except ValueError:
        return {"model_id": model, "slice": kind, "total_score": np.nan}


def audit(output: Path, seed: int = 42) -> dict:
    metrics = output / "metrics"; metrics.mkdir(parents=True, exist_ok=True)
    all_predictions = pd.read_csv(output / f"predictions/decision_predictions_seed{seed}.csv", parse_dates=[TIME_COL])
    candidate = all_predictions.loc[all_predictions["model_id"].eq(SELECTED_MODEL)].copy()
    reference = pd.read_csv(EXP04, parse_dates=[TIME_COL]); reference["quarter"] = reference["fold"]
    candidate = _attach_high_wind(output, candidate, seed)
    reference = reference.merge(candidate[["fold", TIME_COL, "target", "group_id", "high_wind_mask"]],
                                on=["fold", TIME_COL, "target", "group_id"], how="left", validate="one_to_one")
    csum, cgroups = _score(candidate); rsum, rgroups = _score(reference)
    quarters = []
    for model, frame in ((SELECTED_MODEL, candidate), ("exp04", reference)):
        for quarter, part in frame.groupby("fold", sort=True):
            summary, _ = _score(part); quarters.append({"model_id": model, "quarter": quarter, **summary})
    quarter_df = pd.DataFrame(quarters)
    comparison = quarter_df.pivot(index="quarter", columns="model_id", values="total_score").reset_index()
    comparison["delta"] = comparison[SELECTED_MODEL] - comparison["exp04"]
    improved = int((comparison["delta"] >= 0).sum())
    worst_degradation = float((-comparison["delta"]).max())
    group_df = pd.concat([cgroups.assign(model_id=SELECTED_MODEL), rgroups.assign(model_id="exp04")], ignore_index=True)
    slices = []
    for model, frame in ((SELECTED_MODEL, candidate), ("exp04", reference)):
        slices.append(_slice_row(model, "january", frame.loc[frame[TIME_COL].dt.month.eq(1)]))
        slices.append(_slice_row(model, "high_wind", frame.loc[frame["high_wind_mask"].fillna(False)]))
    residual = residual_correlations(reference, candidate)
    residual.insert(0, "model_id", SELECTED_MODEL)
    aligned = candidate.merge(reference[["fold", TIME_COL, "target", "group_id", "y_pred_kwh"]],
                              on=["fold", TIME_COL, "target", "group_id"], suffixes=("", "_reference"), validate="one_to_one")
    shift = np.abs(aligned["y_pred_kwh"] - aligned["y_pred_kwh_reference"]) / aligned["target"].map(CAPACITY_KWH)
    shift_summary = {"mean_cf": float(shift.mean()), "p50_cf": float(shift.quantile(.5)),
                     "p95_cf": float(shift.quantile(.95)), "maximum_cf": float(shift.max())}
    calibration, group_calibration, pit = _calibration_evidence(output, seed)
    # Stability is pre-declared as >=6/8 quarters within 10 percentage points of nominal 90% coverage.
    stable_quarters = int((calibration["interval_90_coverage"].sub(.90).abs() <= .10).sum())
    q_b = pd.read_csv(metrics / f"candidate_scores_seed{seed}.csv").set_index("model_id").loc["q_b_hubwind_nested_shrink", "total_score"]
    q_c_improves_q_b = csum["total_score"] > float(q_b)
    calibration_stable = stable_quarters >= 6
    group3_c = float(cgroups.loc[cgroups["group_id"].eq(3), "score"].iloc[0])
    group3_r = float(rgroups.loc[rgroups["group_id"].eq(3), "score"].iloc[0])
    checks = {
        "rolling_at_least_0_649440": csum["total_score"] >= .649440,
        "improvement_at_least_0_002": csum["total_score"] - rsum["total_score"] >= .002,
        "improved_quarters_at_least_6": improved >= 6,
        "worst_quarter_degradation_at_most_0_002": worst_degradation <= .002,
        "ficr_maintained": csum["ficr"] >= rsum["ficr"],
        "one_minus_nmae_within_0_0005": csum["one_minus_nmae"] >= rsum["one_minus_nmae"] - .0005,
        "group_3_maintained": group3_c >= group3_r,
        "three_seed_mean_improves": False,
        "decision_shift_p95_at_most_0_03": shift_summary["p95_cf"] <= .03,
        "not_single_seed_dependent": False,
    }
    result = {"selected_model": SELECTED_MODEL, "reference_score": rsum["total_score"],
              "candidate_score": csum["total_score"], "delta": csum["total_score"] - rsum["total_score"],
              "equal_quarter_mean": float(quarter_df.loc[quarter_df.model_id.eq(SELECTED_MODEL), "total_score"].mean()),
              "worst_quarter": float(quarter_df.loc[quarter_df.model_id.eq(SELECTED_MODEL), "total_score"].min()),
              "improved_quarters": improved, "worst_quarter_degradation": worst_degradation,
              "one_minus_nmae": csum["one_minus_nmae"], "ficr": csum["ficr"],
              "group_3_score": group3_c, "decision_shift": shift_summary,
              "seed_gate": {"q_c_improves_q_b": bool(q_c_improves_q_b),
                            "q_c_minus_q_b": csum["total_score"] - float(q_b),
                            "calibration_stable": calibration_stable,
                            "stable_quarters": stable_quarters, "required_stable_quarters": 6,
                            "seeds_52_62_executed": False},
              "acceptance": {"accepted": bool(all(checks.values())), "checks": checks},
              "submission_executed": False, "public_used_for_selection": False}
    quarter_df.to_csv(metrics / "nested_quarter_scores.csv", index=False)
    comparison.to_csv(metrics / "quarter_comparison.csv", index=False)
    group_df.to_csv(metrics / "group_scores.csv", index=False)
    pd.DataFrame(slices).to_csv(metrics / "slice_scores.csv", index=False)
    residual.to_csv(metrics / "residual_correlations.csv", index=False)
    pd.DataFrame([shift_summary]).to_csv(metrics / "decision_shift.csv", index=False)
    calibration.to_csv(metrics / "quantile_diagnostics.csv", index=False)
    group_calibration.to_csv(metrics / "quantile_group_calibration.csv", index=False)
    pit.to_csv(metrics / "pit_histogram.csv", index=False)
    pd.read_csv(metrics / f"candidate_scores_seed{seed}.csv").to_csv(metrics / "final_candidate_scores.csv", index=False)
    (output / "final_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); print(json.dumps(audit(args.output_root, args.seed), indent=2))


if __name__ == "__main__":
    main()
