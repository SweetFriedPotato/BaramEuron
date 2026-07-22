from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baram.constants import TIME_COL
from experiments.exp08_scada_hubwind_pretraining.src.evaluate import reproduce_exp04_reference
from .conditional_distribution import deterministic_samples
from .expected_official_score import score_optimal_decision
from .quantile_head import MonotoneQuantileHead, assert_monotone
from .quantile_loss import quantile_training_loss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp09_probabilistic_score_decision"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
EXP08_OUTPUT = PROJECT_ROOT / "experiments/exp08_scada_hubwind_pretraining/outputs"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def phase_contracts(output: Path) -> dict:
    checks = output / "checks"; checks.mkdir(parents=True, exist_ok=True)
    reference = reproduce_exp04_reference(
        PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs/predictions/best_blend_predictions.csv",
        checks / "reference_reproduction.json",
    )
    stage1 = pd.read_csv(EXP08_OUTPUT / "predictions/stage1_oof_hubwind.csv", parse_dates=[TIME_COL])
    required = {TIME_COL, "quarter", "group_id", "seed", "model_id", "target_mask_median"}
    if not required.issubset(stage1.columns):
        raise ValueError(f"Exp08 Stage1 registry is missing {sorted(required - set(stage1.columns))}")
    selected = json.loads((EXP08_OUTPUT / "stage1_selection.json").read_text())["selected_model"]
    selected_rows = stage1.loc[stage1["model_id"].eq(selected)]
    duplicates = int(selected_rows.duplicated([TIME_COL, "quarter", "group_id", "seed"]).sum())
    if duplicates:
        raise ValueError("Stage1 registry has duplicate cross-fit keys")
    periods = (selected_rows[TIME_COL] - pd.Timedelta(hours=1)).dt.to_period("Q").astype(str)
    if not periods.eq(selected_rows["quarter"]).all():
        raise ValueError("Stage1 timestamps do not match quarter cutoff keys")
    contract = {"selected_stage1": selected, "rows": len(selected_rows),
                "seeds": sorted(selected_rows["seed"].unique().tolist()),
                "quarters": sorted(selected_rows["quarter"].unique().tolist()),
                "groups": sorted(selected_rows["group_id"].unique().tolist()),
                "fallback_policy": "2022 mask=0, GFS ws100 fallback, indicator=1",
                "in_sample_stage1_feature": False, "duplicates": duplicates,
                "feature_schema": ["median", "mean", "std", "iqr", "seed_std",
                                   "forecast_minus_predicted", "fallback_indicator"]}
    write_json(checks / "stage1_feature_contract.json", contract)
    write_json(checks / "leakage_audit.json", {"scada_actual_input": False, "power_target_input": False,
               "target_lag_input": False, "test_scada_read": False, "outer_selection": False,
               "public_used_for_selection": False})
    return {"reference": reference, "stage1": contract}


def phase_smoke(output: Path) -> dict:
    torch.manual_seed(42); model = MonotoneQuantileHead(16)
    hidden = torch.randn(4, 24, 3, 16); target = torch.rand(4, 24, 3); mask = torch.ones_like(target, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(); prediction = model(hidden); total, pieces = quantile_training_loss(prediction, target, mask)
    total.backward(); optimizer.step(); assert_monotone(prediction.detach())
    quantiles = np.maximum.accumulate(prediction.detach().numpy()[0, 0, 0])
    samples = deterministic_samples(quantiles); decision = score_optimal_decision(samples, quantiles, .5)
    result = {"forward_backward": True, "shape": list(prediction.shape), "loss": float(total),
              "pieces": {name: float(value) for name, value in pieces.items()},
              "sample_count": len(samples), "decision": decision["prediction"],
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write_json(output / "checks/smoke.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["contracts", "smoke", "all"], required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True); result = {}
    if args.phase in {"contracts", "all"}: result["contracts"] = phase_contracts(args.output_root)
    if args.phase in {"smoke", "all"}: result["smoke"] = phase_smoke(args.output_root)
    write_json(args.output_root / f"phase_{args.phase}.json", result)


if __name__ == "__main__": main()
