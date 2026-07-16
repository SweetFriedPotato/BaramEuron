import numpy as np
import pandas as pd
import pytest

from official.dacon_baram_metric.metric import CAPACITY_KWH, TARGET_COLS
from experiments.exp03_official_score_calibration.src.backtest import (
    assert_selection_precedes_evaluation,
    expanding_quarter_window,
    issue_quarter,
)
from experiments.exp03_official_score_calibration.src.calibration import apply_affine
from experiments.exp03_official_score_calibration.src.prediction_loader import (
    EXP02_PREDICTIONS,
    GROUP_IDS,
    load_best_blend,
    validate_prediction_contract,
)
from experiments.exp03_official_score_calibration.src.make_submission import convex_weight_grid


def _long_frame():
    times = pd.date_range("2024-03-31 01:00", periods=48, freq="h")
    rows = []
    for target, capacity in CAPACITY_KWH.items():
        for timestamp in times:
            rows.append(
                {
                    "fold": "fold_b",
                    "forecast_kst_dtm": timestamp,
                    "target": target,
                    "group_id": GROUP_IDS[target],
                    "y_true_kwh": capacity * 0.5,
                    "y_pred_kwh": capacity * 0.45,
                    "model_id": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def test_issue_quarter_keeps_a_daily_issue_block_together():
    timestamps = pd.Series(pd.date_range("2024-03-31 01:00", periods=24, freq="h"))
    assert issue_quarter(timestamps).nunique() == 1
    assert issue_quarter(timestamps).iloc[0] == "2024Q1"


def test_expanding_quarter_window_preserves_issue_block_boundaries():
    window = expanding_quarter_window("2023Q2")
    assert window["train_end"] == pd.Timestamp("2023-04-01 00:00:00")
    assert window["valid_start"] == pd.Timestamp("2023-04-01 01:00:00")
    assert window["valid_end"] == pd.Timestamp("2023-07-01 00:00:00")


def test_rolling_selection_rejects_target_leakage():
    assert_selection_precedes_evaluation(["2023Q1", "2023Q2"], "2023Q3")
    with pytest.raises(ValueError):
        assert_selection_precedes_evaluation(["2023Q1", "2023Q3"], "2023Q3")


def test_affine_calibration_does_not_read_targets_for_application():
    frame = _long_frame()
    parameters = {target: (1.0, 1.0) for target in TARGET_COLS}
    first = apply_affine(frame, parameters, "calibrated")["y_pred_kwh"]
    changed_truth = frame.copy()
    changed_truth["y_true_kwh"] *= 0.2
    second = apply_affine(changed_truth, parameters, "calibrated")["y_pred_kwh"]
    assert np.array_equal(first.to_numpy(), second.to_numpy())


def test_prediction_contract_rejects_duplicate_keys():
    frame = _long_frame()
    with pytest.raises(ValueError):
        validate_prediction_contract(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_existing_best_blend_contract():
    if not (EXP02_PREDICTIONS / "best_blend_predictions.csv").exists():
        pytest.skip("exp02 ignored artifacts are not present in a fresh clone")
    frame = load_best_blend()
    assert set(frame["target"]) == set(TARGET_COLS)
    assert not frame.duplicated(["fold", "forecast_kst_dtm", "target", "group_id", "model_id"]).any()
    assert np.isfinite(frame[["y_true_kwh", "y_pred_kwh"]]).all().all()


def test_final_ensemble_weights_are_nonnegative_and_sum_to_one():
    weights = list(convex_weight_grid(0.05))
    assert len(weights) > 1
    assert all(min(item) >= 0.0 for item in weights)
    assert all(sum(item) == pytest.approx(1.0) for item in weights)
