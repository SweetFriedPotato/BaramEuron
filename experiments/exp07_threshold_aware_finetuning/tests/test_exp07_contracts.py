from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from baram.constants import TARGETS, TIME_COL
from experiments.exp02_daily_tcn_scada_aux.src.models import DailyTCN
from experiments.exp03_official_score_calibration.src.official_scorer import score_wide
from experiments.exp04_raw_grid_spatiotemporal.src.models import RawGridSpatiotemporalModel
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import RawGridDataset
from experiments.exp05_cross_group_transfer.src.oof_contract import (
    EXPECTED_GLOBAL_SCORE,
    load_oof_contract,
    score_prediction,
)
from experiments.exp07_threshold_aware_finetuning.src.blend import blend_prediction
from experiments.exp07_threshold_aware_finetuning.src.evaluate import rescue_gain, threshold_transitions
from experiments.exp07_threshold_aware_finetuning.src.freeze_policy import apply_freeze_policy
from experiments.exp07_threshold_aware_finetuning.src.make_submission import validate_submission
from experiments.exp07_threshold_aware_finetuning.src.nested_finetune import (
    assert_no_outer_leakage,
    nested_windows,
    select_inner_checkpoint,
)
from experiments.exp07_threshold_aware_finetuning.src.threshold_loss import (
    boundary_weight,
    group_balanced_mean,
    normalized_soft_reward,
)
from experiments.exp07_threshold_aware_finetuning.src.trainer import TCNFineTuneDataset


def test_exp04_exact_reproduction():
    data = load_oof_contract()
    score = score_prediction(data, "global_blend_prediction")
    assert abs(score["total_score"] - EXPECTED_GLOBAL_SCORE) < 1e-8


def test_official_thresholds_are_inclusive_at_six_and_eight_percent():
    actual = pd.DataFrame({target: [10000.0, 10000.0, 10000.0] for target in TARGETS})
    prediction = actual.copy()
    capacities = [21600.0, 21600.0, 21000.0]
    for target, capacity in zip(TARGETS, capacities):
        prediction[target] = [10000.0 + .06 * capacity, 10000.0 + .08 * capacity, 10000.0 + .080001 * capacity]
    score = score_wide(actual, prediction)
    for group in score.groups:
        # Energy weights are equal here: rewards 4, 3, 0.
        assert group.ficr == pytest.approx(7 / 12)


def test_soft_reward_is_monotone_decreasing_in_error():
    error = torch.linspace(0.0, 0.15, 1000)
    reward = normalized_soft_reward(error, 0.006)
    assert torch.all(reward[:-1] >= reward[1:])


def test_boundary_weight_is_symmetric_around_joint_center():
    errors = torch.tensor([0.05, 0.055, 0.085, 0.09])
    weights = boundary_weight(errors, 0.006)
    torch.testing.assert_close(weights[0], weights[3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights[1], weights[2], atol=1e-6, rtol=1e-6)


def test_boundary_routing_weight_is_detached():
    error = torch.tensor([0.06, 0.08], requires_grad=True)
    assert not boundary_weight(error, 0.006).requires_grad


def test_group_balanced_loss_does_not_sample_weight_groups():
    values = torch.tensor([[1.0, 10.0], [3.0, 0.0], [100.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    # group means are (1+3)/2=2 and 10, so balanced mean is 6.
    assert group_balanced_mean(values, mask).item() == pytest.approx(6.0)


def test_head_only_freeze_policy_keeps_auxiliary_and_encoder_frozen():
    tcn = DailyTCN(5, projection_dim=8, hidden_channels=8, dilations=(1, 2, 4, 8), auxiliary_heads=True)
    manifest = apply_freeze_policy(tcn, "head_only")
    assert manifest.trainable_names
    assert all(name.startswith("power_heads") for name in manifest.trainable_names)
    assert all(not name.startswith("aux_heads") for name in manifest.trainable_names)


def test_outer_quarter_has_no_training_or_inner_overlap():
    windows = nested_windows()
    assert windows[0].fallback
    assert all(not window.fallback for window in windows[1:])
    train = np.array(["2023-01-01T00"], dtype="datetime64[h]")
    inner = np.array(["2023-04-01T00"], dtype="datetime64[h]")
    outer = np.array(["2023-07-01T00"], dtype="datetime64[h]")
    assert_no_outer_leakage(train, inner, outer)
    with pytest.raises(ValueError):
        assert_no_outer_leakage(outer, inner, outer)


def test_inner_checkpoint_selection_uses_documented_tie_breaks():
    rows = [
        {"epoch": 1, "total_score": .6500, "ficr": .40, "one_minus_nmae": .90, "parameter_distance": .001},
        {"epoch": 2, "total_score": .6504, "ficr": .41, "one_minus_nmae": .89, "parameter_distance": .003},
        {"epoch": 3, "total_score": .6490, "ficr": .99, "one_minus_nmae": .99, "parameter_distance": 0.0},
    ]
    assert select_inner_checkpoint(rows)["epoch"] == 2


def test_training_inputs_exclude_scada_and_targets():
    assert TCNFineTuneDataset.input_names == ("forecast_features",)
    assert "scada" not in " ".join(RawGridDataset.input_names).lower()
    assert "target" not in " ".join(RawGridDataset.input_names).lower()


def _transition_fixture():
    errors_base = np.array([.09, .09, .07, .05, .05, .07, .09, .07, .05])
    errors_new = np.array([.07, .05, .05, .07, .09, .09, .09, .07, .05])
    return pd.DataFrame({
        "quarter": ["2024Q1"] * 9,
        TIME_COL: pd.date_range("2024-01-01", periods=9, freq="h"),
        "target": [TARGETS[0]] * 9,
        "group_id": [1] * 9,
        "capacity_kwh": [100.0] * 9,
        "y_true_kwh": [20.0] * 9,
        "official_mask": [True] * 9,
        "base": 20.0 + 100.0 * errors_base,
        "candidate": 20.0 + 100.0 * errors_new,
    })


def test_transition_matrix_contains_all_nine_cells():
    transitions = threshold_transitions(_transition_fixture(), "base", "candidate")
    assert len(transitions) == 9
    assert transitions["count"].sum() == 9


def test_rescue_gain_is_rescues_minus_losses():
    transitions = threshold_transitions(_transition_fixture(), "base", "candidate")
    # Three rescues (0->3, 0->4, 3->4) and three losses (4->3, 4->0, 3->0).
    assert rescue_gain(transitions) == 0


def test_blend_calculation_uses_one_global_raw_weight():
    exp03 = np.array([1.0, 3.0]); raw = np.array([5.0, 7.0])
    np.testing.assert_allclose(blend_prediction(exp03, raw, .25), [2.0, 4.0])


def test_submission_contract_checks_rows_keys_values_and_dtypes():
    rows = 8760
    sample = pd.DataFrame({
        "forecast_id": [f"forecast_{index:04d}" for index in range(rows)],
        TIME_COL: pd.date_range("2025-01-01 01:00", periods=rows, freq="h").astype(str),
        **{target: np.zeros(rows) for target in TARGETS},
    })
    output = sample.copy()
    for target in TARGETS:
        output[target] = np.arange(rows, dtype=float)
    validate_submission(output, sample)
    broken = output.copy(); broken.loc[0, TARGETS[0]] = np.nan
    with pytest.raises(ValueError):
        validate_submission(broken, sample)
