from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from baram.constants import TARGETS, TIME_COL
from experiments.exp08_scada_hubwind_pretraining.src.evaluate import reproduce_exp04_reference
from experiments.exp08_scada_hubwind_pretraining.src.stage2_dataset import build_stage2_hub_features
from experiments.exp09_probabilistic_score_decision.src.conditional_distribution import deterministic_samples, interpolate_quantile_function
from experiments.exp09_probabilistic_score_decision.src.dataset import HUB_FEATURES, assert_input_contract
from experiments.exp09_probabilistic_score_decision.src.expected_official_score import decision_candidates, expected_components
from experiments.exp09_probabilistic_score_decision.src.make_submission import validate_submission
from experiments.exp09_probabilistic_score_decision.src.nested_trainer import assert_inner_precedes_outer
from experiments.exp09_probabilistic_score_decision.src.quantile_head import MonotoneQuantileHead, assert_monotone
from experiments.exp09_probabilistic_score_decision.src.quantile_loss import group_balanced_pinball

ROOT = Path(__file__).resolve().parents[3]


def test_01_exp04_exact_reproduction():
    path = ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs/predictions/best_blend_predictions.csv"
    assert reproduce_exp04_reference(path)["absolute_difference"] < 1e-8


def test_02_quantile_output_is_monotone_and_has_contract_shape():
    output = MonotoneQuantileHead(7)(torch.randn(2, 24, 3, 7))
    assert output.shape == (2, 24, 3, 11); assert_monotone(output)


def test_03_pinball_loss_matches_scalar_definition():
    prediction = torch.zeros(1, 1, 3, 11); target = torch.ones(1, 1, 3); mask = torch.ones_like(target, dtype=torch.bool)
    assert group_balanced_pinball(prediction, target, mask).item() == pytest.approx(np.mean([.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]))


def test_04_pinball_is_group_balanced_and_respects_mask():
    prediction = torch.zeros(1, 2, 3, 11); target = torch.zeros(1, 2, 3); mask = torch.zeros_like(target, dtype=torch.bool)
    target[..., 0] = 1; mask[..., 0] = True; target[:, 0, 1] = 3; mask[:, 0, 1] = True
    value = group_balanced_pinball(prediction, target, mask)
    assert value.item() == pytest.approx((0.5 + 1.5) / 2)


def test_05_input_contract_excludes_scada_actual():
    assert_input_contract()
    with pytest.raises(ValueError): assert_input_contract(["scada_actual_ws"])


def test_06_input_contract_excludes_target_and_target_lag():
    with pytest.raises(ValueError): assert_input_contract(["power_target"])
    with pytest.raises(ValueError): assert_input_contract(["target_lag_1"])


def test_07_exp08_crossfitted_feature_schema_and_fallback():
    prediction = np.ones((2,24,3,4), np.float32); forecast = np.ones((2,24), np.float32)
    fallback = np.zeros((2,24,3), np.float32); fallback[0] = 1
    features = build_stage2_hub_features(prediction, forecast, fallback_indicator=fallback)
    assert features.shape == (2,24,3,8)
    assert np.array_equal(features[..., -1], fallback)
    assert set(HUB_FEATURES).isdisjoint({"scada_actual", "power_target"})


def test_08_conditional_quantile_interpolation():
    quantiles = np.linspace(0.05, 0.95, 11)
    values = interpolate_quantile_function(quantiles, np.array([0.05, 0.5, 0.95]))
    assert np.allclose(values, [0.05, 0.5, 0.95]); assert deterministic_samples(quantiles).shape == (401,)


def test_09_expected_nmae_computation():
    nmae, _, _ = expected_components(np.array([0.2, 0.4]), np.array([0.3]))
    assert nmae[0] == pytest.approx(0.9)


def test_10_expected_ficr_is_energy_weighted():
    _, ficr, _ = expected_components(np.array([0.2, 0.4]), np.array([0.2]))
    assert ficr[0] == pytest.approx(0.2 / 0.6)


def test_11_official_threshold_parity_is_inclusive():
    _, ficr, _ = expected_components(np.array([0.5]), np.array([0.56 - 1e-12, 0.58 - 1e-12, 0.58001]))
    assert ficr.tolist() == pytest.approx([1.0, 0.75, 0.0])


def test_12_outer_quarter_cannot_enter_selection_or_calibration():
    assert_inner_precedes_outer(["2023Q1", "2023Q2"], "2023Q3")
    with pytest.raises(ValueError): assert_inner_precedes_outer(["2023Q1", "2023Q3"], "2023Q3")


def test_13_decision_grid_contains_q50_and_exp04():
    q = np.linspace(0.1, 0.9, 11); grid = decision_candidates(q, 0.437)
    assert q[5] in grid and 0.437 in grid


def test_14_submission_contract():
    rows=8760
    sample=pd.DataFrame({"forecast_id":[f"f{i}" for i in range(rows)], TIME_COL:pd.date_range("2025-01-01 01:00",periods=rows,freq="h").astype(str), **{t:np.zeros(rows) for t in TARGETS}})
    output=sample.copy(); output[TARGETS]=1.0; validate_submission(output,sample)
