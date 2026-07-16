from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from baram.data import load_sample_submission
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config
from experiments.exp05_cross_group_transfer.src.nested_rolling import assert_nested_order, nested_outer_plan
from experiments.exp05_cross_group_transfer.src.oof_contract import EXPECTED_GLOBAL_SCORE, score_prediction
from experiments.exp06_ficr_threshold_calibration.src.ficr_gate import (
    BlendGate,
    build_gate_features,
)
from experiments.exp06_ficr_threshold_calibration.src.make_submission import write_diagnostic_submission
from experiments.exp06_ficr_threshold_calibration.src.oof_loader import MODEL_COLUMNS, load_exp06_oof
from experiments.exp06_ficr_threshold_calibration.src.piecewise_calibration import (
    CalibrationPenalty,
    calibration_regularization,
    fit_band_boundaries,
    fit_piecewise,
)
from experiments.exp06_ficr_threshold_calibration.src.threshold_audit import (
    reward_from_error,
    tier_from_error,
    transition_matrix,
    write_tier_check,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def oof():
    return load_exp06_oof()


def test_exp04_reference_exact_reproduction(oof):
    assert score_prediction(oof, "global_blend_prediction")["total_score"] == pytest.approx(
        EXPECTED_GLOBAL_SCORE, abs=1e-12
    )


def test_tier_boundaries_are_inclusive_at_six_and_eight_percent():
    values = np.asarray([0.06, np.nextafter(0.06, 1), 0.08, np.nextafter(0.08, 1)])
    assert tier_from_error(values).tolist() == ["tier_4", "tier_3", "tier_3", "tier_0"]
    assert reward_from_error(values).tolist() == [4.0, 3.0, 3.0, 0.0]


def test_tier_reward_matches_official_ficr(tmp_path):
    rows = []
    time = pd.Timestamp("2024-01-01 12:00")
    for group, target in enumerate(TARGETS, 1):
        capacity = CAPACITY_KWH[target]
        row = {TIME_COL: time, "quarter": "2024Q1", "target": target, "group_id": group,
               "capacity_kwh": capacity, "y_true_kwh": .5*capacity, "official_mask": True}
        for column in MODEL_COLUMNS.values(): row[column] = .5*capacity
        rows.append(row)
    result = write_tier_check(pd.DataFrame(rows), tmp_path / "tier.json")
    assert all(value["official_ficr"] == 1.0 for value in result["models"].values())


def test_transition_matrix_counts_all_rows():
    capacity = 21600.0; actual = .5*capacity
    frame = pd.DataFrame({
        "quarter": ["2024Q1"]*4, TIME_COL: pd.date_range("2024-01-01", periods=4, freq="h"),
        "target": [TARGETS[0]]*4, "group_id": [1]*4, "capacity_kwh": [capacity]*4,
        "y_true_kwh": [actual]*4, "official_mask": [True]*4,
        "global_blend_prediction": actual+capacity*np.asarray([.05,.05,.07,.09]),
        "ridge_prediction": actual+capacity*np.asarray([.05,.07,.05,.05]),
    })
    table = transition_matrix(frame, {"ridge": "ridge_prediction"})
    assert table["count"].sum() == 4
    counts = table.set_index(["from_tier", "to_tier"])["count"]
    assert counts.loc[("tier_4", "tier_3")] == 1
    assert counts.loc[("tier_0", "tier_4")] == 1


def test_nested_quarter_time_order():
    assert_nested_order(["2023Q1", "2023Q2"], "2023Q3")
    with pytest.raises(ValueError): assert_nested_order(["2023Q3"], "2023Q3")


def test_evaluation_quarter_is_absent_from_fit_plan():
    plan = nested_outer_plan(["2023Q1", "2023Q2", "2023Q3"])
    assert plan[-1]["fit_quarters"] == ["2023Q1", "2023Q2"]
    assert plan[-1]["evaluation_quarter"] not in plan[-1]["fit_quarters"]


def test_quantile_piecewise_bins_fit_on_past_oof_only(oof):
    past = oof.loc[oof["quarter"].isin(["2023Q1", "2023Q2"])]
    before = fit_band_boundaries(past, "quantile_three")
    changed_evaluation = oof.loc[oof["quarter"].eq("2023Q3")].copy()
    changed_evaluation["global_blend_prediction"] *= 100
    after = fit_band_boundaries(past, "quantile_three")
    assert before == after


def test_piecewise_parameters_respect_scale_and_offset_bounds():
    config = yaml.safe_load((ROOT / "experiments/exp06_ficr_threshold_calibration/configs/piecewise_affine.yaml").read_text())
    target = TARGETS[0]; capacity = CAPACITY_KWH[target]
    data = pd.DataFrame({"target": [target]*20, "global_blend_prediction": np.linspace(.2,.9,20)*capacity,
                         "y_true_kwh": np.linspace(.21,.88,20)*capacity})
    model, _ = fit_piecewise(data, "physical_three", CalibrationPenalty(.001,.001,.001), config)
    assert model.parameters["scale"].between(.97, 1.03).all()
    assert model.parameters["offset_fraction"].between(-.02, .02).all()


def test_identity_and_smoothness_regularization():
    parameters = pd.DataFrame({"target": [TARGETS[0]]*3, "bin": [0,1,2],
                               "scale": [1.0,1.01,1.02], "offset_fraction": [0,.01,.02]})
    value = calibration_regularization(parameters)
    assert value["identity"] > 0
    assert value["smoothness"] == pytest.approx(.0001)
    assert value["instability"] == 0


def _gate_frame():
    time = pd.date_range("2025-01-01 01:00", periods=3, freq="h")
    return pd.DataFrame({TIME_COL: time, "target": TARGETS, "group_id": [1,2,3],
                         "capacity_kwh": [21600,21600,21000], "exp03_prediction": [1000,2000,3000],
                         "raw_prediction": [1100,1900,3100], "lead_time_h": [12,13,14]})


def test_gate_inputs_exclude_target_scada_and_lags():
    _, columns = build_gate_features(_gate_frame())
    assert all("target" not in value.lower() and "scada" not in value.lower() and "lag" not in value.lower()
               for value in columns)


def test_gate_output_is_between_zero_and_one():
    output = BlendGate(5)(torch.randn(20, 5))
    assert torch.all((output >= 0) & (output <= 1))


def test_oof_and_test_gate_schema_match():
    _, left = build_gate_features(_gate_frame())
    _, right = build_gate_features(_gate_frame().copy())
    assert left == right


def test_submission_contract(tmp_path):
    sample = load_sample_submission(baseline_config())
    predictions = sample.copy()
    for index, target in enumerate(TARGETS): predictions[target] = index+np.linspace(0,1,len(sample))
    output = write_diagnostic_submission(sample, predictions, tmp_path/"submission.csv", "unused", False)
    assert len(output) == 8760
    assert np.isfinite(output[TARGETS].to_numpy(dtype=float)).all()
