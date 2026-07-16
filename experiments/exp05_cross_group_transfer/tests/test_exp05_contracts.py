from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from baram.data import load_sample_submission
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config
from experiments.exp03_official_score_calibration.src.official_scorer import score_long
from experiments.exp05_cross_group_transfer.src.constrained_blend import (
    Penalties,
    apply_group_weights,
    regularization_penalty,
)
from experiments.exp05_cross_group_transfer.src.cross_group_attention import CrossGroupAttentionBlock
from experiments.exp05_cross_group_transfer.src.make_submission import make_submission
from experiments.exp05_cross_group_transfer.src.nested_rolling import (
    assert_nested_order,
    nested_outer_plan,
)
from experiments.exp05_cross_group_transfer.src.oof_contract import (
    EXPECTED_GLOBAL_SCORE,
    assert_prediction_alignment,
    load_oof_contract,
    score_prediction,
)
from experiments.exp05_cross_group_transfer.src.residual_stacker import bounded_correction
from experiments.exp05_cross_group_transfer.src.stacker_features import build_stacker_features


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def oof():
    return load_oof_contract()


def test_exp04_reference_score_reproduction(oof):
    observed = score_prediction(oof, "global_blend_prediction")
    assert abs(observed["total_score"] - EXPECTED_GLOBAL_SCORE) < 1e-8


def test_stacking_contract_contains_oof_only(oof):
    assert set(oof["quarter"]) == {f"{year}Q{quarter}" for year in (2023, 2024) for quarter in (1, 2, 3, 4)}
    assert not any("full" in column.lower() or "test" in column.lower() for column in oof)
    assert oof["lead_time_h"].between(12, 35).all()


def test_nested_quarter_time_order():
    assert_nested_order(["2023Q1", "2023Q2"], "2023Q3")
    with pytest.raises(ValueError, match="leaked"):
        assert_nested_order(["2023Q1", "2023Q3"], "2023Q3")


def test_outer_evaluation_target_is_not_in_fit_plan():
    plan = nested_outer_plan(["2023Q1", "2023Q2", "2023Q3"])
    assert plan[2]["fit_quarters"] == ["2023Q1", "2023Q2"]
    assert plan[2]["evaluation_quarter"] not in plan[2]["fit_quarters"]


def test_base_prediction_timestamp_alignment(oof):
    left = oof[["quarter", TIME_COL, "target", "group_id"]].copy()
    assert_prediction_alignment(left, left.copy())
    right = left.copy(); right.loc[right.index[0], TIME_COL] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="mismatch"):
        assert_prediction_alignment(left, right)


def test_group_specific_blend_calculation():
    frame = pd.DataFrame({
        "target": TARGETS,
        "exp03_prediction": [10.0, 20.0, 30.0],
        "raw_prediction": [20.0, 40.0, 60.0],
    })
    weights = dict(zip(TARGETS, [0.1, 0.2, 0.3]))
    result = apply_group_weights(frame, weights, "prediction")
    assert np.allclose(result["prediction"], [11.0, 24.0, 39.0])


def test_shrinkage_regularization_penalty_calculation():
    weights = np.asarray([[0.4, 0.4, 0.4], [0.3, 0.4, 0.5]])
    penalties = Penalties(0.002, 0.003, 0.001)
    values = regularization_penalty(weights, penalties, np.asarray([0.0, 0.02]))
    expected = 0.002 * (0.01 + 0.01) + 0.003 * np.var([0.3, 0.4, 0.5]) + 0.001 * 0.02
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(expected)


def test_residual_correction_bound():
    correction = bounded_correction(np.asarray([-1e9, 100.0, 1e9]), TARGETS[0], 0.5, 0.05)
    bound = CAPACITY_KWH[TARGETS[0]] * 0.05 * 0.5
    assert correction[0] == pytest.approx(-bound)
    assert correction[-1] == pytest.approx(bound)
    assert correction[1] == pytest.approx(50.0)


def test_oof_test_feature_schema_is_identical():
    time = pd.Timestamp("2024-01-01 12:00:00")
    rows = []
    for group, target in enumerate(TARGETS, 1):
        rows.append({TIME_COL: time, "target": target, "group_id": group,
                     "exp03_prediction": 1000.0, "raw_prediction": 1100.0,
                     "base_prediction": 1040.0})
    frame = pd.DataFrame(rows)
    weather = pd.DataFrame({TIME_COL: [time], "ldaps_ws50_mean": [1.0], "ldaps_ws50_max": [2.0],
                            "gfs_ws100_mean": [3.0], "gfs_ws100_max": [4.0], "gfs_ws850_mean": [5.0],
                            "gust_mean": [6.0], "air_density": [1.2],
                            **{f"g{group}_{kind}_hub_wind": [7.0]
                               for group in (1, 2, 3) for kind in ("nearest", "distance_weighted")}})
    _, oof_schema = build_stacker_features(frame, weather)
    _, test_schema = build_stacker_features(frame.copy(), weather.copy())
    assert oof_schema == test_schema


def test_cross_group_attention_shape_and_gradient():
    block = CrossGroupAttentionBlock(128, heads=2, dropout=0.0)
    values = torch.randn(2, 24, 3, 128, requires_grad=True)
    output = block(values); output.mean().backward()
    assert output.shape == values.shape
    assert block.last_attention.shape == (2 * 24, 2, 3, 3)
    assert values.grad is not None


def test_target_and_scada_are_not_model_inputs_or_stacker_features():
    source = (ROOT / "experiments/exp05_cross_group_transfer/src/stacker_features.py").read_text()
    feature_list = source[source.index("feature_columns = ["):source.index("forbidden =")]
    assert "scada" not in feature_list.lower()
    assert "target lag" not in feature_list.lower()
    assert "y_true" not in feature_list.lower()


def test_exp05_score_matches_official_scorer(oof):
    complete_times = oof.groupby(TIME_COL)["target"].nunique().loc[lambda x: x.eq(3)].index
    complete = oof.loc[oof[TIME_COL].isin(complete_times)]
    long = complete[[TIME_COL, "target", "group_id", "y_true_kwh", "global_blend_prediction"]].rename(
        columns={"global_blend_prediction": "y_pred_kwh"}
    )
    official = score_long(long)
    exp05 = score_prediction(complete, "global_blend_prediction")
    assert exp05["total_score"] == pytest.approx(official.total_score, abs=1e-12)
    assert exp05["ficr"] == pytest.approx(official.ficr, abs=1e-12)


def test_exp05_submission_contract(tmp_path):
    sample = load_sample_submission(baseline_config())
    predictions = sample.copy()
    for index, target in enumerate(TARGETS):
        predictions[target] = np.linspace(index, index + 1, len(sample))
    submission = make_submission(sample, predictions, tmp_path / "submission.csv")
    assert len(submission) == 8760
    assert list(pd.to_datetime(submission[TIME_COL])) == list(pd.to_datetime(sample[TIME_COL]))
    assert np.isfinite(submission[TARGETS].to_numpy(dtype=float)).all()
