import numpy as np
import pandas as pd
import pytest

from official.dacon_baram_metric.metric import CAPACITY_KWH, TARGET_COLS, metric
from experiments.exp03_official_score_calibration.src.official_scorer import score_wide


def _frames(actual_fraction=0.5, error_fraction=0.0, rows=4):
    answer = pd.DataFrame(
        {target: np.full(rows, capacity * actual_fraction) for target, capacity in CAPACITY_KWH.items()}
    )
    prediction = answer.copy()
    for target, capacity in CAPACITY_KWH.items():
        prediction[target] += capacity * error_fraction
    return answer, prediction


def test_wrapper_matches_official_raw_function():
    answer, prediction = _frames(error_fraction=0.07)
    raw = metric(answer, prediction)
    wrapped = score_wide(answer, prediction)
    assert np.allclose(raw, [wrapped.total_score, wrapped.one_minus_nmae, wrapped.ficr])


def test_perfect_prediction_has_maximum_components():
    answer, prediction = _frames()
    result = score_wide(answer, prediction)
    assert result.total_score == pytest.approx(1.0)
    assert result.one_minus_nmae == pytest.approx(1.0)
    assert result.ficr == pytest.approx(1.0)


def test_official_ten_percent_boundary_is_inclusive():
    answer = pd.DataFrame(
        {
            target: [capacity * 0.10 - 1e-6, capacity * 0.10]
            for target, capacity in CAPACITY_KWH.items()
        }
    )
    result = score_wide(answer, answer.copy())
    assert all(group.evaluated_samples == 1 for group in result.groups)
    assert all(group.total_samples == 2 for group in result.groups)


@pytest.mark.parametrize("error_fraction,expected_ficr", [(0.06, 1.0), (0.08, 0.75), (0.080001, 0.0)])
def test_ficr_thresholds_are_exact(error_fraction, expected_ficr):
    answer, prediction = _frames(error_fraction=error_fraction)
    assert score_wide(answer, prediction).ficr == pytest.approx(expected_ficr)


def test_group_macro_average_is_unweighted():
    answer, prediction = _frames()
    prediction[TARGET_COLS[0]] += CAPACITY_KWH[TARGET_COLS[0]] * 0.03
    prediction[TARGET_COLS[1]] += CAPACITY_KWH[TARGET_COLS[1]] * 0.06
    prediction[TARGET_COLS[2]] += CAPACITY_KWH[TARGET_COLS[2]] * 0.09
    result = score_wide(answer, prediction)
    assert result.one_minus_nmae == pytest.approx(1.0 - (0.03 + 0.06 + 0.09) / 3)
    assert result.ficr == pytest.approx((1.0 + 1.0 + 0.0) / 3)
