import numpy as np
import pytest
import torch

from experiments.exp03_official_score_calibration.src.ficr_surrogate import (
    official_power_mask,
    soft_ficr_reward,
)
from experiments.exp03_official_score_calibration.src.train_variants import (
    _score_prediction_frame,
    is_better_official_score,
    official_validation_score,
    temporal_sample_weights,
)


def test_official_training_mask_is_inclusive_at_ten_percent():
    target = torch.tensor([[[0.099999, 0.10, 0.11]]])
    mask = official_power_mask(target, torch.ones_like(target, dtype=torch.bool))
    assert mask.tolist() == [[[False, True, True]]]


def test_soft_ficr_reward_respects_official_reward_order():
    reward = soft_ficr_reward(torch.tensor([0.0, 0.07, 0.20]), temperature=0.001)
    assert reward[0] > reward[1] > reward[2]
    assert reward[0].item() == pytest.approx(1.0, abs=1e-5)
    assert reward[1].item() == pytest.approx(0.75, abs=1e-4)


def test_checkpoint_selection_maximizes_official_score():
    assert is_better_official_score(0.61, 0.60)
    assert not is_better_official_score(0.59, 0.60)


def test_numpy_validation_matches_perfect_score():
    target = np.full((2, 24, 3), 0.5)
    mask = np.ones_like(target, dtype=bool)
    score, one_minus_nmae, ficr, groups = official_validation_score(target, target, mask)
    assert score == one_minus_nmae == ficr == pytest.approx(1.0)
    assert len(groups) == 3


def test_winter_and_recency_weights_use_timestamps_only():
    timestamps = np.array([["2022-01-01T01", "2024-07-01T01"]], dtype="datetime64[h]")
    weights = temporal_sample_weights(timestamps, winter_weight=1.15, year_weights={2022: 0.7, 2024: 1.0})
    assert weights[0, 0, 0] == pytest.approx(1.15 * 0.7)
    assert weights[0, 1, 0] == pytest.approx(1.0)


def test_prediction_scoring_accepts_non_contiguous_source_index():
    import pandas as pd

    rows = []
    for group, target in enumerate(("kpx_group_1", "kpx_group_2", "kpx_group_3"), start=1):
        capacity = 21000.0 if group == 3 else 21600.0
        rows.append({"target": target, "y_true_kwh": capacity * 0.5, "y_pred_kwh": capacity * 0.5})
    frame = pd.DataFrame(rows, index=[10, 20, 30])
    result = _score_prediction_frame(frame)
    assert result["total_score"] == pytest.approx(1.0)
