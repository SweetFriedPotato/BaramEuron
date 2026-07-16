import numpy as np
import pandas as pd
import pytest
import torch

from baram.constants import TARGETS, TIME_COL
from baram.data import load_sample_submission
from baram.feature_builder import load_raw_feature_artifacts
from baram.submission import create_submission
from experiments.exp02_daily_tcn_scada_aux.src.blend import align_predictions
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import (
    SelectedFeatureUnionBuilder,
    baseline_config,
    fold_time_mask,
    inspect_issue_blocks,
    issue_mapping,
)
from experiments.exp02_daily_tcn_scada_aux.src.losses import group_balanced_masked_l1
from experiments.exp02_daily_tcn_scada_aux.src.preprocessing import NeuralFoldPreprocessor
from experiments.exp02_daily_tcn_scada_aux.src.scada_targets import AuxiliaryTargetScaler
from experiments.exp02_daily_tcn_scada_aux.src.sequence_builder import build_sequences, fold_bundle


def _synthetic_issue_data():
    times = pd.date_range("2022-01-01 01:00:00", periods=48, freq="h")
    issues = np.repeat(pd.to_datetime(["2021-12-31 13:00", "2022-01-01 13:00"]), 24)
    features = pd.DataFrame({TIME_COL: times, "feature": np.repeat([1.0, 2.0], 24)})
    mapping = pd.DataFrame({TIME_COL: times, "data_available_kst_dtm": issues})
    labels = pd.DataFrame({TIME_COL: times, **{target: np.arange(48, dtype=float) for target in TARGETS}})
    return features, mapping, labels


def test_issue_blocks_are_exactly_24_hours():
    cfg = baseline_config()
    for split, expected in (("train", 1096), ("test", 365)):
        summary, incomplete = inspect_issue_blocks(issue_mapping(cfg, split), split)
        assert summary["issue_blocks"] == expected
        assert summary["forecast_hours_distribution"] == {"24": expected}
        assert summary["all_hourly_contiguous"]
        assert incomplete.empty


def test_sequence_shape_is_daily_multioutput():
    features, mapping, labels = _synthetic_issue_data()
    bundle, incomplete = build_sequences(features, mapping, labels)
    assert incomplete.empty
    assert bundle.x.shape == (2, 24, 1)
    assert bundle.y_cf.shape == (2, 24, 3)
    assert bundle.label_mask.shape == (2, 24, 3)
    assert bundle.timestamps.shape == (2, 24)


def test_train_test_selected_feature_schema_and_forbidden_features():
    cfg = baseline_config(); train, test, _ = load_raw_feature_artifacts(cfg)
    builder = SelectedFeatureUnionBuilder(cfg)
    selected_train = builder.fit_transform(train, fold_time_mask(train[TIME_COL], "fold_b", "train"))
    selected_test = builder.transform("test", test)
    manifest = builder.manifest(selected_train, selected_test)
    assert manifest["train_test_schema_equal"]
    assert not any(manifest["forbidden_feature_flags"].values())
    assert manifest["common_features_not_duplicated"]
    assert manifest["train_inf_count"] == manifest["test_inf_count"] == 0


def test_future_issue_features_are_not_mixed():
    features, mapping, labels = _synthetic_issue_data()
    bundle, _ = build_sequences(features, mapping, labels)
    assert np.all(bundle.x[0] == 1.0)
    assert np.all(bundle.x[1] == 2.0)
    assert bundle.issue_times[0] != bundle.issue_times[1]


def test_scada_is_not_in_input_feature_tensor():
    features, mapping, labels = _synthetic_issue_data()
    bundle, _ = build_sequences(features, mapping, labels)
    assert bundle.feature_names == ["feature"]
    assert all("scada" not in name.lower() and "aux" not in name.lower() for name in bundle.feature_names)


def test_test_sequence_does_not_require_scada():
    features, mapping, _ = _synthetic_issue_data()
    bundle, _ = build_sequences(features, mapping, labels=None, aux_targets=None)
    assert bundle.aux_wind is None
    assert bundle.aux_mask is None
    assert not bundle.label_mask.any()


def test_fold_a_group3_label_mask_is_zero():
    features, mapping, labels = _synthetic_issue_data()
    bundle, _ = build_sequences(features, mapping, labels)
    folded = fold_bundle(bundle, "fold_a", "train")
    assert not folded.label_mask[:, :, 2].any()


def test_group_balanced_loss_averages_group_losses():
    prediction = torch.tensor([[[2.0, 6.0, 100.0], [4.0, 10.0, 100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, True, False], [True, False, False]]])
    loss = group_balanced_masked_l1(prediction, target, mask)
    # group 1 mean=3, group 2 mean=6; group 3 unavailable => (3+6)/2
    assert torch.isclose(loss, torch.tensor(4.5))


def test_neural_preprocessor_fits_train_only_statistics():
    train = np.array([[[0.0, np.nan], [1.0, 2.0]], [[2.0, 4.0], [3.0, 6.0]]], dtype=float)
    valid = np.array([[[1000.0, 1000.0]]], dtype=float)
    preprocessor = NeuralFoldPreprocessor(0.0, 1.0).fit(train)
    before = preprocessor.mean_.copy()
    transformed = preprocessor.transform(valid)
    assert np.array_equal(before, preprocessor.mean_)
    assert np.isfinite(transformed).all()
    assert preprocessor.fit_rows_ == 4


def test_auxiliary_extreme_mask_uses_train_bounds():
    values = np.arange(30, dtype=float).reshape(2, 5, 3)
    mask = np.ones_like(values, dtype=bool)
    scaler = AuxiliaryTargetScaler(0.1, 0.9).fit(values, mask)
    candidate = values.copy(); candidate[0, 0, 0] = 999.0
    _, transformed_mask = scaler.transform(candidate, mask)
    assert not transformed_mask[0, 0, 0]
    assert transformed_mask[:, :, 1].any()


def test_submission_contract_has_8760_rows(tmp_path):
    cfg = baseline_config(); sample = load_sample_submission(cfg)
    path = tmp_path / "submission.csv"
    created = create_submission(sample, {target: np.zeros(len(sample)) for target in TARGETS}, path)
    assert len(created) == 8760
    assert path.exists()


def test_catboost_tcn_timestamp_alignment_is_strict():
    frame = pd.DataFrame(
        {
            TIME_COL: pd.date_range("2024-01-01", periods=2, freq="h"),
            "target": [TARGETS[0]] * 2,
            "group_id": [1, 1],
            "fold": ["fold_b"] * 2,
            "y_true_kwh": [1.0, 2.0],
            "y_pred_kwh": [1.5, 2.5],
        }
    )
    assert len(align_predictions(frame, frame)) == 2
    with pytest.raises(ValueError):
        align_predictions(frame, frame.iloc[:1])
