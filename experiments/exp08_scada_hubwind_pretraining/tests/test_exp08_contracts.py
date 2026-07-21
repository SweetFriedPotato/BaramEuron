from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from baram.constants import TARGETS, TIME_COL
from experiments.exp03_official_score_calibration.src.train_variants import official_validation_score
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import RawModelInputs
from experiments.exp08_scada_hubwind_pretraining.src.evaluate import reproduce_exp04_reference
from experiments.exp08_scada_hubwind_pretraining.src.make_submission import validate_submission
from experiments.exp08_scada_hubwind_pretraining.src.scada_contract import (
    GROUP_TURBINES,
    assert_test_pipeline_has_no_scada,
    load_scada_wind,
    validate_group_mapping,
)
from experiments.exp08_scada_hubwind_pretraining.src.scada_hourly_targets import (
    FoldScadaCleaner,
    HubWindTargetScaler,
    _hourly_group,
    hour_ending,
)
from experiments.exp08_scada_hubwind_pretraining.src.stage1_crossfit import (
    CrossfitWindow,
    assert_temporal_crossfit,
    expanding_crossfit_windows,
)
from experiments.exp08_scada_hubwind_pretraining.src.stage1_dataset import (
    STAGE1_INPUT_NAMES,
    Stage1Dataset,
    assert_stage1_input_schema,
)
from experiments.exp08_scada_hubwind_pretraining.src.stage1_model import (
    build_stage1_model,
    group_balanced_distribution_loss,
)
from experiments.exp08_scada_hubwind_pretraining.src.stage2_dataset import (
    DISTRIBUTION_FEATURE_INDICES,
    FoldHubFeatureImputer,
    STAGE2_HUB_FEATURES,
    Stage2Dataset,
    assert_stage2_feature_schema,
    build_stage2_hub_features,
)
from experiments.exp08_scada_hubwind_pretraining.src.stage2_model import build_stage2_model


ROOT = Path(__file__).resolve().parents[3]


def _config(stage: int) -> dict:
    value = {
        "use_geo": True, "use_thermo": True, "use_engineered": True, "gated_fusion": True,
        "model": {"token_dim": 8, "attention_heads": 2, "attention_dropout": 0.0,
                  "hidden_channels": 8, "kernel_size": 3, "dilations": [1, 2],
                  "temporal_dropout": 0.0, "non_causal": True},
        "training": {"batch_size": 2, "max_epochs": 3, "patience": 2},
    }
    if stage == 1:
        value["stage1"] = {"target_count": 4}
    else:
        value["stage2"] = {"variant": "distribution_hubwind"}
    return value


def _raw_inputs(batch: int = 2) -> RawModelInputs:
    rng = np.random.default_rng(42)
    return RawModelInputs(
        rng.normal(size=(batch, 24, 2, 3)).astype("float32"),
        rng.normal(size=(batch, 24, 2, 4)).astype("float32"),
        rng.normal(size=(batch, 24, 5)).astype("float32"),
        rng.normal(size=(batch, 24, 3, 4)).astype("float32"),
    )


def _static() -> np.ndarray:
    return np.zeros((3, 2, 11), dtype="float32")


def test_01_exp04_exact_reproduction():
    path = ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs/predictions/best_blend_predictions.csv"
    assert reproduce_exp04_reference(path)["exact_within_1e-12"]


def test_02_scada_hour_ending_alignment_is_right_closed():
    times = pd.Series(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 01:10", "2024-01-01 01:50", "2024-01-01 02:00"]))
    expected = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 02:00", "2024-01-01 02:00"]))
    assert hour_ending(times).equals(expected)


def test_03_group_turbine_mapping_matches_contract_and_real_headers():
    frames = {source: load_scada_wind(ROOT / "open", source) for source in ("vestas", "unison")}
    mapping = validate_group_mapping({name: frame.columns for name, frame in frames.items()})
    assert len(mapping["group_1"]) == 6 and len(mapping["group_2"]) == 6 and len(mapping["group_3"]) == 5
    assert set(GROUP_TURBINES[1]).isdisjoint(GROUP_TURBINES[2])


def test_04_scada_target_mask_requires_half_readings_and_turbines():
    columns = list(GROUP_TURBINES[1])
    times = pd.date_range("2024-01-01 00:10", periods=6, freq="10min")
    frame = pd.DataFrame({"kst_dtm": times, **{column: np.arange(6, dtype=float) for column in columns}})
    frame.loc[3:, columns[0]] = np.nan  # 3/6 remains valid.
    frame.loc[2:, columns[1:4]] = np.nan  # 2/6 invalid for three turbines.
    hourly = _hourly_group(frame, 1)
    assert bool(hourly.loc[0, "group_1__target_mask"])
    frame.loc[2:, columns[4]] = np.nan  # now only two valid turbines.
    assert not bool(_hourly_group(frame, 1).loc[0, "group_1__target_mask"])


def test_05_scada_cleaning_statistics_are_fit_on_fold_train_only():
    columns = list(GROUP_TURBINES[1])
    train_times = pd.date_range("2024-01-01", periods=10, freq="10min")
    frame = pd.DataFrame({"kst_dtm": train_times, **{column: np.arange(1, 11, dtype=float) for column in columns}})
    future = frame.iloc[[-1]].copy(); future["kst_dtm"] = pd.Timestamp("2025-01-01"); future[columns] = 1_000_000.0
    combined = pd.concat([frame, future], ignore_index=True)
    cleaner = FoldScadaCleaner().fit({"vestas": combined}, fit_end=pd.Timestamp("2024-01-02"))
    assert cleaner.states["vestas"].fit_end < "2025"
    assert cleaner.states["vestas"].upper < 11.0
    assert cleaner.transform(combined, "vestas").iloc[-1][columns].isna().all()


def test_06_stage1_inputs_contain_no_scada_target_lag_or_disagreement():
    assert_stage1_input_schema(STAGE1_INPUT_NAMES)
    assert all(token not in " ".join(Stage1Dataset.input_names).lower() for token in ("scada", "target", "lag", "disagreement"))


def test_07_test_pipeline_rejects_any_scada_path():
    assert_test_pipeline_has_no_scada(["open/test/ldaps_test.csv", "open/test/gfs_test.csv"])
    with pytest.raises(ValueError):
        assert_test_pipeline_has_no_scada(["open/train/scada_vestas_train.csv"])
    with pytest.raises(RuntimeError):
        load_scada_wind(ROOT / "open", "vestas", split="test")


def test_08_stage1_crossfit_is_strictly_temporal():
    starts = pd.date_range("2022-01-01 01:00", "2023-03-31 01:00", freq="D")
    timestamps = np.stack([start + pd.to_timedelta(np.arange(24), unit="h") for start in starts]).astype("datetime64[ns]")
    records = expanding_crossfit_windows(timestamps, "2023Q1", min_train_blocks=30)
    assert_temporal_crossfit(records, pd.DatetimeIndex(timestamps[:, 0]))
    assert records[-1].role == "outer_validation"
    assert max(records[-1].train_indices) < min(records[-1].predict_indices)


def test_09_stage2_cannot_receive_in_sample_stage1_prediction():
    invalid = CrossfitWindow("bad", "2023-01-01", "2023-03-01", "2023-02-01", "2023-02-28", (1, 2), (2, 3), False)
    with pytest.raises(ValueError):
        assert_temporal_crossfit([invalid])


def test_10_stage1_loss_is_group_balanced_not_sample_balanced():
    prediction = torch.zeros((1, 4, 3, 1))
    target = torch.zeros_like(prediction); mask = torch.zeros_like(prediction, dtype=torch.bool)
    target[0, :, 0, 0] = 1.0; mask[0, :, 0, 0] = True
    target[0, 0, 1, 0] = 3.0; mask[0, 0, 1, 0] = True
    # SmoothL1(1)=0.5 and SmoothL1(3)=2.5; equal group mean is 1.5.
    assert group_balanced_distribution_loss(prediction, target, mask, (1.0,)).item() == pytest.approx(1.5)


def test_11_stage1_and_stage2_tensor_schemas_match():
    inputs = _raw_inputs(); static = _static()
    stage1 = build_stage1_model(_config(1), 3, 4, static, static, 5, (4, 4, 4))
    distribution, _, _ = stage1(torch.from_numpy(inputs.ldaps), torch.from_numpy(inputs.gfs),
                                 torch.from_numpy(inputs.engineered_common), torch.from_numpy(inputs.engineered_group))
    assert distribution.shape == (2, 24, 3, 4)
    scaler = HubWindTargetScaler().fit(np.ones(distribution.shape), np.ones(distribution.shape, dtype=bool))
    physical = scaler.inverse_transform(distribution.detach().numpy())
    hub = FoldHubFeatureImputer().fit(build_stage2_hub_features(physical, np.ones((2, 24)))).transform(
        build_stage2_hub_features(physical, np.ones((2, 24)))
    )
    stage2 = build_stage2_model(_config(2), 3, 4, static, static, 5, (4, 4, 4))
    power, _, _ = stage2(torch.from_numpy(inputs.ldaps), torch.from_numpy(inputs.gfs),
                         torch.from_numpy(inputs.engineered_common), torch.from_numpy(inputs.engineered_group),
                         torch.from_numpy(hub))
    assert power.shape == (2, 24, 3)
    assert Stage2Dataset(inputs, hub)[0][4].shape == (24, 3, 8)


def test_12_stage2_schema_excludes_target_target_lag_and_availability():
    assert_stage2_feature_schema(STAGE2_HUB_FEATURES)
    joined = " ".join(STAGE2_HUB_FEATURES).lower()
    assert "target" not in joined and "lag" not in joined and "availability" not in joined
    assert len(DISTRIBUTION_FEATURE_INDICES) == 8


def test_13_official_scorer_reuse_matches_known_threshold_rewards():
    # float64 keeps the inclusive 0.06/0.08 boundary values exact enough for
    # the published scorer comparison (float32 addition can cross a boundary).
    target = np.full((1, 3, 3), 0.5, dtype=np.float64)
    prediction = target.copy()
    prediction[:, 0, :] += 0.06 - 1e-12
    prediction[:, 1, :] += 0.08 - 1e-12
    prediction[:, 2, :] += 0.08001
    score, one_minus_nmae, ficr, groups = official_validation_score(prediction, target, np.ones_like(target, dtype=bool))
    assert ficr == pytest.approx(7 / 12)
    assert score == pytest.approx(0.5 * one_minus_nmae + 0.5 * ficr)
    assert len(groups) == 3


def test_14_submission_contract_checks_rows_keys_values_dtypes_and_duplicates():
    rows = 8760
    sample = pd.DataFrame({
        "forecast_id": [f"forecast_{index:04d}" for index in range(rows)],
        TIME_COL: pd.date_range("2025-01-01 01:00", periods=rows, freq="h").astype(str),
        **{target: np.zeros(rows) for target in TARGETS},
    })
    output = sample.copy(); output[TARGETS] = np.arange(rows * 3, dtype=float).reshape(rows, 3)
    validate_submission(output, sample)
    broken = output.copy(); broken.loc[0, TARGETS[0]] = np.nan
    with pytest.raises(ValueError):
        validate_submission(broken, sample)
