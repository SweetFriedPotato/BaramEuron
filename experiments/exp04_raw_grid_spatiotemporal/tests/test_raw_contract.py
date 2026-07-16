from pathlib import Path

import numpy as np
import pytest

from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_contract import validate_raw_contract
from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_loader import (
    GFS_WIND_CHANNELS,
    LDAPS_WIND_CHANNELS,
    channel_manifest,
    load_raw_grid_bundle,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def bundles():
    return load_raw_grid_bundle(ROOT / "open", "train"), load_raw_grid_bundle(ROOT / "open", "test")


def test_raw_tensor_shape_and_issue_block_hours(bundles):
    train, test = bundles
    assert train.ldaps.dynamic.shape == (1096, 24, 16, 16)
    assert train.gfs.dynamic.shape == (1096, 24, 9, 26)
    assert test.ldaps.dynamic.shape == (365, 24, 16, 16)
    assert test.gfs.dynamic.shape == (365, 24, 9, 26)


def test_grid_order_and_dynamic_schema_match_train_test(bundles):
    train, test = bundles
    assert np.array_equal(train.ldaps.grid_ids, np.arange(1, 17))
    assert np.array_equal(train.gfs.grid_ids, np.arange(1, 10))
    assert np.array_equal(train.ldaps.grid_ids, test.ldaps.grid_ids)
    assert np.array_equal(train.gfs.grid_ids, test.gfs.grid_ids)
    assert train.ldaps.channel_names == test.ldaps.channel_names
    assert train.gfs.channel_names == test.gfs.channel_names
    assert train.ldaps.selected_channels(False) == LDAPS_WIND_CHANNELS
    assert train.gfs.selected_channels(False) == GFS_WIND_CHANNELS


def test_issue_time_precedes_every_forecast_and_labels_align(bundles):
    train, test = bundles
    for bundle in (train, test):
        assert np.all(bundle.forecast_times > bundle.issue_times[:, None])
        assert np.all(np.diff(bundle.forecast_times.astype("datetime64[h]").astype(int), axis=1) == 1)
    assert train.targets_cf.shape == (1096, 24, 3)
    assert train.label_mask.shape == train.targets_cf.shape


def test_scada_target_and_target_lag_are_not_inputs(bundles):
    manifest = channel_manifest(bundles[0])
    assert manifest["forbidden_input_matches"] == []
    all_names = sum(
        [manifest["ldaps_wind"], manifest["ldaps_thermodynamic"],
         manifest["gfs_wind"], manifest["gfs_thermodynamic"], manifest["static"]], []
    )
    assert all("scada" not in name.lower() and "target" not in name.lower() and "lag" not in name.lower()
               for name in all_names)


def test_group_static_features_have_required_shapes_and_differ(bundles):
    train, _ = bundles
    assert train.ldaps_group_static.shape == (3, 16, 11)
    assert train.gfs_group_static.shape == (3, 9, 11)
    assert not np.allclose(train.ldaps_group_static[0], train.ldaps_group_static[1])
    contract = validate_raw_contract(*bundles)
    assert contract["checks"]["ldaps_grid_order_equal"]
    assert contract["checks"]["gfs_channel_schema_equal"]
