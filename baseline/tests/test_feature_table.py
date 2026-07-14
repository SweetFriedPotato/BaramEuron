from baram.constants import TARGETS, TIME_COL
from baram.feature_builder import get_features_for_group, label_table, merge_labels


def test_feature_table_timestamp_unique(feature_tables):
    train_features, test_features = feature_tables
    assert train_features[TIME_COL].is_unique
    assert test_features[TIME_COL].is_unique


def test_train_test_schema_and_test_rows(feature_tables):
    train_features, test_features = feature_tables
    assert list(train_features.columns) == list(test_features.columns)
    assert len(test_features) == 8760


def test_no_scada_or_targets_in_features(feature_tables):
    train_features, test_features = feature_tables
    for frame in (train_features, test_features):
        lowered = [c.lower() for c in frame.columns]
        assert not any("scada" in c for c in lowered)
        assert not any(target in frame.columns for target in TARGETS)


def test_group_feature_selector_schema(feature_tables):
    train_features, test_features = feature_tables
    for group_id in (1, 2, 3):
        train_group = get_features_for_group(train_features, group_id)
        test_group = get_features_for_group(test_features, group_id)
        assert list(train_group.columns) == list(test_group.columns)
        assert any(c.startswith(f"group_{group_id}__") for c in train_group.columns)
        assert not any(c.startswith("group_") and not c.startswith(f"group_{group_id}__") for c in train_group.columns)


def test_label_merge_keeps_target_masks_separate(cfg, feature_tables):
    train_features, _ = feature_tables
    labels = label_table(cfg)
    labeled = merge_labels(train_features, labels)
    assert labeled["kpx_group_1"].notna().sum() != labeled["kpx_group_3"].notna().sum()
