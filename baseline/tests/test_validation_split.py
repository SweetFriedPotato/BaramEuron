from baram.feature_builder import label_table, merge_labels
from baram.validation import split_labeled_table


def test_validation_is_future_for_all_groups(cfg, feature_tables):
    train_features, _ = feature_tables
    labeled = merge_labels(train_features, label_table(cfg))
    for target in ("kpx_group_1", "kpx_group_2", "kpx_group_3"):
        train_mask, valid_mask = split_labeled_table(labeled, target, cfg)
        assert labeled.loc[train_mask, "forecast_kst_dtm"].max() < labeled.loc[valid_mask, "forecast_kst_dtm"].min()


def test_validation_excludes_missing_labels(cfg, feature_tables):
    train_features, _ = feature_tables
    labeled = merge_labels(train_features, label_table(cfg))
    for target in ("kpx_group_1", "kpx_group_2", "kpx_group_3"):
        train_mask, valid_mask = split_labeled_table(labeled, target, cfg)
        assert labeled.loc[train_mask, target].notna().all()
        assert labeled.loc[valid_mask, target].notna().all()


def test_group3_train_has_no_2022_target(cfg, feature_tables):
    train_features, _ = feature_tables
    labeled = merge_labels(train_features, label_table(cfg))
    train_mask, _ = split_labeled_table(labeled, "kpx_group_3", cfg)
    assert not (labeled.loc[train_mask, "forecast_kst_dtm"].dt.year == 2022).any()
