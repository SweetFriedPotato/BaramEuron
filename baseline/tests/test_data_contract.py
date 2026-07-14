from baram.constants import TARGETS
from baram.data import load_gfs, load_labels, load_ldaps, load_sample_submission


def test_ldaps_grid_counts(cfg):
    assert load_ldaps("train", cfg).groupby("forecast_kst_dtm").grid_id.nunique().eq(16).all()
    assert load_ldaps("test", cfg).groupby("forecast_kst_dtm").grid_id.nunique().eq(16).all()


def test_gfs_grid_counts(cfg):
    assert load_gfs("train", cfg).groupby("forecast_kst_dtm").grid_id.nunique().eq(9).all()
    assert load_gfs("test", cfg).groupby("forecast_kst_dtm").grid_id.nunique().eq(9).all()


def test_target_masks_and_group3_2022(cfg):
    labels = load_labels(cfg)
    masks = {target: labels[target].notna() for target in TARGETS}
    assert masks["kpx_group_1"].sum() > masks["kpx_group_3"].sum()
    assert masks["kpx_group_2"].sum() > masks["kpx_group_3"].sum()
    assert labels.loc[labels.kst_dtm.dt.year == 2022, "kpx_group_3"].isna().all()


def test_submission_rows(cfg):
    sample = load_sample_submission(cfg)
    assert len(sample) == 8760
    assert sample.forecast_kst_dtm.is_unique
