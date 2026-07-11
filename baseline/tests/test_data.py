from baram.data import load_ldaps,load_gfs,load_labels
def test_grid_counts(cfg):
    assert load_ldaps("train",cfg).groupby("forecast_kst_dtm").grid_id.nunique().eq(16).all()
    assert load_gfs("train",cfg).groupby("forecast_kst_dtm").grid_id.nunique().eq(9).all()
def test_group3_has_no_2022_labels(cfg):
    d=load_labels(cfg); assert d.loc[d.kst_dtm.dt.year==2022,"kpx_group_3"].isna().all()

