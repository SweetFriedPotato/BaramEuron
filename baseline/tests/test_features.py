def test_hourly_unique_and_same_schema(feature_tables):
    tr,te=feature_tables; assert tr.forecast_kst_dtm.is_unique; assert te.forecast_kst_dtm.is_unique; assert list(tr)==list(te)
def test_no_scada_or_target(feature_tables):
    cols=[c.lower() for c in feature_tables[0].columns]; assert not any("scada" in c or "kpx_group" in c for c in cols)

