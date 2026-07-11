from pathlib import Path

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
TIME_COL = "forecast_kst_dtm"
SCHEMA_VERSION = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
