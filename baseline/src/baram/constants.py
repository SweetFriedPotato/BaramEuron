from pathlib import Path

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
TIME_COL = "forecast_kst_dtm"
LABEL_TIME_COL = "kst_dtm"
GROUP_IDS = [1, 2, 3]
TARGET_TO_GROUP = {"kpx_group_1": 1, "kpx_group_2": 2, "kpx_group_3": 3}
GROUP_TO_TARGET = {v: k for k, v in TARGET_TO_GROUP.items()}
SCHEMA_VERSION = "2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
