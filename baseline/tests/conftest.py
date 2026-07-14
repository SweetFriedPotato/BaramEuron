import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"baseline/src"))
from baram.config import load_config
from baram.features.common import build_feature_tables
@pytest.fixture(scope="session")
def cfg(): return load_config(ROOT/"baseline/configs/preprocessing.yaml")
@pytest.fixture(scope="session")
def feature_tables(cfg): return build_feature_tables(cfg)
