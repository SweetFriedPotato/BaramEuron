#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baram.config import load_config
from baram.features.common import build_feature_tables
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--force",action="store_true"); a=p.parse_args(); tr,te=build_feature_tables(load_config(a.config),a.force); print(f"train={tr.shape}, test={te.shape}")

