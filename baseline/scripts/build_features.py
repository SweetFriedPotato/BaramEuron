#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baram.config import load_config
from baram.feature_builder import build_feature_tables, feature_cache_dir

p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--force",action="store_true")
a=p.parse_args()
cfg=load_config(a.config)
tr,te=build_feature_tables(cfg,a.force)
cache=feature_cache_dir(cfg)
print(json.dumps({
    "train_shape": list(tr.shape),
    "test_shape": list(te.shape),
    "cache_dir": str(cache),
    "train_features": str(cache/"train_features_raw.parquet"),
    "test_features": str(cache/"test_features_raw.parquet"),
    "labels": str(cache/"train_labels.parquet"),
    "metadata": str(cache/"feature_metadata.json"),
}, ensure_ascii=False, indent=2))
