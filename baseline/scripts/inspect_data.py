#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baram.config import load_config
from baram.data import data_contract, time_semantics


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_config(args.config)

out_dir = Path(cfg["output_root"]) / "checks"
contract = data_contract(cfg)
semantics = time_semantics(cfg)
write_json(contract, out_dir / "data_contract.json")
write_json(semantics, out_dir / "time_semantics.json")
print(json.dumps({
    "data_contract": str(out_dir / "data_contract.json"),
    "time_semantics": str(out_dir / "time_semantics.json"),
    "train_forecast_times": contract["weather"]["ldaps_train"]["unique_timestamps"],
    "test_forecast_times": contract["weather"]["ldaps_test"]["unique_timestamps"],
    "ldaps_test_missing_rows": len(contract["weather"]["ldaps_test"]["ldaps_test_missing_locations"]),
}, ensure_ascii=False, indent=2))
