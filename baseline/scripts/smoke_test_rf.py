#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baram.config import load_config
from baram.constants import CAPACITY_KWH, GROUP_TO_TARGET, TARGETS, TARGET_TO_GROUP, TIME_COL
from baram.feature_builder import (
    build_feature_tables,
    get_features_for_group,
    label_table,
    merge_labels,
)
from baram.preprocessing import fit_tree_preprocessor
from baram.submission import create_submission
from baram.data import load_sample_submission
from baram.validation import split_labeled_table, validation_split_summary


def dump_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_prediction(pred, target, config):
    if not config.get("smoke_test", {}).get("clip_predictions", True):
        return pred
    return np.clip(pred, 0, CAPACITY_KWH[target])


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--force-features", action="store_true")
args = parser.parse_args()

cfg = load_config(args.config)
train_features, test_features = build_feature_tables(cfg, force=args.force_features)
labels = label_table(cfg)
labeled = merge_labels(train_features, labels)

checks_dir = Path(cfg["output_root"]) / "checks"
split_summary = validation_split_summary(labeled, cfg)
dump_json(split_summary, checks_dir / "validation_split_summary.json")

out_dir = Path(cfg["output_root"]) / "baseline_preprocessing_smoke"
out_dir.mkdir(parents=True, exist_ok=True)

rf_params = dict(cfg.get("smoke_test", {}).get("model", {}))
rf_params.setdefault("random_state", cfg.get("seed", 42))

metrics = {
    "note": "RandomForest smoke test for preprocessing validation only; not a tuned final model.",
    "groups": {},
}
prediction_frames = []
submission_predictions = {}

for target in TARGETS:
    group_id = TARGET_TO_GROUP[target]
    group_train = get_features_for_group(train_features, group_id)
    group_test = get_features_for_group(test_features, group_id)
    model_table = merge_labels(group_train, labels)
    train_mask, valid_mask = split_labeled_table(model_table, target, cfg)

    x_cols = [c for c in group_train.columns if c != TIME_COL]
    x_train = model_table.loc[train_mask, x_cols]
    y_train = model_table.loc[train_mask, target].to_numpy()
    x_valid = model_table.loc[valid_mask, x_cols]
    y_valid = model_table.loc[valid_mask, target].to_numpy()
    x_test = group_test[x_cols]

    preprocessor, x_train_z, x_valid_z, x_test_z, feature_names = fit_tree_preprocessor(
        x_train, x_valid, x_test, config=cfg
    )
    model = RandomForestRegressor(**rf_params)
    model.fit(x_train_z, y_train)

    valid_pred = clip_prediction(model.predict(x_valid_z), target, cfg)
    test_pred = clip_prediction(model.predict(x_test_z), target, cfg)
    mae = float(mean_absolute_error(y_valid, valid_pred))
    nmae = float(mae / CAPACITY_KWH[target])
    metrics["groups"][target] = {
        "group_id": int(group_id),
        "mae": mae,
        "nmae": nmae,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "feature_count": int(len(feature_names)),
    }
    prediction_frames.append(pd.DataFrame({
        TIME_COL: model_table.loc[valid_mask, TIME_COL].to_numpy(),
        "target": target,
        "y_true": y_valid,
        "prediction": valid_pred,
    }))
    submission_predictions[target] = test_pred

metrics["macro_mae"] = float(np.mean([v["mae"] for v in metrics["groups"].values()]))
metrics["macro_nmae"] = float(np.mean([v["nmae"] for v in metrics["groups"].values()]))
metrics["rf_params"] = rf_params

dump_json(metrics, out_dir / "metrics.json")
pd.concat(prediction_frames, ignore_index=True).to_csv(
    out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
)
sample = load_sample_submission(cfg)
create_submission(sample, submission_predictions, out_dir / "submission_smoke.csv")

print(json.dumps({
    "metrics": str(out_dir / "metrics.json"),
    "validation_predictions": str(out_dir / "validation_predictions.csv"),
    "submission_smoke": str(out_dir / "submission_smoke.csv"),
    "macro_mae": metrics["macro_mae"],
    "macro_nmae": metrics["macro_nmae"],
}, ensure_ascii=False, indent=2))
