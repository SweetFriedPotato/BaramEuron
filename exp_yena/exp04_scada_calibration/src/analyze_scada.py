from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baram.constants import TIME_COL
from baram.feature_builder import get_features_for_group, load_raw_feature_artifacts

from .scada import load_hourly_scada
from .config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S0: aggregate and audit SCADA forecast offsets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    train_features = load_raw_feature_artifacts(config)[0]
    hourly = load_hourly_scada(config)
    labels = pd.read_csv(
        Path(config["data"]["train_dir"]) / "train_labels.csv", encoding="utf-8-sig"
    ).rename(columns={"kst_dtm": TIME_COL})
    labels[TIME_COL] = pd.to_datetime(labels[TIME_COL])
    summaries = []
    offset_frames = []
    for group_id in (1, 2, 3):
        target = f"kpx_group_{group_id}"
        features = get_features_for_group(train_features, group_id).copy()
        features[TIME_COL] = pd.to_datetime(features[TIME_COL])
        table = (
            features[[TIME_COL, "month", "lead_time_h", "gfs__ws100__mean"]]
            .merge(hourly[group_id], on=TIME_COL, how="left", validate="one_to_one")
            .merge(labels[[TIME_COL, target]], on=TIME_COL, how="left", validate="one_to_one")
        )
        table["group_id"] = group_id
        table["forecast_scada_offset"] = table["scada_ws_mean"] - table["gfs__ws100__mean"]
        table["wind_bin"] = np.floor(table["gfs__ws100__mean"].clip(0, 30) / 2)
        table["lead_bin"] = np.floor(table["lead_time_h"] / 6)
        summaries.append({
            "group_id": group_id,
            "rows": len(table),
            "scada_available_rows": int(table["scada_ws_mean"].notna().sum()),
            "complete_six_sample_hours": int((table["scada_samples"] == 6).sum()),
            "offset_mean": float(table["forecast_scada_offset"].mean()),
            "offset_std": float(table["forecast_scada_offset"].std()),
            "forecast_scada_correlation": float(table[["gfs__ws100__mean", "scada_ws_mean"]].corr().iloc[0, 1]),
            "scada_power_label_correlation": float(table[["scada_power_kwh", target]].corr().iloc[0, 1]),
        })
        offset = (
            table.groupby(["group_id", "month", "lead_bin", "wind_bin"], dropna=False)
            ["forecast_scada_offset"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        offset_frames.append(offset)
        table.to_csv(output / f"scada_hourly_group_{group_id}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(summaries).to_csv(output / "scada_quality_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(offset_frames, ignore_index=True).to_csv(
        output / "forecast_scada_offset_map.csv", index=False, encoding="utf-8-sig"
    )
    (output / "scada_quality_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
