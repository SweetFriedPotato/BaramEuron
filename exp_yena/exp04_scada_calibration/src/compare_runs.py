from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed Exp04 validation runs")
    parser.add_argument("--outputs", type=Path, default=Path("exp_yena/exp04_scada_calibration/outputs"))
    args = parser.parse_args()
    rows = []
    for result_path in sorted(args.outputs.glob("*/val_results.yaml")):
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        rows.append({
            "run": result_path.parent.name,
            "total_score": result["total_score"],
            "one_minus_nmae": result["one_minus_nmae"],
            "ficr": result["ficr"],
        })
    if not rows:
        raise FileNotFoundError(f"No val_results.yaml files found below {args.outputs}")
    comparison = pd.DataFrame(rows).sort_values("total_score", ascending=False)
    comparison.to_csv(args.outputs / "run_comparison.csv", index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
