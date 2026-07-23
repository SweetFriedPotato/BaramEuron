from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from shared.constants import CAPACITY_KWH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp05: fine M2 threshold and group calibration search"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--test-diagnostics", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def group_score(actual: np.ndarray, prediction: np.ndarray, capacity: float) -> dict:
    valid = actual >= capacity * 0.10
    actual = actual[valid]
    prediction = prediction[valid]
    error_rate = np.abs(prediction - actual) / capacity
    nmae = float(error_rate.mean())
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    one_minus_nmae = 1.0 - nmae
    return {
        "total_score": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def values(spec: dict) -> np.ndarray:
    start = float(spec["start"])
    stop = float(spec["stop"])
    step = float(spec["step"])
    return np.arange(start, stop + step * 0.5, step)


def objective_score(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    target: str,
    objective: str,
) -> dict:
    capacity = CAPACITY_KWH[target]
    if objective == "pooled":
        return group_score(frame["y_true"].to_numpy(float), prediction, capacity)
    fold_scores = []
    for fold, indices in frame.groupby("fold").groups.items():
        positions = frame.index.get_indexer(indices)
        score = group_score(
            frame.loc[indices, "y_true"].to_numpy(float),
            prediction[positions],
            capacity,
        )
        fold_scores.append((fold, score))
    return {
        "total_score": float(np.mean([item[1]["total_score"] for item in fold_scores])),
        "one_minus_nmae": float(
            np.mean([item[1]["one_minus_nmae"] for item in fold_scores])
        ),
        "ficr": float(np.mean([item[1]["ficr"] for item in fold_scores])),
    }


def search(
    frame: pd.DataFrame,
    config: dict,
    objective: str,
) -> tuple[pd.DataFrame, dict]:
    threshold_values = values(config["search"]["threshold"])
    rows: list[dict] = []
    best_result: dict | None = None

    for threshold in threshold_values:
        group_results: dict[str, dict] = {}
        for target, group in frame.groupby("target", sort=True):
            group = group.reset_index(drop=True)
            probability = group["generation_probability"].to_numpy(float)
            regression = group["regression_prediction"].to_numpy(float)
            base_prediction = np.where(probability >= threshold, regression, 0.0)
            search_spec = config["search"]["groups"][target]
            best_group: dict | None = None
            for scale in values(search_spec["scale"]):
                for bias in values(search_spec["bias"]):
                    prediction = np.maximum(0.0, base_prediction * scale + bias)
                    metrics = objective_score(group, prediction, target, objective)
                    candidate = {
                        "scale": float(scale),
                        "bias": float(bias),
                        **metrics,
                    }
                    if best_group is None or (
                        candidate["total_score"],
                        candidate["ficr"],
                        candidate["one_minus_nmae"],
                    ) > (
                        best_group["total_score"],
                        best_group["ficr"],
                        best_group["one_minus_nmae"],
                    ):
                        best_group = candidate
            assert best_group is not None
            group_results[target] = best_group

        total_score = float(
            np.mean([result["total_score"] for result in group_results.values()])
        )
        one_minus_nmae = float(
            np.mean([result["one_minus_nmae"] for result in group_results.values()])
        )
        ficr = float(np.mean([result["ficr"] for result in group_results.values()]))
        row = {
            "objective": objective,
            "threshold": float(threshold),
            "total_score": total_score,
            "one_minus_nmae": one_minus_nmae,
            "ficr": ficr,
        }
        for target, result in group_results.items():
            row[f"{target}_scale"] = result["scale"]
            row[f"{target}_bias"] = result["bias"]
            row[f"{target}_score"] = result["total_score"]
        rows.append(row)
        candidate_result = {
            **row,
            "groups": group_results,
        }
        if best_result is None or (
            total_score,
            ficr,
            one_minus_nmae,
        ) > (
            best_result["total_score"],
            best_result["ficr"],
            best_result["one_minus_nmae"],
        ):
            best_result = candidate_result

    assert best_result is not None
    results = pd.DataFrame(rows).sort_values(
        ["total_score", "ficr", "one_minus_nmae"], ascending=False
    )
    return results, best_result


def experiment_config(result: dict) -> dict:
    return {
        "base_config": "exp_yena/exp04_scada_calibration/configs/m2_two_stage.yaml",
        "two_stage": {
            "gating_mode": "hard",
            "probability_threshold": float(result["threshold"]),
            "group_calibration": {
                target: {
                    "scale": float(group["scale"]),
                    "bias": float(group["bias"]),
                }
                for target, group in result["groups"].items()
            },
        },
        "search_score": {
            "objective": result["objective"],
            "total_score": float(result["total_score"]),
            "one_minus_nmae": float(result["one_minus_nmae"]),
            "ficr": float(result["ficr"]),
        },
    }


def apply_to_test(
    diagnostics: pd.DataFrame,
    sample_path: Path,
    result: dict,
    destination: Path,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    output = sample.copy()
    threshold = float(result["threshold"])
    summary = []
    for target, group in diagnostics.groupby("target", sort=True):
        group = group.sort_values("forecast_kst_dtm").reset_index(drop=True)
        probability = group["generation_probability"].to_numpy(float)
        regression = group["regression_prediction"].to_numpy(float)
        calibration = result["groups"][target]
        prediction = np.where(probability >= threshold, regression, 0.0)
        prediction = np.maximum(
            0.0,
            prediction * float(calibration["scale"]) + float(calibration["bias"]),
        )
        if len(prediction) != len(output):
            raise ValueError(
                f"{target} prediction length {len(prediction)} != sample length {len(output)}"
            )
        output[target] = prediction
        summary.append({
            "target": target,
            "threshold": threshold,
            "scale": calibration["scale"],
            "bias": calibration["bias"],
            "prediction_mean": float(prediction.mean()),
            "zero_rate": float(np.mean(prediction == 0.0)),
        })
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    return pd.DataFrame(summary)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    oof_path = args.oof or Path(config["inputs"]["oof"])
    output_root = args.output_root or Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(oof_path)
    required = {
        "fold",
        "target",
        "y_true",
        "generation_probability",
        "regression_prediction",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"OOF file is missing required columns: {missing}")

    best_results: dict[str, dict] = {}
    for objective in ("pooled", "fold_robust"):
        table, best = search(frame, config, objective)
        table.to_csv(output_root / f"{objective}_threshold_search.csv", index=False)
        best_config = experiment_config(best)
        (output_root / f"best_{objective}.yaml").write_text(
            yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8"
        )
        best_results[objective] = best
        print(yaml.safe_dump({objective: best_config}, sort_keys=False))

    diagnostics_path = args.test_diagnostics
    if diagnostics_path is None and config.get("inputs", {}).get("test_diagnostics"):
        diagnostics_path = Path(config["inputs"]["test_diagnostics"])
    sample_path = args.sample_submission
    if sample_path is None and config.get("inputs", {}).get("sample_submission"):
        sample_path = Path(config["inputs"]["sample_submission"])
    if diagnostics_path and sample_path and diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        summaries = []
        for objective, best in best_results.items():
            destination = output_root / f"submission_exp05_{objective}.csv"
            summary = apply_to_test(diagnostics, sample_path, best, destination)
            summary.insert(0, "objective", objective)
            summaries.append(summary)
        pd.concat(summaries, ignore_index=True).to_csv(
            output_root / "submission_diagnostics.csv", index=False
        )

    (output_root / "search_manifest.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "oof": str(oof_path),
                "rows": len(frame),
                "folds": sorted(frame["fold"].unique().tolist()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
