"""Nested rolling split and inner-only model-selection contracts."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config, raw_artifacts
from experiments.exp02_daily_tcn_scada_aux.src.run_experiment import prediction_frame as tcn_prediction_frame
from experiments.exp02_daily_tcn_scada_aux.src.scada_targets import build_scada_aux_targets
from experiments.exp03_official_score_calibration.src.backtest import (
    ROLLING_QUARTERS,
    expanding_quarter_window,
    issue_quarter,
)
from experiments.exp03_official_score_calibration.src.train_variants import _prepare_expanding_quarter
from experiments.exp04_raw_grid_spatiotemporal.src.evaluate import prediction_frame as raw_prediction_frame
from experiments.exp04_raw_grid_spatiotemporal.src.raw_grid_loader import load_raw_grid_bundle
from experiments.exp04_raw_grid_spatiotemporal.src.run_experiment import (
    _model as build_raw_model,
    load_variant_config,
    prepare_raw_fold,
)
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import predict_raw
from experiments.exp05_cross_group_transfer.src.oof_contract import load_oof_contract

from .checkpoint_loader import (
    load_tcn_checkpoint,
    rolling_checkpoint_path,
    validate_checkpoint,
    write_reference_contract,
)
from .evaluate import score_column


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp07_threshold_aware_finetuning"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
RAW_CONFIG_PATH = PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/configs/raw_hybrid_gated.yaml"
EXP03_CONFIG_PATH = PROJECT_ROOT / "experiments/exp07_threshold_aware_finetuning/configs/exp03_head_annealed_tau.yaml"
RAW_CONFIG_EXP07_PATH = PROJECT_ROOT / "experiments/exp07_threshold_aware_finetuning/configs/raw_head_annealed_tau.yaml"


@dataclass(frozen=True)
class NestedWindow:
    outer_quarter: str
    outer_start: str
    outer_end: str
    inner_quarter: str | None
    inner_start: str | None
    inner_end: str | None
    fine_tune_end: str | None
    fallback: bool


def quarter_bounds(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    window = expanding_quarter_window(quarter)
    return window["valid_start"], window["valid_end"]


def nested_windows(quarters: list[str] | tuple[str, ...] = ROLLING_QUARTERS) -> list[NestedWindow]:
    windows = []
    for index, outer in enumerate(quarters):
        outer_start, outer_end = quarter_bounds(outer)
        if index == 0:
            windows.append(NestedWindow(
                outer, str(outer_start), str(outer_end), None, None, None, None, True,
            ))
            continue
        inner = quarters[index - 1]
        inner_start, inner_end = quarter_bounds(inner)
        fine_tune_end = inner_start - pd.Timedelta(hours=1)
        windows.append(NestedWindow(
            outer, str(outer_start), str(outer_end), inner, str(inner_start), str(inner_end),
            str(fine_tune_end), False,
        ))
    return windows


def assert_no_outer_leakage(
    train_timestamps,
    inner_timestamps,
    outer_timestamps,
) -> None:
    train = pd.DatetimeIndex(pd.to_datetime(np.asarray(train_timestamps).reshape(-1)))
    inner = pd.DatetimeIndex(pd.to_datetime(np.asarray(inner_timestamps).reshape(-1)))
    outer = pd.DatetimeIndex(pd.to_datetime(np.asarray(outer_timestamps).reshape(-1)))
    if train.empty or inner.empty or outer.empty:
        raise ValueError("nested split cannot be empty")
    if train.max() >= inner.min():
        raise ValueError("fine-tune labels overlap inner validation")
    if inner.max() >= outer.min():
        raise ValueError("inner validation overlaps outer evaluation")


def checkpoint_rank(row: dict) -> tuple:
    return (
        -float(row["ficr"]),
        -float(row["one_minus_nmae"]),
        float(row["parameter_distance"]),
        int(row["epoch"]),
    )


def select_inner_checkpoint(rows: list[dict], tolerance: float = 0.0005) -> dict:
    if not rows:
        raise ValueError("no inner checkpoint candidates")
    best_score = max(float(row["total_score"]) for row in rows)
    tied = [row for row in rows if best_score - float(row["total_score"]) <= tolerance]
    return min(tied, key=checkpoint_rank)


def write_leakage_audit(output_root: Path) -> dict:
    windows = nested_windows()
    audit = {
        "selection_target": "inner validation official Score only",
        "outer_target_uses": ["single final evaluation"],
        "fallback_quarters": [item.outer_quarter for item in windows if item.fallback],
        "windows": [asdict(item) for item in windows],
        "random_split": False,
        "public_score_used": False,
    }
    path = Path(output_root) / "checks/leakage_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _directories(output_root: Path) -> None:
    for name in ("checks", "metrics", "predictions", "checkpoints", "figures", "submissions"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def candidate_configs(model_id: str) -> dict[str, dict]:
    path = EXP03_CONFIG_PATH if model_id == "exp03" else RAW_CONFIG_EXP07_PATH
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = {}
    for candidate_id, changes in {
        "baseline_no_boundary": {
            "tau_schedule": "fixed", "tau_start": 0.006, "tau_end": 0.006,
            "soft_ficr_weight": 0.0, "lambda_boundary": 0.0,
        },
        "fixed_006_lambda_005": {
            "tau_schedule": "fixed", "tau_start": 0.006, "tau_end": 0.006,
            "soft_ficr_weight": 0.20, "lambda_boundary": 0.05,
        },
        "annealed_015_004_lambda_005": {
            "tau_schedule": "cosine", "tau_start": 0.015, "tau_end": 0.004,
            "soft_ficr_weight": 0.20, "lambda_boundary": 0.05,
        },
        "annealed_015_004_lambda_010": {
            "tau_schedule": "cosine", "tau_start": 0.015, "tau_end": 0.004,
            "soft_ficr_weight": 0.20, "lambda_boundary": 0.10,
        },
    }.items():
        config = copy.deepcopy(base); config.update(changes); config["candidate_id"] = candidate_id
        variants[candidate_id] = config
    return variants


def _split_indices(timestamps: np.ndarray, inner_quarter: str) -> tuple[np.ndarray, np.ndarray]:
    quarters = issue_quarter(pd.Series(pd.to_datetime(timestamps[:, 0]))).to_numpy()
    inner = np.flatnonzero(quarters == inner_quarter)
    train = np.flatnonzero(
        np.asarray([pd.Period(value, freq="Q") < pd.Period(inner_quarter, freq="Q") for value in quarters])
    )
    if not len(train) or not len(inner):
        raise ValueError(f"empty nested train/inner split for {inner_quarter}")
    return train, inner


def _canonical_outer(reference: pd.DataFrame, quarter: str, model_id: str) -> pd.DataFrame:
    column = "exp03_prediction" if model_id == "exp03" else "raw_prediction"
    out = reference.loc[reference["quarter"].eq(quarter)].copy()
    out["incumbent_component_prediction"] = out[column]
    return out


def _merge_outer_prediction(canonical: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    keys = ["quarter", TIME_COL, "target", "group_id"]
    values = prediction[keys + ["y_pred_kwh"]].rename(columns={"y_pred_kwh": "prediction"})
    merged = canonical.merge(values, on=keys, how="left", validate="one_to_one")
    if merged["prediction"].isna().any():
        missing = merged.loc[merged["prediction"].isna(), keys].head().to_dict("records")
        raise ValueError(f"fine-tuned outer prediction is incomplete: {missing}")
    return merged


def _sync_quarter(
    drive_run: Path | None,
    model_id: str,
    candidate_id: str,
    seed: int,
    quarter: str,
    files: list[Path],
) -> None:
    if drive_run is None:
        return
    destination = drive_run / "nested" / model_id / candidate_id / str(seed) / quarter
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        if source.exists():
            shutil.copy2(source, destination / source.name)


def _source_inner_score_tcn(model, x, y, mask, batch_size: int) -> dict:
    from experiments.exp02_daily_tcn_scada_aux.src.trainer import predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prediction = predict(model, x, batch_size, device)
    score, _ = score_column_from_arrays(prediction, y, mask)
    return score


def score_column_from_arrays(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray):
    parts = []
    for group, target_name in enumerate(TARGETS):
        valid = mask[..., group].reshape(-1)
        capacity = float(CAPACITY_KWH[target_name])
        parts.append(pd.DataFrame({
            TIME_COL: pd.date_range("2000-01-01", periods=int(valid.sum()), freq="h"),
            "target": target_name,
            "group_id": group + 1,
            "y_true_kwh": (target[..., group].reshape(-1)[valid] * capacity),
            "y_pred_kwh": (np.maximum(prediction[..., group].reshape(-1)[valid], 0.0) * capacity),
        }))
    return score_column(pd.concat(parts, ignore_index=True), "y_pred_kwh")


def run_nested_model(
    model_id: str,
    config: dict,
    seed: int,
    output_root: Path,
    *,
    drive_run: Path | None = None,
    smoke_epochs: int | None = None,
    raw_bundle=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one fixed candidate; every learned choice uses the prior inner quarter."""
    from .trainer import finetune_raw, finetune_tcn

    if model_id not in {"exp03", "raw"}:
        raise ValueError(model_id)
    reference = load_oof_contract()
    candidate_id = str(config["candidate_id"])
    all_parts, records = [], []
    baseline_cfg = baseline_config()
    train_features, _, labels = raw_artifacts(baseline_cfg)
    scada = build_scada_aux_targets(Path(baseline_cfg["data"]["root"]))
    if model_id == "raw" and raw_bundle is None:
        raw_bundle = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    quarters = ROLLING_QUARTERS[-1:] if smoke_epochs is not None else ROLLING_QUARTERS
    for quarter in quarters:
        canonical = _canonical_outer(reference, quarter, model_id)
        if quarter == ROLLING_QUARTERS[0]:
            canonical["prediction"] = canonical["incumbent_component_prediction"]
            canonical["fallback"] = True
            all_parts.append(canonical)
            records.append({
                "model_id": model_id, "candidate_id": candidate_id, "seed": seed,
                "outer_quarter": quarter, "fallback": True,
            })
            continue
        inner_quarter = ROLLING_QUARTERS[ROLLING_QUARTERS.index(quarter) - 1]
        prepared = _prepare_expanding_quarter(
            quarter, baseline_cfg, train_features, labels, scada
        )
        train_idx, inner_idx = _split_indices(prepared.train.timestamps, inner_quarter)
        assert_no_outer_leakage(
            prepared.train.timestamps[train_idx], prepared.train.timestamps[inner_idx],
            prepared.valid.timestamps,
        )
        source = rolling_checkpoint_path(model_id, quarter)
        validate_checkpoint(model_id, quarter)
        run_config = copy.deepcopy(config)
        if smoke_epochs is not None:
            run_config["training"]["max_epochs"] = int(smoke_epochs)
            run_config["training"]["patience"] = int(smoke_epochs)
        checkpoint = output_root / "checkpoints" / (
            f"{model_id}_{candidate_id}_{quarter}_seed_{seed}.pt"
        )
        if model_id == "exp03":
            model, _ = load_tcn_checkpoint(source)
            batch_size = int(run_config["training"].get("batch_size", 32))
            source_inner = _source_inner_score_tcn(
                model, prepared.train_x[inner_idx], prepared.train.y_cf[inner_idx],
                prepared.train.label_mask[inner_idx], batch_size,
            )
            result = finetune_tcn(
                model, prepared.train_x[train_idx], prepared.train.y_cf[train_idx],
                prepared.train.label_mask[train_idx], prepared.train_x[inner_idx],
                prepared.train.y_cf[inner_idx], prepared.train.label_mask[inner_idx],
                prepared.valid_x, run_config, seed, source, checkpoint,
            )
            frame = tcn_prediction_frame(prepared, result.outer_prediction_cf, candidate_id, seed)
        else:
            raw_prepared = prepare_raw_fold(
                raw_bundle, prepared, True, output_root / "checks",
                f"nested_{quarter}_thermo",
            )
            model = build_raw_model(load_variant_config(RAW_CONFIG_PATH), raw_prepared)
            payload = torch.load(source, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"], strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            batch_size = int(run_config["training"].get("batch_size", 16))
            source_inner_prediction = predict_raw(
                model, raw_prepared.train_inputs.subset(inner_idx), batch_size, device
            )[0]
            source_inner, _ = score_column_from_arrays(
                source_inner_prediction, raw_prepared.train_y[inner_idx],
                raw_prepared.train_mask[inner_idx],
            )
            result = finetune_raw(
                model, raw_prepared.train_inputs.subset(train_idx), raw_prepared.train_y[train_idx],
                raw_prepared.train_mask[train_idx], raw_prepared.train_inputs.subset(inner_idx),
                raw_prepared.train_y[inner_idx], raw_prepared.train_mask[inner_idx],
                raw_prepared.valid_inputs, run_config, seed, source, checkpoint,
            )
            frame = raw_prediction_frame(
                raw_prepared.valid_timestamps, raw_prepared.valid_y, raw_prepared.valid_mask,
                result.outer_prediction_cf, candidate_id, quarter, seed,
                raw_prepared.validation_wind, raw_prepared.high_wind_threshold,
            )
        frame["quarter"] = quarter
        outer = _merge_outer_prediction(canonical, frame)
        outer["fallback"] = False; all_parts.append(outer)
        record = {
            "model_id": model_id, "candidate_id": candidate_id, "seed": seed,
            "outer_quarter": quarter, "inner_quarter": inner_quarter, "fallback": False,
            "source_inner_score": source_inner["total_score"],
            "inner_total_score": result.best_total_score,
            "inner_one_minus_nmae": result.best_one_minus_nmae,
            "inner_ficr": result.best_ficr,
            "inner_score_delta": result.best_total_score - source_inner["total_score"],
            "best_epoch": result.best_epoch,
            "parameter_distance": result.best_parameter_distance,
            "training_seconds": result.training_seconds,
            "device": result.device,
            "source_checkpoint": result.source_checkpoint,
            "source_checkpoint_sha256": result.source_checkpoint_sha256,
        }
        records.append(record)
        quarter_path = output_root / "predictions" / (
            f"nested_{model_id}_{candidate_id}_{quarter}_seed_{seed}.csv"
        )
        outer.to_csv(quarter_path, index=False)
        metric_path = checkpoint.with_suffix(".metrics.json"); _write_json(metric_path, record)
        _sync_quarter(
            drive_run, model_id, candidate_id, seed, quarter,
            [checkpoint, checkpoint.with_suffix(".history.json"), metric_path, quarter_path],
        )
    predictions = pd.concat(all_parts, ignore_index=True)
    run_table = pd.DataFrame(records)
    if smoke_epochs is None:
        prediction_path = output_root / "predictions" / f"nested_{model_id}_{candidate_id}_seed_{seed}.csv"
        predictions.to_csv(prediction_path, index=False)
    return predictions, run_table


def run_head_candidates(output_root: Path, drive_run: Path | None = None) -> dict:
    raw_bundle = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    loss_rows, nested_rows = [], []
    for model_id in ("exp03", "raw"):
        for candidate_id, config in candidate_configs(model_id).items():
            predictions, runs = run_nested_model(
                model_id, config, 42, output_root, drive_run=drive_run, raw_bundle=raw_bundle
            )
            score, _ = score_column(predictions, "prediction")
            valid_runs = runs.loc[~runs["fallback"]].copy()
            row = {
                "model_id": model_id, "candidate_id": candidate_id,
                "selection_inner_score_mean": float(valid_runs["inner_total_score"].mean()),
                "selection_inner_ficr_mean": float(valid_runs["inner_ficr"].mean()),
                "selection_inner_nmae_mean": float(valid_runs["inner_one_minus_nmae"].mean()),
                "selection_parameter_distance_mean": float(valid_runs["parameter_distance"].mean()),
                "outer_total_score_diagnostic": score["total_score"],
                "outer_one_minus_nmae_diagnostic": score["one_minus_nmae"],
                "outer_ficr_diagnostic": score["ficr"],
            }
            loss_rows.append(row); nested_rows.append(runs)
            pd.DataFrame(loss_rows).to_csv(output_root / "metrics/loss_ablation.csv", index=False)
            pd.concat(nested_rows, ignore_index=True).to_csv(
                output_root / "metrics/nested_quarter_scores.csv", index=False
            )
    table = pd.DataFrame(loss_rows)
    selected = {}
    for model_id, part in table.groupby("model_id"):
        best = part.sort_values(
            ["selection_inner_score_mean", "selection_inner_ficr_mean",
             "selection_inner_nmae_mean", "selection_parameter_distance_mean"],
            ascending=[False, False, False, True],
        ).iloc[0]
        selected[model_id] = str(best["candidate_id"])
    result = {"selection_source": "inner validation only", "selected": selected}
    _write_json(output_root / "head_selection.json", result)
    return result


def run_smoke(output_root: Path, drive_run: Path | None = None) -> dict:
    raw_bundle = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    rows = []
    for model_id in ("exp03", "raw"):
        config = candidate_configs(model_id)["annealed_015_004_lambda_005"]
        _, runs = run_nested_model(
            model_id, config, 42, output_root, drive_run=drive_run,
            smoke_epochs=3, raw_bundle=raw_bundle,
        )
        rows.append(runs)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_root / "metrics/smoke_scores.csv", index=False)
    return {"runs": len(result), "devices": sorted(result["device"].dropna().unique())}


def _load_or_run_selected(
    model_id: str,
    config: dict,
    seed: int,
    output_root: Path,
    drive_run: Path | None,
    raw_bundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = output_root / "predictions" / f"nested_{model_id}_{config['candidate_id']}_seed_{seed}.csv"
    metrics_path = output_root / "metrics/nested_quarter_scores.csv"
    if path.exists() and metrics_path.exists():
        predictions = pd.read_csv(path, parse_dates=[TIME_COL])
        runs = pd.read_csv(metrics_path)
        runs = runs.loc[
            runs["model_id"].eq(model_id)
            & runs["candidate_id"].eq(config["candidate_id"])
            & runs["seed"].eq(seed)
        ].copy()
        if len(runs) == len(ROLLING_QUARTERS):
            return predictions, runs
    predictions, runs = run_nested_model(
        model_id, config, seed, output_root, drive_run=drive_run, raw_bundle=raw_bundle
    )
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        combined = pd.concat([old, runs], ignore_index=True, sort=False).drop_duplicates(
            ["model_id", "candidate_id", "seed", "outer_quarter"], keep="last"
        )
    else:
        combined = runs
    combined.to_csv(metrics_path, index=False)
    return predictions, runs


def run_selected_seeds(output_root: Path, drive_run: Path | None = None) -> dict:
    selection = json.loads((output_root / "head_selection.json").read_text())
    raw_bundle = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    seed_rows = []
    for model_id in ("exp03", "raw"):
        candidate_id = selection["selected"][model_id]
        config = candidate_configs(model_id)[candidate_id]
        for seed in (42, 52, 62):
            predictions, runs = _load_or_run_selected(
                model_id, config, seed, output_root, drive_run, raw_bundle
            )
            score, _ = score_column(predictions, "prediction")
            incumbent, _ = score_column(predictions, "incumbent_component_prediction")
            valid = runs.loc[~runs["fallback"].astype(bool)]
            seed_rows.append({
                "model_id": model_id, "candidate_id": candidate_id, "seed": seed,
                **score,
                "incumbent_component_score": incumbent["total_score"],
                "score_delta": score["total_score"] - incumbent["total_score"],
                "inner_score_mean": float(valid["inner_total_score"].mean()),
                "inner_score_delta_mean": float(valid["inner_score_delta"].mean()),
            })
            pd.DataFrame(seed_rows).to_csv(output_root / "metrics/seed_scores.csv", index=False)
    table = pd.DataFrame(seed_rows)
    result = {
        model_id: {
            "candidate_id": selection["selected"][model_id],
            "mean_outer_score": float(part["total_score"].mean()),
            "outer_improved_seed_count": int((part["score_delta"] > 0).sum()),
            "mean_inner_score_delta": float(part["inner_score_delta_mean"].mean()),
        }
        for model_id, part in table.groupby("model_id")
    }
    _write_json(output_root / "seed_selection.json", result)
    return result


def run_last_block_if_eligible(output_root: Path, drive_run: Path | None = None) -> dict:
    head = json.loads((output_root / "head_selection.json").read_text())["selected"]
    seeds = pd.read_csv(output_root / "metrics/seed_scores.csv")
    nested = pd.read_csv(output_root / "metrics/nested_quarter_scores.csv")
    raw_bundle = load_raw_grid_bundle(PROJECT_ROOT / "open", "train")
    result = {}
    for model_id in ("exp03", "raw"):
        candidate_id = head[model_id]
        head_runs = nested.loc[
            nested["model_id"].eq(model_id) & nested["candidate_id"].eq(candidate_id)
            & nested["seed"].eq(42) & ~nested["fallback"].astype(bool)
        ]
        eligible = float(head_runs["inner_score_delta"].mean()) > 0
        entry = {
            "eligible": eligible,
            "eligibility_source": "mean inner validation delta only",
            "head_candidate": candidate_id,
            "selected_candidate": candidate_id,
            "executed": False,
        }
        if eligible:
            config = candidate_configs(model_id)[candidate_id]
            config = copy.deepcopy(config)
            config["candidate_id"] = f"last_block_{candidate_id}"
            config["freeze_policy"] = "last_block"
            config["training"].update({
                "head_learning_rate": 5e-5,
                "block_learning_rate": 1e-5,
                "max_epochs": 15,
                "patience": 4,
            })
            prediction42, runs42 = _load_or_run_selected(
                model_id, config, 42, output_root, drive_run, raw_bundle
            )
            last_inner = float(runs42.loc[~runs42["fallback"].astype(bool), "inner_total_score"].mean())
            head_inner = float(head_runs["inner_total_score"].mean())
            entry.update({"executed": True, "last_block_inner_score": last_inner,
                          "head_inner_score": head_inner})
            if last_inner > head_inner + 1e-12:
                entry["selected_candidate"] = config["candidate_id"]
                for seed in (52, 62):
                    predictions, runs = _load_or_run_selected(
                        model_id, config, seed, output_root, drive_run, raw_bundle
                    )
                    score, _ = score_column(predictions, "prediction")
                    incumbent, _ = score_column(predictions, "incumbent_component_prediction")
                    valid = runs.loc[~runs["fallback"].astype(bool)]
                    row = {
                        "model_id": model_id, "candidate_id": config["candidate_id"], "seed": seed,
                        **score, "incumbent_component_score": incumbent["total_score"],
                        "score_delta": score["total_score"] - incumbent["total_score"],
                        "inner_score_mean": float(valid["inner_total_score"].mean()),
                        "inner_score_delta_mean": float(valid["inner_score_delta"].mean()),
                    }
                    seeds = pd.concat([seeds, pd.DataFrame([row])], ignore_index=True)
                score, _ = score_column(prediction42, "prediction")
                incumbent, _ = score_column(prediction42, "incumbent_component_prediction")
                row42 = {
                    "model_id": model_id, "candidate_id": config["candidate_id"], "seed": 42,
                    **score, "incumbent_component_score": incumbent["total_score"],
                    "score_delta": score["total_score"] - incumbent["total_score"],
                    "inner_score_mean": last_inner,
                    "inner_score_delta_mean": float(runs42.loc[~runs42["fallback"].astype(bool), "inner_score_delta"].mean()),
                }
                seeds = pd.concat([seeds, pd.DataFrame([row42])], ignore_index=True)
        result[model_id] = entry
    seeds.drop_duplicates(["model_id", "candidate_id", "seed"], keep="last").to_csv(
        output_root / "metrics/seed_scores.csv", index=False
    )
    _write_json(output_root / "component_selection.json", result)
    return result


def _ensemble_component(output_root: Path, model_id: str, candidate_id: str) -> pd.DataFrame:
    parts = []
    for seed in (42, 52, 62):
        path = output_root / "predictions" / f"nested_{model_id}_{candidate_id}_seed_{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=[TIME_COL])
        parts.append(frame[["quarter", TIME_COL, "target", "group_id", "prediction"]].assign(seed=seed))
    keys = ["quarter", TIME_COL, "target", "group_id"]
    return pd.concat(parts, ignore_index=True).groupby(keys, sort=False)["prediction"].mean().reset_index()


def finalize_nested(output_root: Path) -> dict:
    from experiments.exp05_cross_group_transfer.src.evaluate import slice_metrics
    from .blend import blend_prediction, search_global_blends
    from .evaluate import (
        acceptance, boundary_region_scores, summarize_candidate, threshold_transitions, rescue_gain,
    )
    from .make_report import write_report

    component_selection = json.loads((output_root / "component_selection.json").read_text())
    reference = load_oof_contract()
    exp03_id = component_selection["exp03"]["selected_candidate"]
    raw_id = component_selection["raw"]["selected_candidate"]
    exp03 = _ensemble_component(output_root, "exp03", exp03_id).rename(
        columns={"prediction": "finetuned_exp03_prediction"}
    )
    raw = _ensemble_component(output_root, "raw", raw_id).rename(
        columns={"prediction": "finetuned_raw_prediction"}
    )
    keys = ["quarter", TIME_COL, "target", "group_id"]
    data = reference.merge(exp03, on=keys, how="left", validate="one_to_one")
    data = data.merge(raw, on=keys, how="left", validate="one_to_one")
    if data[["finetuned_exp03_prediction", "finetuned_raw_prediction"]].isna().any().any():
        raise ValueError("fine-tuned component ensemble is incomplete")
    pairs = {
        "A_original_exp03_original_raw": ("exp03_prediction", "raw_prediction"),
        "B_finetuned_exp03_original_raw": ("finetuned_exp03_prediction", "raw_prediction"),
        "C_original_exp03_finetuned_raw": ("exp03_prediction", "finetuned_raw_prediction"),
        "D_finetuned_exp03_finetuned_raw": ("finetuned_exp03_prediction", "finetuned_raw_prediction"),
    }
    search, candidates = search_global_blends(data, pairs)
    search.to_csv(output_root / "metrics/blend_search.csv", index=False)
    best = search.iloc[0]
    final = candidates.loc[
        candidates["combination"].eq(best["combination"])
        & candidates["raw_weight"].eq(float(best["raw_weight"]))
    ].copy()
    final.to_csv(output_root / "predictions/final_nested_oof.csv", index=False)
    component_rows, group_parts = [], []
    for model_id, column in {
        "original_exp03": "exp03_prediction",
        "finetuned_exp03": "finetuned_exp03_prediction",
        "original_raw": "raw_prediction",
        "finetuned_raw": "finetuned_raw_prediction",
        "exp04_global": "global_blend_prediction",
        "exp07_best_blend": "blend_prediction",
    }.items():
        source = final if column == "blend_prediction" else data
        summary, quarters, groups = summarize_candidate(source, column, model_id)
        component_rows.append(summary); group_parts.append(groups)
        if model_id == "exp07_best_blend":
            quarters.to_csv(output_root / "metrics/nested_quarter_scores_final.csv", index=False)
    components = pd.DataFrame(component_rows)
    components.to_csv(output_root / "metrics/component_scores.csv", index=False)
    pd.concat(group_parts, ignore_index=True).to_csv(output_root / "metrics/group_scores.csv", index=False)
    incumbent = components.loc[components["model_id"].eq("exp04_global")].iloc[0].to_dict()
    candidate = components.loc[components["model_id"].eq("exp07_best_blend")].iloc[0].to_dict()
    transitions = threshold_transitions(
        final, "global_blend_prediction", "blend_prediction", "exp07_best_blend"
    )
    transitions.to_csv(output_root / "metrics/threshold_transitions.csv", index=False)
    rescue = rescue_gain(transitions)
    boundary_region_scores(
        final, "global_blend_prediction", "blend_prediction"
    ).to_csv(output_root / "metrics/boundary_region_scores.csv", index=False)
    january, high = slice_metrics(final, "blend_prediction")
    january.to_csv(output_root / "metrics/january_scores.csv", index=False)
    high.to_csv(output_root / "metrics/high_wind_scores.csv", index=False)

    seed_rows = []
    exp03_column, raw_column = pairs[str(best["combination"])]
    for seed in (42, 52, 62):
        seed_data = reference.copy()
        if exp03_column.startswith("finetuned"):
            part = pd.read_csv(
                output_root / "predictions" / f"nested_exp03_{exp03_id}_seed_{seed}.csv",
                parse_dates=[TIME_COL],
            )[keys + ["prediction"]].rename(columns={"prediction": exp03_column})
            seed_data = seed_data.merge(part, on=keys, how="left", validate="one_to_one")
        if raw_column.startswith("finetuned"):
            part = pd.read_csv(
                output_root / "predictions" / f"nested_raw_{raw_id}_seed_{seed}.csv",
                parse_dates=[TIME_COL],
            )[keys + ["prediction"]].rename(columns={"prediction": raw_column})
            seed_data = seed_data.merge(part, on=keys, how="left", validate="one_to_one")
        seed_data["seed_prediction"] = blend_prediction(
            seed_data[exp03_column], seed_data[raw_column], float(best["raw_weight"])
        )
        metric, _ = score_column(seed_data, "seed_prediction")
        seed_rows.append({"seed": seed, **metric,
                          "delta_vs_exp04": metric["total_score"] - incumbent["total_score"]})
    seed_stability = pd.DataFrame(seed_rows)
    seed_stability.to_csv(output_root / "metrics/final_seed_scores.csv", index=False)
    seed_ok = (
        float(seed_stability["total_score"].mean()) > incumbent["total_score"]
        and int((seed_stability["delta_vs_exp04"] > 0).sum()) >= 2
    )
    config = yaml.safe_load((EXPERIMENT_DIR / "configs/final_blend.yaml").read_text())["acceptance"]
    decision = acceptance(candidate, incumbent, rescue=rescue, seed_mean_improved=seed_ok, config=config)

    clipping_rows = []
    upper = final.groupby("target")["y_true_kwh"].max().to_dict()
    for clipping in ("none", "nominal_capacity", "eda_observed_upper"):
        clipped = final.copy()
        if clipping == "nominal_capacity":
            clipped["clipped_prediction"] = np.minimum(
                clipped["blend_prediction"], clipped["capacity_kwh"]
            )
        elif clipping == "eda_observed_upper":
            clipped["clipped_prediction"] = np.minimum(
                clipped["blend_prediction"], clipped["target"].map(upper)
            )
        else:
            clipped["clipped_prediction"] = clipped["blend_prediction"]
        score, _ = score_column(clipped, "clipped_prediction")
        clipping_rows.append({"clipping": clipping, **score})
    clipping = pd.DataFrame(clipping_rows).sort_values("total_score", ascending=False)
    clipping.to_csv(output_root / "metrics/clipping_diagnostic.csv", index=False)

    final_selection = {
        "accepted": decision["accepted"],
        "acceptance": decision,
        "combination": str(best["combination"]),
        "raw_weight": float(best["raw_weight"]),
        "rolling_score": candidate["total_score"],
        "score_delta": candidate["total_score"] - incumbent["total_score"],
        "rescue_gain": rescue,
        "exp03_candidate": exp03_id,
        "raw_candidate": raw_id,
        "seed_score_mean": float(seed_stability["total_score"].mean()),
        "improved_seed_count": int((seed_stability["delta_vs_exp04"] > 0).sum()),
        "selected_clipping": str(clipping.iloc[0]["clipping"]),
        "full_finetuning_required": bool(decision["accepted"]),
        "auto_submitted": False,
    }
    _write_json(output_root / "final_selection.json", final_selection)
    pd.DataFrame([candidate]).to_csv(output_root / "metrics/final_candidate_scores.csv", index=False)
    write_report(output_root)
    return final_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["contract", "smoke", "head-candidates", "seeds", "last-block", "finalize", "all"],
        default="contract",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drive-run", type=Path)
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve(); _directories(output_root)
    drive_run = None if args.drive_run is None else args.drive_run.resolve()
    if drive_run is not None:
        drive_run.mkdir(parents=True, exist_ok=True)
    audit = write_leakage_audit(output_root)
    contract = write_reference_contract(
        output_root,
        require_checkpoints=(args.phase != "contract" or not args.allow_missing_checkpoints),
    )
    if args.phase == "contract":
        result = {"contract": contract, "leakage": audit}
    elif args.phase == "smoke":
        result = {"contract": contract, "leakage": audit, "smoke": run_smoke(output_root, drive_run)}
    elif args.phase == "head-candidates":
        result = {
            "contract": contract, "leakage": audit,
            "head_candidates": run_head_candidates(output_root, drive_run),
        }
    elif args.phase == "seeds":
        result = {"seeds": run_selected_seeds(output_root, drive_run)}
    elif args.phase == "last-block":
        result = {"last_block": run_last_block_if_eligible(output_root, drive_run)}
    elif args.phase == "finalize":
        result = {"final_selection": finalize_nested(output_root)}
    else:
        result = {
            "smoke": run_smoke(output_root, drive_run),
            "head_candidates": run_head_candidates(output_root, drive_run),
            "seeds": run_selected_seeds(output_root, drive_run),
            "last_block": run_last_block_if_eligible(output_root, drive_run),
        }
        result["final_selection"] = finalize_nested(output_root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
