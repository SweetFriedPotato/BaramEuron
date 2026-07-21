"""Immutable reference and checkpoint contracts for Exp07."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.exp02_daily_tcn_scada_aux.src.models import build_model as build_tcn
from experiments.exp03_official_score_calibration.src.backtest import ROLLING_QUARTERS
from experiments.exp05_cross_group_transfer.src.oof_contract import (
    EXPECTED_GLOBAL_SCORE,
    load_oof_contract,
    write_oof_checks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP03_OUTPUT = PROJECT_ROOT / "experiments/exp03_official_score_calibration/outputs"
EXP04_OUTPUT = PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs"
SCORER_PATH = PROJECT_ROOT / "official/dacon_baram_metric/metric.ipynb"
EXPECTED_SCORER_SHA256 = "0a3ab5a57dba0705dbdbda73cd723be37ef39cce388fcb22b1a220ce523a70f9"


@dataclass(frozen=True)
class CheckpointContract:
    model_id: str
    quarter: str
    path: str
    sha256: str
    stored_seed: int
    stored_variant: str
    state_keys: int
    preprocessing_evidence: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rolling_checkpoint_path(model_id: str, quarter: str, output_root: Path | None = None) -> Path:
    if quarter not in ROLLING_QUARTERS:
        raise ValueError(f"unknown rolling quarter: {quarter}")
    if model_id == "exp03":
        root = Path(output_root or EXP03_OUTPUT)
        return root / "checkpoints" / f"rolling_ficr_lambda_02_{quarter}_seed_42.pt"
    if model_id == "raw":
        root = Path(output_root or EXP04_OUTPUT)
        return root / "checkpoints" / f"rolling_raw_hybrid_gated_{quarter}_seed_42.pt"
    raise ValueError(f"unknown model_id: {model_id}")


def _preprocessing_evidence(model_id: str, quarter: str) -> tuple[Path, ...]:
    if model_id == "exp03":
        return (
            EXP03_OUTPUT / "checks/tcn_feature_manifest.json",
            EXP03_OUTPUT / "run_manifest.json",
        )
    return (
        EXP04_OUTPUT / "checks/preprocessors" / f"rolling_{quarter}_thermo.json",
        EXP04_OUTPUT / "run_manifest.json",
    )


def validate_checkpoint(
    model_id: str,
    quarter: str,
    *,
    output_root: Path | None = None,
) -> CheckpointContract:
    path = rolling_checkpoint_path(model_id, quarter, output_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"required immutable {model_id} checkpoint is missing: {path}; "
            "restore the original Exp03/Exp04 Drive artifact before fine-tuning"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "config", "seed"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint metadata missing {sorted(missing)}: {path}")
    seed = int(payload["seed"])
    if seed != 42:
        raise ValueError(f"rolling seed checkpoint must be 42, got {seed}: {path}")
    config = payload["config"]
    expected = "ficr_lambda_02" if model_id == "exp03" else "raw_hybrid_gated"
    variant = str(config.get("experiment_id", ""))
    if variant != expected:
        raise ValueError(f"checkpoint variant {variant!r} != {expected!r}: {path}")
    evidence = _preprocessing_evidence(model_id, quarter)
    missing_evidence = [str(item) for item in evidence if not item.is_file()]
    if missing_evidence:
        raise FileNotFoundError(f"preprocessing evidence missing: {missing_evidence}")
    return CheckpointContract(
        model_id=model_id,
        quarter=quarter,
        path=str(path),
        sha256=sha256(path),
        stored_seed=seed,
        stored_variant=variant,
        state_keys=len(payload["state_dict"]),
        preprocessing_evidence=tuple(str(item) for item in evidence),
    )


def load_tcn_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_dim = int(payload.get("feature_dim", 0))
    if feature_dim <= 0:
        first = payload["state_dict"].get("input_projection.0.weight")
        if first is None:
            raise ValueError("cannot infer Exp03 feature dimension")
        feature_dim = int(first.shape[1])
    model = build_tcn(payload["config"], feature_dim)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def write_reference_contract(output_root: Path, *, require_checkpoints: bool = True) -> dict:
    output_root = Path(output_root)
    data = load_oof_contract()
    reproduction = write_oof_checks(data, output_root)
    if abs(reproduction["reproduced"]["total_score"] - EXPECTED_GLOBAL_SCORE) >= 1e-8:
        raise AssertionError("Exp04 reference reproduction exceeded tolerance")
    scorer_hash = sha256(SCORER_PATH)
    if scorer_hash != EXPECTED_SCORER_SHA256:
        raise ValueError(f"official scorer hash changed: {scorer_hash}")
    contracts, missing = [], []
    for model_id in ("exp03", "raw"):
        for quarter in ROLLING_QUARTERS:
            try:
                contracts.append(asdict(validate_checkpoint(model_id, quarter)))
            except FileNotFoundError as exc:
                missing.append(str(exc))
    result = {
        "reference": reproduction,
        "official_scorer_sha256": scorer_hash,
        "expected_official_scorer_sha256": EXPECTED_SCORER_SHA256,
        "checkpoint_contracts": contracts,
        "missing_checkpoint_contracts": missing,
        "fine_tuning_allowed": not missing,
    }
    path = output_root / "checks/checkpoint_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if require_checkpoints and missing:
        raise FileNotFoundError("; ".join(missing))
    return result

