"""Load Exp03/Exp04 rolling OOF and aligned Exp05 diagnostic candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import TIME_COL
from experiments.exp05_cross_group_transfer.src.oof_contract import (
    EXPECTED_GLOBAL_SCORE,
    KEYS,
    assert_prediction_alignment,
    load_oof_contract,
    write_oof_checks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP05_OUTPUT = PROJECT_ROOT / "experiments/exp05_cross_group_transfer/outputs"
SCORER_PATH = PROJECT_ROOT / "official/dacon_baram_metric/metric.ipynb"
SCORER_PY_PATH = PROJECT_ROOT / "official/dacon_baram_metric/metric.py"
EXPECTED_SCORER_SHA256 = "0a3ab5a57dba0705dbdbda73cd723be37ef39cce388fcb22b1a220ce523a70f9"
CANDIDATE_FILES = {
    "constrained_prediction": "predictions/constrained_blend_oof.csv",
    "ridge_prediction": "predictions/ridge_stacker_oof.csv",
    "catboost_prediction": "predictions/catboost_stacker_oof.csv",
    "final_prediction": "predictions/final_candidate_oof.csv",
}
MODEL_COLUMNS = {
    "exp03": "exp03_prediction",
    "raw": "raw_prediction",
    "exp04_global": "global_blend_prediction",
    "exp05_constrained": "constrained_prediction",
    "exp05_ridge": "ridge_prediction",
    "exp05_catboost": "catboost_prediction",
    "exp05_final": "final_prediction",
}


def scorer_sha256(path: Path = SCORER_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exp06_oof(exp05_root: Path = EXP05_OUTPUT) -> pd.DataFrame:
    base = load_oof_contract().sort_values(KEYS).reset_index(drop=True)
    for column, relative in CANDIDATE_FILES.items():
        path = Path(exp05_root) / relative
        candidate = pd.read_csv(path, parse_dates=[TIME_COL]).sort_values(KEYS).reset_index(drop=True)
        assert_prediction_alignment(base, candidate)
        values = candidate[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains NaN/inf")
        base[column] = values
    return base


def write_contract_checks(data: pd.DataFrame, output_root: Path) -> dict:
    output_root = Path(output_root); checks = output_root / "checks"; checks.mkdir(parents=True, exist_ok=True)
    reproduction = write_oof_checks(data, output_root)
    digest = scorer_sha256()
    if digest != EXPECTED_SCORER_SHA256:
        raise ValueError(f"official scorer hash changed: {digest}")
    contract = json.loads((checks / "oof_contract.json").read_text())
    contract.update({
        "candidate_prediction_columns": MODEL_COLUMNS,
        "candidate_alignment_exact": True,
        "official_scorer_sha256": digest,
        "extracted_metric_py_sha256": hashlib.sha256(SCORER_PY_PATH.read_bytes()).hexdigest(),
        "expected_reference_score": EXPECTED_GLOBAL_SCORE,
    })
    (checks / "oof_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    audit = {
        "rolling_oof_only": True,
        "evaluation_quarter_target_used_for_selection": False,
        "full_or_test_target_fit_rows": 0,
        "inference_target_features": [],
        "target_lag_features": [],
        "scada_features": [],
        "raw_source_gate": "unavailable per rolling quarter; neutral fallback only if gate executes",
        "seed_std": "unavailable because raw rolling OOF is seed42 only",
        "public_used_for_selection": False,
    }
    (checks / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return reproduction
