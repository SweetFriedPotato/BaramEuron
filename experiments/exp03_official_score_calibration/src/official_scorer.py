"""Strict thin wrapper around DACON's unmodified BARAM 2026 metric logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from official.dacon_baram_metric.metric import (
    CAPACITY_KWH,
    TARGET_COLS,
    metric as dacon_metric,
)


@dataclass(frozen=True)
class GroupScore:
    target: str
    capacity_kwh: float
    nmae: float
    one_minus_nmae: float
    ficr: float
    evaluated_samples: int
    total_samples: int
    evaluated_rate: float


@dataclass(frozen=True)
class OfficialScore:
    total_score: float
    one_minus_nmae: float
    ficr: float
    evaluated_samples: int
    total_samples: int
    evaluated_rate: float
    groups: tuple[GroupScore, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["groups"] = [asdict(group) for group in self.groups]
        return value


def _validate_wide(answer_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    missing_answer = [column for column in TARGET_COLS if column not in answer_df]
    missing_prediction = [column for column in TARGET_COLS if column not in pred_df]
    if missing_answer or missing_prediction:
        raise ValueError(
            f"official scorer columns missing: answer={missing_answer}, prediction={missing_prediction}"
        )
    if len(answer_df) != len(pred_df):
        raise ValueError("answer and prediction row counts differ")
    if len(answer_df) == 0:
        raise ValueError("official scorer received no rows")
    for name, frame in (("answer", answer_df), ("prediction", pred_df)):
        values = frame[TARGET_COLS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite target values")


def score_wide(answer_df: pd.DataFrame, pred_df: pd.DataFrame) -> OfficialScore:
    """Call the official function, then expose its exact group-level components."""
    _validate_wide(answer_df, pred_df)
    total_score, one_minus_nmae, ficr = dacon_metric(answer_df, pred_df)
    groups: list[GroupScore] = []
    for target in TARGET_COLS:
        capacity = float(CAPACITY_KWH[target])
        actual_all = answer_df[target].to_numpy(dtype=float)
        forecast_all = pred_df[target].to_numpy(dtype=float)
        valid = actual_all >= capacity * 0.10
        if not valid.any():
            raise ValueError(f"{target} has no samples in the official evaluation mask")
        actual = actual_all[valid]
        error_rate = np.abs(forecast_all[valid] - actual) / capacity
        nmae = float(np.mean(error_rate))
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0
        )
        max_settlement = float(np.sum(actual * 4.0))
        group_ficr = float(np.sum(actual * unit_price) / max_settlement)
        groups.append(
            GroupScore(
                target=target,
                capacity_kwh=capacity,
                nmae=nmae,
                one_minus_nmae=1.0 - nmae,
                ficr=group_ficr,
                evaluated_samples=int(valid.sum()),
                total_samples=int(len(valid)),
                evaluated_rate=float(valid.mean()),
            )
        )
    evaluated = sum(group.evaluated_samples for group in groups)
    total = sum(group.total_samples for group in groups)
    result = OfficialScore(
        total_score=float(total_score),
        one_minus_nmae=float(one_minus_nmae),
        ficr=float(ficr),
        evaluated_samples=evaluated,
        total_samples=total,
        evaluated_rate=float(evaluated / total),
        groups=tuple(groups),
    )
    expected = np.asarray(
        [
            0.5 * result.one_minus_nmae + 0.5 * result.ficr,
            1.0 - np.mean([group.nmae for group in result.groups]),
            np.mean([group.ficr for group in result.groups]),
        ]
    )
    observed = np.asarray([result.total_score, result.one_minus_nmae, result.ficr])
    if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
        raise AssertionError("wrapper components disagree with the official function")
    return result


def long_to_wide(
    predictions: pd.DataFrame,
    *,
    time_col: str = "forecast_kst_dtm",
    target_col: str = "target",
    answer_col: str = "y_true_kwh",
    prediction_col: str = "y_pred_kwh",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {time_col, target_col, answer_col, prediction_col}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"long prediction columns missing: {missing}")
    data = predictions[[time_col, target_col, answer_col, prediction_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col])
    if data.duplicated([time_col, target_col]).any():
        raise ValueError("duplicate timestamp/target rows in long predictions")
    unknown = sorted(set(data[target_col]) - set(TARGET_COLS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    answer = data.pivot(index=time_col, columns=target_col, values=answer_col).sort_index()
    forecast = data.pivot(index=time_col, columns=target_col, values=prediction_col).sort_index()
    missing_targets = [target for target in TARGET_COLS if target not in answer or target not in forecast]
    if missing_targets:
        raise ValueError(f"long predictions do not contain all groups: {missing_targets}")
    answer = answer[TARGET_COLS]
    forecast = forecast[TARGET_COLS]
    if answer.isna().any().any() or forecast.isna().any().any():
        raise ValueError("timestamp/target grid is incomplete")
    return answer.reset_index(drop=True), forecast.reset_index(drop=True)


def score_long(predictions: pd.DataFrame, **kwargs: Any) -> OfficialScore:
    answer, forecast = long_to_wide(predictions, **kwargs)
    return score_wide(answer, forecast)
