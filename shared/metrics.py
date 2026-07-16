# shared/metrics.py

import numpy as np
import pandas as pd
from shared.constants import CAPACITY_KWH  # 내 shared 내부 상수 참조

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def calculate_competition_metric(answer_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """대회 공식 평가 산식을 계산합니다."""
    group_nmae = []
    group_ficr = []

    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]

        # 실제 발전량이 설비용량의 10% 이상인 시간대만 평가
        valid = actual >= (capacity * 0.10)

        if not np.any(valid):
            group_nmae.append(1.0)
            group_ficr.append(0.0)
            continue

        actual_valid = actual[valid]
        forecast_valid = forecast[valid]

        # NMAE 계산
        error_rate = np.abs(forecast_valid - actual_valid) / capacity
        group_nmae.append(np.mean(error_rate))

        # FICR 계산
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )

        earned_settlement = np.sum(actual_valid * unit_price)
        max_settlement = np.sum(actual_valid * 4.0)

        if max_settlement == 0:
            group_ficr.append(0.0)
        else:
            group_ficr.append(earned_settlement / max_settlement)

    one_minus_nmae = 1.0 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return {
        "total_score": float(total_score),
        "one_minus_nmae": float(one_minus_nmae),
        "ficr": float(ficr)
    }