"""Verbatim code cell from DACON codeshare 14035 (metric.ipynb)."""

import numpy as np

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


def metric(answer_df, pred_df):
    group_nmae = []
    group_ficr = []

    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]

        # 실제 발전량이 설비용량의 10% 이상인 시간대만 평가
        valid = actual >= capacity * 0.10

        actual = actual[valid]
        forecast = forecast[valid]

        # NMAE 계산
        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(np.mean(error_rate))

        # FICR 계산
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )

        earned_settlement = np.sum(actual * unit_price)
        max_settlement = np.sum(actual * 4.0)

        group_ficr.append(earned_settlement / max_settlement)

    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)

    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return total_score, one_minus_nmae, ficr
