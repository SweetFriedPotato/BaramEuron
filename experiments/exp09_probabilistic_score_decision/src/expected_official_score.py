from __future__ import annotations

import numpy as np


def expected_components(samples: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(samples, dtype=float)
    p = np.asarray(candidates, dtype=float)
    valid = y >= 0.10
    if not valid.any():
        raise ValueError("conditional distribution has no officially evaluated samples")
    actual = np.sort(y[valid])
    prefix = np.r_[0.0, np.cumsum(actual)]
    count = len(actual)
    split = np.searchsorted(actual, p, side="right")
    left_sum, total = prefix[split], prefix[-1]
    absolute_sum = p * split - left_sum + (total - left_sum) - p * (count - split)
    one_minus_nmae = 1.0 - absolute_sum / count

    def energy_between(radius: float) -> np.ndarray:
        lo = np.searchsorted(actual, p - radius, side="left")
        hi = np.searchsorted(actual, p + radius, side="right")
        return prefix[hi] - prefix[lo]

    within_6 = energy_between(0.06)
    within_8 = energy_between(0.08)
    ficr = (4.0 * within_6 + 3.0 * (within_8 - within_6)) / (4.0 * total)
    return one_minus_nmae, ficr, 0.5 * (one_minus_nmae + ficr)


def decision_candidates(quantiles: np.ndarray, exp04_prediction: float, step: float = 0.0025) -> np.ndarray:
    q = np.asarray(quantiles, dtype=float)
    low, high = q[0] - step, q[-1] + step
    grid = np.arange(np.floor(low / step) * step, np.ceil(high / step) * step + step / 2, step)
    return np.unique(np.r_[grid, q[5], float(exp04_prediction)])


def score_optimal_decision(samples: np.ndarray, quantiles: np.ndarray, exp04_prediction: float) -> dict:
    candidates = decision_candidates(quantiles, exp04_prediction)
    one_minus, ficr, score = expected_components(samples, candidates)
    best = int(np.argmax(score))
    return {"prediction": float(candidates[best]), "expected_one_minus_nmae": float(one_minus[best]),
            "expected_ficr": float(ficr[best]), "expected_score": float(score[best]),
            "candidates": candidates}


def shrink_decision(decision: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    if alpha not in {0.25, 0.50, 0.75, 1.00}:
        raise ValueError("alpha must be selected from the nested contract")
    return float(alpha) * np.asarray(decision) + (1.0 - float(alpha)) * np.asarray(reference)
