from __future__ import annotations

import numpy as np

from .expected_official_score import shrink_decision

ALPHAS = (0.25, 0.50, 0.75, 1.00)


def shrinkage_candidates(decision: np.ndarray, reference: np.ndarray) -> dict[float, np.ndarray]:
    return {alpha: shrink_decision(decision, reference, alpha) for alpha in ALPHAS}
