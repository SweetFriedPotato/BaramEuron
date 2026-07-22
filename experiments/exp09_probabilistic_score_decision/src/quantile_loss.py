from __future__ import annotations

import torch

from . import QUANTILE_LEVELS


def group_balanced_pinball(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape[:-1] != target.shape or target.shape != mask.shape:
        raise ValueError("pinball prediction/target/mask schema differs")
    levels = prediction.new_tensor(QUANTILE_LEVELS)
    error = target[..., None] - prediction
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    groups = [loss[..., g, :][mask[..., g]].mean() for g in range(3) if torch.any(mask[..., g])]
    return torch.stack(groups).mean() if groups else prediction.sum() * 0.0


def approximate_crps(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return 2.0 * group_balanced_pinball(prediction, target, mask)


def quantile_training_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    pinball = group_balanced_pinball(prediction, target, mask)
    crps = 2.0 * pinball
    official = mask & (target >= 0.10)
    med = prediction[..., 5]
    group_losses = [torch.abs(med[..., g] - target[..., g])[official[..., g]].mean()
                    for g in range(3) if torch.any(official[..., g])]
    q50 = torch.stack(group_losses).mean() if group_losses else prediction.sum() * 0.0
    total = pinball + 0.10 * crps + 0.10 * q50
    return total, {"pinball": pinball, "approximate_crps": crps, "q50_nmae": q50}
