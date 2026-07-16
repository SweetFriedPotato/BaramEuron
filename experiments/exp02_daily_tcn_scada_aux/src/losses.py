"""Group-balanced masked power and auxiliary losses."""

from __future__ import annotations

import torch
from torch import nn


def _group_balanced(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for group in range(values.shape[-1]):
        valid = mask[..., group]
        if torch.any(valid):
            losses.append(values[..., group][valid].mean())
    if not losses:
        return values.sum() * 0.0
    return torch.stack(losses).mean()


def group_balanced_masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return _group_balanced(torch.abs(prediction - target), mask.bool())


def group_balanced_masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    values = nn.functional.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return _group_balanced(values, mask.bool())


def total_multitask_loss(
    power_prediction: torch.Tensor,
    power_target: torch.Tensor,
    power_mask: torch.Tensor,
    aux_prediction: torch.Tensor | None,
    aux_target: torch.Tensor | None,
    aux_mask: torch.Tensor | None,
    aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    main = group_balanced_masked_l1(power_prediction, power_target, power_mask)
    if aux_weight <= 0 or aux_prediction is None or aux_target is None or aux_mask is None:
        aux = main.detach() * 0.0
        return main, main, aux
    aux = group_balanced_masked_smooth_l1(aux_prediction, aux_target, aux_mask)
    return main + float(aux_weight) * aux, main, aux
