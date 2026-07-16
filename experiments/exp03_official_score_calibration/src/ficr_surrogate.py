"""Losses derived directly from DACON BARAM 2026's published thresholds."""

from __future__ import annotations

import torch

from experiments.exp02_daily_tcn_scada_aux.src.losses import (
    group_balanced_masked_smooth_l1,
)


OFFICIAL_TARGET_FRACTION = 0.10
OFFICIAL_FULL_REWARD_THRESHOLD = 0.06
OFFICIAL_PARTIAL_REWARD_THRESHOLD = 0.08
OFFICIAL_PARTIAL_REWARD_FRACTION = 3.0 / 4.0


def official_power_mask(target_cf: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    return label_mask.bool() & (target_cf >= OFFICIAL_TARGET_FRACTION)


def _weighted_group_mean(
    values: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor | None = None
) -> torch.Tensor:
    group_values = []
    for group in range(values.shape[-1]):
        valid = mask[..., group]
        if not torch.any(valid):
            continue
        weights = torch.ones_like(values[..., group]) if sample_weight is None else sample_weight[..., group]
        weights = weights[valid]
        group_values.append((values[..., group][valid] * weights).sum() / weights.sum().clamp_min(1e-12))
    if not group_values:
        return values.sum() * 0.0
    return torch.stack(group_values).mean()


def official_masked_mae(
    prediction_cf: torch.Tensor,
    target_cf: torch.Tensor,
    label_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = official_power_mask(target_cf, label_mask)
    return _weighted_group_mean(torch.abs(prediction_cf - target_cf), mask, sample_weight)


def soft_ficr_reward(error_rate: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("soft FICR temperature must be positive")
    full_increment = 1.0 - OFFICIAL_PARTIAL_REWARD_FRACTION
    return (
        full_increment
        * torch.sigmoid((OFFICIAL_FULL_REWARD_THRESHOLD - error_rate) / temperature)
        + OFFICIAL_PARTIAL_REWARD_FRACTION
        * torch.sigmoid((OFFICIAL_PARTIAL_REWARD_THRESHOLD - error_rate) / temperature)
    )


def soft_ficr_loss(
    prediction_cf: torch.Tensor,
    target_cf: torch.Tensor,
    label_mask: torch.Tensor,
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = official_power_mask(target_cf, label_mask)
    reward = soft_ficr_reward(torch.abs(prediction_cf - target_cf), temperature)
    # Official FICR weights every hourly reward by actual generation.
    energy_weight = target_cf if sample_weight is None else target_cf * sample_weight
    mean_reward = _weighted_group_mean(reward, mask, energy_weight)
    return 1.0 - mean_reward


def total_official_loss(
    power_prediction: torch.Tensor,
    power_target: torch.Tensor,
    power_mask: torch.Tensor,
    aux_prediction: torch.Tensor | None,
    aux_target: torch.Tensor | None,
    aux_mask: torch.Tensor | None,
    *,
    aux_weight: float,
    lambda_ficr: float,
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    main = official_masked_mae(power_prediction, power_target, power_mask, sample_weight)
    ficr = (
        soft_ficr_loss(power_prediction, power_target, power_mask, temperature, sample_weight)
        if lambda_ficr > 0
        else main.detach() * 0.0
    )
    if aux_weight > 0 and aux_prediction is not None and aux_target is not None and aux_mask is not None:
        auxiliary = group_balanced_masked_smooth_l1(aux_prediction, aux_target, aux_mask)
    else:
        auxiliary = main.detach() * 0.0
    total = main + float(lambda_ficr) * ficr + float(aux_weight) * auxiliary
    return total, main, ficr, auxiliary
