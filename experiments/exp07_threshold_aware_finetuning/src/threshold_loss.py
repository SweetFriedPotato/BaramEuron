"""Differentiable approximation to the exact 6%/8% official reward tiers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThresholdLossParts:
    total: torch.Tensor
    base_nmae: torch.Tensor
    soft_ficr: torch.Tensor
    boundary: torch.Tensor
    tau: float


def normalized_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    capacity: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Absolute capacity-normalized error; capacity=1 for existing CF targets."""
    capacity_tensor = torch.as_tensor(capacity, dtype=prediction.dtype, device=prediction.device)
    if torch.any(capacity_tensor <= 0):
        raise ValueError("capacity must be positive")
    return torch.abs(prediction - target) / capacity_tensor


def official_mask(
    target: torch.Tensor,
    label_mask: torch.Tensor,
    capacity: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    capacity_tensor = torch.as_tensor(capacity, dtype=target.dtype, device=target.device)
    return label_mask.bool() & torch.isfinite(target) & (target >= 0.10 * capacity_tensor)


def group_balanced_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average within each available last-axis group, then average groups."""
    if values.shape != mask.shape:
        raise ValueError("values and mask shapes differ")
    if values.ndim < 1:
        raise ValueError("group-balanced tensors need a group axis")
    means = []
    for group in range(values.shape[-1]):
        valid = mask[..., group].bool()
        if torch.any(valid):
            means.append(values[..., group][valid].mean())
    if not means:
        raise ValueError("loss contains no officially evaluated labels")
    return torch.stack(means).mean()


def normalized_soft_reward(error: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return (
        3.0 * torch.sigmoid((0.08 - error) / tau)
        + torch.sigmoid((0.06 - error) / tau)
    ) / 4.0


def boundary_weight(error: torch.Tensor, sigma: float) -> torch.Tensor:
    """Symmetric 6%/8% Gaussian emphasis with detached routing weights."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    detached = error.detach()
    w6 = torch.exp(-0.5 * ((detached - 0.06) / sigma).square())
    w8 = torch.exp(-0.5 * ((detached - 0.08) / sigma).square())
    return w6 + w8


def scheduled_tau(
    epoch: int,
    max_epochs: int,
    start: float,
    end: float,
    schedule: str = "fixed",
) -> float:
    if max_epochs < 1 or epoch < 1 or epoch > max_epochs:
        raise ValueError("epoch must be within [1, max_epochs]")
    if start <= 0 or end <= 0:
        raise ValueError("temperature endpoints must be positive")
    if schedule == "fixed":
        return float(start)
    progress = 0.0 if max_epochs == 1 else (epoch - 1) / (max_epochs - 1)
    if schedule == "linear":
        return float(start + progress * (end - start))
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(end + (start - end) * weight)
    raise ValueError(f"unknown tau schedule: {schedule}")


def threshold_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    capacity: torch.Tensor | float = 1.0,
    tau: float = 0.006,
    sigma: float = 0.006,
    soft_ficr_weight: float = 0.20,
    lambda_boundary: float = 0.05,
) -> ThresholdLossParts:
    error = normalized_error(prediction, target, capacity)
    mask = official_mask(target, label_mask, capacity)
    base = group_balanced_mean(error, mask)
    soft = 1.0 - group_balanced_mean(normalized_soft_reward(error, tau), mask)
    boundary = group_balanced_mean(boundary_weight(error, sigma) * error, mask)
    total = base + float(soft_ficr_weight) * soft + float(lambda_boundary) * boundary
    return ThresholdLossParts(total, base, soft, boundary, float(tau))

