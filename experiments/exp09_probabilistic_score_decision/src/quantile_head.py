from __future__ import annotations

import torch
from torch import nn

from . import QUANTILE_LEVELS


class MonotoneQuantileHead(nn.Module):
    """Median plus positive adjacent increments, output ``[..., 11]``."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 11))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(hidden)
        median = raw[..., 5:6]
        lower_steps = nn.functional.softplus(raw[..., :5])
        upper_steps = nn.functional.softplus(raw[..., 6:])
        lower = median - torch.flip(torch.cumsum(torch.flip(lower_steps, dims=(-1,)), dim=-1), dims=(-1,))
        upper = median + torch.cumsum(upper_steps, dim=-1)
        return torch.cat([lower, median, upper], dim=-1)


def assert_monotone(quantiles: torch.Tensor, atol: float = 1e-7) -> None:
    if quantiles.shape[-1] != len(QUANTILE_LEVELS):
        raise ValueError("quantile axis must contain 11 levels")
    if torch.any(quantiles[..., 1:] + atol < quantiles[..., :-1]):
        raise ValueError("quantile crossing detected")
