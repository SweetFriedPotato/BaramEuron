"""Shared non-causal daily TCN over group-specific representations."""

from __future__ import annotations

import torch
from torch import nn

from experiments.exp02_daily_tcn_scada_aux.src.models import TemporalResidualBlock


class GroupTemporalTCN(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.15,
        non_causal: bool = True,
    ) -> None:
        super().__init__()
        self.group_embedding = nn.Parameter(torch.empty(3, hidden_channels))
        nn.init.normal_(self.group_embedding, std=0.02)
        self.temporal = nn.Sequential(
            *[
                TemporalResidualBlock(hidden_channels, kernel_size, dilation, dropout, non_causal)
                for dilation in dilations
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[2] != 3:
            raise ValueError("temporal input must be [B,T,3,H]")
        batch, steps, groups, hidden = values.shape
        values = values + self.group_embedding.view(1, 1, 3, hidden)
        flat = values.permute(0, 2, 1, 3).reshape(batch * groups, steps, hidden)
        flat = self.temporal(flat.transpose(1, 2)).transpose(1, 2)
        return flat.reshape(batch, groups, steps, hidden).permute(0, 2, 1, 3)
