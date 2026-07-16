"""LDAPS/GFS fusion blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConcatenatedSourceFusion(nn.Module):
    output_dim = 128

    def __init__(self, source_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(source_dim * 2, 128), nn.LayerNorm(128), nn.GELU()
        )

    def forward(
        self,
        ldaps: torch.Tensor,
        gfs: torch.Tensor,
        ldaps_dispersion: torch.Tensor,
        gfs_dispersion: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return self.network(torch.cat([ldaps, gfs], dim=-1)), None


class LeadTimeGatedFusion(nn.Module):
    output_dim = 64

    def __init__(self, source_dim: int = 64) -> None:
        super().__init__()
        self.ldaps_projection = nn.Linear(source_dim, 64)
        self.gfs_projection = nn.Linear(source_dim, 64)
        self.gate = nn.Sequential(nn.Linear(64 * 2 + 5, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def _time_features(reference: torch.Tensor) -> torch.Tensor:
        batch, steps, groups = reference.shape[:3]
        index = torch.arange(steps, device=reference.device, dtype=reference.dtype)
        lead = (12.0 + index) / 35.0
        hour = torch.remainder(index + 1.0, 24.0)
        features = torch.stack(
            [lead, torch.sin(2 * math.pi * hour / 24.0), torch.cos(2 * math.pi * hour / 24.0)],
            dim=-1,
        )
        return features.view(1, steps, 1, 3).expand(batch, steps, groups, 3)

    def forward(
        self,
        ldaps: torch.Tensor,
        gfs: torch.Tensor,
        ldaps_dispersion: torch.Tensor,
        gfs_dispersion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.ldaps_projection(ldaps)
        right = self.gfs_projection(gfs)
        gate_input = torch.cat(
            [left, right, self._time_features(left), ldaps_dispersion[..., None], gfs_dispersion[..., None]],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * left + (1.0 - gate) * right, gate
