"""Group-query spatial token encoder with optional geographic attention bias."""

from __future__ import annotations

import math

import torch
from torch import nn

from .raw_grid_loader import STATIC_DISTANCE_INDEX, STATIC_HEIGHT_INDEX


class DynamicTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, token_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, token_dim), nn.LayerNorm(token_dim), nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class StaticTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, token_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, token_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Static geometry is deterministic, so normalizing over the complete
        # source/group grid metadata is leakage-free. It also keeps metre-scale
        # terrain and degree-scale coordinates from dominating initialization.
        mean = values.mean(dim=(0, 1), keepdim=True)
        std = values.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return self.network((values - mean) / std)


class GroupQuerySpatialAttention(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        token_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.10,
        use_geo: bool = True,
    ) -> None:
        super().__init__()
        if token_dim % heads:
            raise ValueError("token dimension must be divisible by attention heads")
        self.token_dim = int(token_dim)
        self.heads = int(heads)
        self.head_dim = token_dim // heads
        self.use_geo = bool(use_geo)
        self.dynamic_encoder = DynamicTokenEncoder(dynamic_dim, token_dim)
        self.static_encoder = StaticTokenEncoder(static_dim, token_dim)
        self.source_embedding = nn.Parameter(torch.zeros(token_dim))
        self.group_query = nn.Parameter(torch.empty(3, token_dim))
        self.query_projection = nn.Linear(token_dim, token_dim, bias=False)
        self.key_projection = nn.Linear(token_dim, token_dim, bias=False)
        self.value_projection = nn.Linear(token_dim, token_dim, bias=False)
        self.output_projection = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        self.raw_beta_distance = nn.Parameter(torch.full((3, heads), -2.0))
        self.raw_beta_height = nn.Parameter(torch.full((3, heads), -2.0))
        nn.init.normal_(self.group_query, std=0.02)

    @property
    def beta_distance(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_beta_distance)

    @property
    def beta_height(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_beta_height)

    def forward(
        self, dynamic: torch.Tensor, group_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dynamic.ndim != 4 or group_static.ndim != 3 or group_static.shape[0] != 3:
            raise ValueError("expected dynamic [B,T,G,C] and static [3,G,S]")
        batch, steps, grids, _ = dynamic.shape
        if group_static.shape[1] != grids:
            raise ValueError("dynamic/static grid count differs")
        dynamic_token = self.dynamic_encoder(dynamic)[:, :, None, :, :]
        if self.use_geo:
            static_token = self.static_encoder(group_static)[None, None, :, :, :]
        else:
            static_token = torch.zeros(
                (1, 1, 3, grids, self.token_dim), device=dynamic.device, dtype=dynamic.dtype
            )
        token = dynamic_token + static_token + self.source_embedding.view(1, 1, 1, 1, -1)
        query = self.query_projection(self.group_query).reshape(3, self.heads, self.head_dim)
        key = self.key_projection(token).reshape(batch, steps, 3, grids, self.heads, self.head_dim)
        value = self.value_projection(token).reshape(batch, steps, 3, grids, self.heads, self.head_dim)
        logits = torch.einsum("phd,btpghd->btphg", query, key) / math.sqrt(self.head_dim)
        if self.use_geo:
            distance = torch.log1p(group_static[..., STATIC_DISTANCE_INDEX]).clamp_min(0.0)
            height = group_static[..., STATIC_HEIGHT_INDEX].abs()
            penalty = (
                self.beta_distance[:, :, None] * distance[:, None, :]
                + self.beta_height[:, :, None] * height[:, None, :]
            )
            logits = logits - penalty[None, None, :, :, :]
        weight = torch.softmax(logits, dim=-1)
        weight = self.dropout(weight)
        attended = torch.einsum("btphg,btpghd->btphd", weight, value)
        attended = attended.reshape(batch, steps, 3, self.token_dim)
        output = self.output_projection(attended)
        mean_attention = weight.mean(dim=3)
        safe = mean_attention.clamp_min(1e-8)
        dispersion = -(safe * safe.log()).sum(dim=-1) / math.log(grids)
        return output, mean_attention, dispersion
