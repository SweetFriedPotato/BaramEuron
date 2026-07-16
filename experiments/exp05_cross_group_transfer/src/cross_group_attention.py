"""Minimal target-free cross-group attention extension of the Exp04 model."""

from __future__ import annotations

import torch
from torch import nn

from experiments.exp04_raw_grid_spatiotemporal.src.models import RawGridSpatiotemporalModel


class CrossGroupAttentionBlock(nn.Module):
    """Apply one shared self-attention layer over the three groups at each hour."""

    def __init__(self, hidden_dim: int, heads: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H], got {tuple(hidden.shape)}")
        batch, steps, groups, width = hidden.shape
        values = hidden.reshape(batch * steps, groups, width)
        normalized = self.norm(values)
        attended, weights = self.attention(
            normalized, normalized, normalized, need_weights=True, average_attn_weights=False
        )
        self.last_attention = weights.detach()
        return (values + self.dropout(attended)).reshape(batch, steps, groups, width)


class CrossGroupRawGridModel(RawGridSpatiotemporalModel):
    """Exp04 raw_hybrid_gated with exactly one cross-group attention block."""

    def __init__(self, *args, cross_group_config: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        settings = cross_group_config or {}
        if int(settings.get("layers", 1)) != 1:
            raise ValueError("Exp05 permits exactly one cross-group attention layer")
        hidden_dim = int(self.power_head[0].in_features)
        self.cross_group_attention = CrossGroupAttentionBlock(
            hidden_dim,
            heads=int(settings.get("heads", 2)),
            dropout=float(settings.get("dropout", 0.10)),
        )

    def transform_group_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.cross_group_attention(hidden)

    def forward(self, *args, **kwargs):
        power, auxiliary, diagnostics = super().forward(*args, **kwargs)
        attention = self.cross_group_attention.last_attention
        if attention is not None:
            diagnostics["cross_group_attention"] = attention.reshape(
                power.shape[0], power.shape[1], attention.shape[1], 3, 3
            )
        return power, auxiliary, diagnostics


def build_cross_group_model(
    base_config: dict,
    cross_group_config: dict,
    ldaps_dynamic_dim: int,
    gfs_dynamic_dim: int,
    ldaps_static,
    gfs_static,
    common_dim: int,
    group_dims: tuple[int, int, int],
) -> CrossGroupRawGridModel:
    return CrossGroupRawGridModel(
        base_config,
        ldaps_dynamic_dim,
        gfs_dynamic_dim,
        ldaps_static,
        gfs_static,
        common_dim,
        group_dims,
        cross_group_config=cross_group_config,
    )
