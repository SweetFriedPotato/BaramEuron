"""Pointwise MLP and non-causal daily TCN models."""

from __future__ import annotations

import torch
from torch import nn


class PointwiseMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, int] = (256, 128),
        dropout: tuple[float, float] = (0.15, 0.10),
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout[1]),
            nn.Linear(hidden_dims[1], 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        batch, steps, features = x.shape
        power = self.network(x.reshape(batch * steps, features)).reshape(batch, steps, 3)
        return power, None


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, non_causal: bool) -> None:
        super().__init__()
        self.non_causal = non_causal
        self.pad = dilation * (kernel_size - 1) // 2 if non_causal else dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_causal:
            return nn.functional.pad(x, (self.pad, self.pad))
        return nn.functional.pad(x, (self.pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.conv1(self._pad(x))
        value = self.dropout(self.activation(self.norm1(value)))
        value = self.conv2(self._pad(value))
        value = self.dropout(self.activation(self.norm2(value)))
        return residual + value


class _GroupHeads(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(channels, 64), nn.GELU(), nn.Linear(64, 1)) for _ in range(3)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


class DailyTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 128,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.15,
        non_causal: bool = True,
        auxiliary_heads: bool = False,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.channel_projection = (
            nn.Identity() if projection_dim == hidden_channels else nn.Linear(projection_dim, hidden_channels)
        )
        self.temporal = nn.Sequential(
            *[
                TemporalResidualBlock(hidden_channels, kernel_size, dilation, dropout, non_causal)
                for dilation in dilations
            ]
        )
        self.power_heads = _GroupHeads(hidden_channels)
        self.aux_heads = _GroupHeads(hidden_channels) if auxiliary_heads else None
        self.non_causal = bool(non_causal)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.channel_projection(self.input_projection(x))
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        power = self.power_heads(hidden)
        auxiliary = None if self.aux_heads is None else self.aux_heads(hidden)
        return power, auxiliary


def build_model(config: dict, feature_dim: int) -> nn.Module:
    if config["model_type"] == "mlp":
        model = config["model"]
        return PointwiseMLP(
            feature_dim,
            hidden_dims=tuple(model.get("hidden_dims", [256, 128])),
            dropout=tuple(model.get("dropout", [0.15, 0.10])),
        )
    model = config["model"]
    return DailyTCN(
        feature_dim,
        projection_dim=int(model.get("projection_dim", 128)),
        hidden_channels=int(model.get("hidden_channels", 128)),
        kernel_size=int(model.get("kernel_size", 3)),
        dilations=tuple(model.get("dilations", [1, 2, 4, 8])),
        dropout=float(model.get("dropout", 0.15)),
        non_causal=bool(model.get("non_causal", True)),
        auxiliary_heads=float(config.get("aux_weight", 0.0)) > 0,
    )
