from __future__ import annotations

from torch import nn
from .quantile_head import MonotoneQuantileHead


class HiddenQuantileModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__(); self.head = MonotoneQuantileHead(hidden_dim)
    def forward(self, hidden): return self.head(hidden)
