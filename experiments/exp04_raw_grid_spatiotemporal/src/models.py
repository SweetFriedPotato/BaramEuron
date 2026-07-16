"""Raw-grid spatial-attention plus daily temporal model."""

from __future__ import annotations

import math

import torch
from torch import nn

from .source_fusion import ConcatenatedSourceFusion, LeadTimeGatedFusion
from .spatial_encoder import GroupQuerySpatialAttention
from .temporal_model import GroupTemporalTCN


def _context_encoder(input_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU())


class RawGridSpatiotemporalModel(nn.Module):
    def __init__(
        self,
        config: dict,
        ldaps_dynamic_dim: int,
        gfs_dynamic_dim: int,
        ldaps_static: torch.Tensor,
        gfs_static: torch.Tensor,
        common_dim: int = 0,
        group_dims: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()
        model = config["model"]
        token_dim = int(model.get("token_dim", 64))
        heads = int(model.get("attention_heads", 4))
        dropout = float(model.get("attention_dropout", 0.10))
        use_geo = bool(config.get("use_geo", False))
        self.use_engineered = bool(config.get("use_engineered", False))
        self.gated_fusion = bool(config.get("gated_fusion", False))
        self.group_dims = tuple(int(value) for value in group_dims)
        self.register_buffer("ldaps_static", torch.as_tensor(ldaps_static, dtype=torch.float32))
        self.register_buffer("gfs_static", torch.as_tensor(gfs_static, dtype=torch.float32))
        self.ldaps_encoder = GroupQuerySpatialAttention(
            ldaps_dynamic_dim, self.ldaps_static.shape[-1], token_dim, heads, dropout, use_geo
        )
        self.gfs_encoder = GroupQuerySpatialAttention(
            gfs_dynamic_dim, self.gfs_static.shape[-1], token_dim, heads, dropout, use_geo
        )
        self.fusion = LeadTimeGatedFusion(token_dim) if self.gated_fusion else ConcatenatedSourceFusion(token_dim)
        self.raw_projection = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.time_encoder = nn.Sequential(nn.Linear(4, 16), nn.GELU())
        if self.use_engineered:
            if common_dim <= 0 or min(self.group_dims) <= 0:
                raise ValueError("hybrid model requires common and group engineered features")
            self.common_encoder = _context_encoder(common_dim)
            self.group_encoders = nn.ModuleList([_context_encoder(value) for value in self.group_dims])
            final_input = 128 + 64 + 64 + 16
        else:
            self.common_encoder = None
            self.group_encoders = nn.ModuleList()
            final_input = 128 + 16
        hidden = int(model.get("hidden_channels", 128))
        self.final_projection = nn.Sequential(
            nn.Linear(final_input, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.temporal = GroupTemporalTCN(
            hidden, int(model.get("kernel_size", 3)), tuple(model.get("dilations", [1, 2, 4, 8])),
            float(model.get("temporal_dropout", 0.15)), bool(model.get("non_causal", True)),
        )
        self.power_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
        self.auxiliary_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def _time_features(reference: torch.Tensor) -> torch.Tensor:
        batch, steps, groups = reference.shape[:3]
        index = torch.arange(steps, device=reference.device, dtype=reference.dtype)
        lead = (12.0 + index) / 35.0
        hour = torch.remainder(index + 1.0, 24.0)
        values = torch.stack(
            [
                lead,
                lead.square(),
                torch.sin(2 * math.pi * hour / 24.0),
                torch.cos(2 * math.pi * hour / 24.0),
            ],
            dim=-1,
        )
        return values.view(1, steps, 1, 4).expand(batch, steps, groups, 4)

    def forward(
        self,
        ldaps_dynamic: torch.Tensor,
        gfs_dynamic: torch.Tensor,
        engineered_common: torch.Tensor | None = None,
        engineered_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        ldaps, ldaps_attention, ldaps_dispersion = self.ldaps_encoder(ldaps_dynamic, self.ldaps_static)
        gfs, gfs_attention, gfs_dispersion = self.gfs_encoder(gfs_dynamic, self.gfs_static)
        fused, gate = self.fusion(ldaps, gfs, ldaps_dispersion, gfs_dispersion)
        raw = self.raw_projection(fused)
        parts = [raw]
        if self.use_engineered:
            if engineered_common is None or engineered_group is None:
                raise ValueError("hybrid model did not receive engineered context")
            common = self.common_encoder(engineered_common)[:, :, None, :].expand(-1, -1, 3, -1)
            groups = torch.stack(
                [
                    encoder(engineered_group[:, :, index, :dimension])
                    for index, (encoder, dimension) in enumerate(zip(self.group_encoders, self.group_dims))
                ],
                dim=2,
            )
            parts.extend([common, groups])
        time = self.time_encoder(self._time_features(raw))
        parts.append(time)
        hidden = self.temporal(self.final_projection(torch.cat(parts, dim=-1)))
        power = self.power_head(hidden).squeeze(-1)
        auxiliary = self.auxiliary_head(hidden).squeeze(-1)
        diagnostics = {
            "ldaps_attention": ldaps_attention,
            "gfs_attention": gfs_attention,
            "source_gate": gate,
        }
        return power, auxiliary, diagnostics


def build_model(
    config: dict,
    ldaps_dynamic_dim: int,
    gfs_dynamic_dim: int,
    ldaps_static,
    gfs_static,
    common_dim: int = 0,
    group_dims: tuple[int, int, int] = (0, 0, 0),
) -> RawGridSpatiotemporalModel:
    return RawGridSpatiotemporalModel(
        config, ldaps_dynamic_dim, gfs_dynamic_dim, ldaps_static, gfs_static,
        common_dim, group_dims,
    )
