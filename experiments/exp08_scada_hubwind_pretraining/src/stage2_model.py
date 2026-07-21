"""Exp04 power model augmented with cross-fitted predicted hub-wind features."""

from __future__ import annotations

import torch
from torch import nn

from experiments.exp04_raw_grid_spatiotemporal.src.models import RawGridSpatiotemporalModel

from .stage2_dataset import DISTRIBUTION_FEATURE_INDICES, EXPLICIT_FEATURE_INDICES


class HubWindPowerModel(RawGridSpatiotemporalModel):
    def __init__(self, *args, hub_feature_indices: tuple[int, ...] = (), retain_hub_head: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hub_feature_indices = tuple(int(value) for value in hub_feature_indices)
        self.retain_hub_head = bool(retain_hub_head)
        base_input = int(self.final_projection[0].in_features)
        hidden = int(self.final_projection[0].out_features)
        if self.hub_feature_indices:
            self.hub_encoder = nn.Sequential(
                nn.Linear(len(self.hub_feature_indices), 32), nn.LayerNorm(32), nn.GELU()
            )
            self.final_projection = nn.Sequential(
                nn.Linear(base_input + 32, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
        else:
            self.hub_encoder = None
        self.hub_retention_head = (
            nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 4))
            if self.retain_hub_head else None
        )

    def forward(
        self,
        ldaps_dynamic: torch.Tensor,
        gfs_dynamic: torch.Tensor,
        engineered_common: torch.Tensor | None = None,
        engineered_group: torch.Tensor | None = None,
        hub_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor | None]]:
        ldaps, ldaps_attention, ldaps_dispersion = self.ldaps_encoder(ldaps_dynamic, self.ldaps_static)
        gfs, gfs_attention, gfs_dispersion = self.gfs_encoder(gfs_dynamic, self.gfs_static)
        fused, gate = self.fusion(ldaps, gfs, ldaps_dispersion, gfs_dispersion)
        parts = [self.raw_projection(fused)]
        if self.use_engineered:
            if engineered_common is None or engineered_group is None:
                raise ValueError("Stage-2 hybrid model did not receive engineered context")
            common = self.common_encoder(engineered_common)[:, :, None, :].expand(-1, -1, 3, -1)
            groups = torch.stack([
                encoder(engineered_group[:, :, index, :dimension])
                for index, (encoder, dimension) in enumerate(zip(self.group_encoders, self.group_dims))
            ], dim=2)
            parts.extend([common, groups])
        parts.append(self.time_encoder(self._time_features(parts[0])))
        if self.hub_feature_indices:
            if hub_features is None:
                raise ValueError("explicit Stage-2 model did not receive predicted hub-wind features")
            selected = hub_features[..., list(self.hub_feature_indices)]
            parts.append(self.hub_encoder(selected))
        hidden = self.temporal(self.final_projection(torch.cat(parts, dim=-1)))
        hidden = self.transform_group_hidden(hidden)
        power = self.power_head(hidden).squeeze(-1)
        retention = self.hub_retention_head(hidden) if self.hub_retention_head is not None else None
        return power, retention, {
            "ldaps_attention": ldaps_attention,
            "gfs_attention": gfs_attention,
            "source_gate": gate,
        }


def feature_indices_for_variant(variant: str) -> tuple[int, ...]:
    if variant == "pretrained_encoder":
        return ()
    if variant == "explicit_hubwind":
        return EXPLICIT_FEATURE_INDICES
    if variant in {"distribution_hubwind", "joint_finetune"}:
        return DISTRIBUTION_FEATURE_INDICES
    raise ValueError(f"unknown Stage-2 variant: {variant}")


def build_stage2_model(
    config: dict,
    ldaps_dynamic_dim: int,
    gfs_dynamic_dim: int,
    ldaps_static,
    gfs_static,
    common_dim: int,
    group_dims: tuple[int, int, int],
) -> HubWindPowerModel:
    variant = config.get("stage2", {}).get("variant", "distribution_hubwind")
    return HubWindPowerModel(
        config,
        ldaps_dynamic_dim,
        gfs_dynamic_dim,
        ldaps_static,
        gfs_static,
        common_dim,
        group_dims,
        hub_feature_indices=feature_indices_for_variant(variant),
        retain_hub_head=variant == "joint_finetune",
    )
