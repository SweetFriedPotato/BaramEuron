"""Exp04 raw-hybrid-gated encoder with a four-target hub-wind head."""

from __future__ import annotations

import torch
from torch import nn

from experiments.exp04_raw_grid_spatiotemporal.src.models import RawGridSpatiotemporalModel


TARGET_ORDER = ("median", "mean", "log1p_std", "log1p_iqr")
TARGET_LOSS_WEIGHTS = (1.0, 0.5, 0.25, 0.25)


class HubWindDistributionModel(RawGridSpatiotemporalModel):
    """Produces standardized targets with shape ``[B,24,3,4]``."""

    def __init__(self, *args, target_count: int = 4, **kwargs) -> None:
        if target_count not in {1, 2, 4}:
            raise ValueError("Stage-1 ablations support 1, 2, or 4 targets")
        super().__init__(*args, **kwargs)
        hidden = int(self.power_head[0].in_features)
        self.target_count = int(target_count)
        self.power_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, target_count))

    @property
    def distribution_head(self) -> nn.Module:
        return self.power_head

    def forward(self, *args, **kwargs):
        distribution, auxiliary, diagnostics = super().forward(*args, **kwargs)
        if self.target_count == 1 and distribution.ndim == 3:
            distribution = distribution.unsqueeze(-1)
        if distribution.ndim != 4 or distribution.shape[2] != 3:
            raise ValueError("Stage-1 output must be [B,T,3,K]")
        return distribution, auxiliary, diagnostics

    def initialize_median_from_auxiliary_head(self) -> None:
        """S1-D: seed the median row from Exp04's SCADA auxiliary head."""
        source_first, source_last = self.auxiliary_head[0], self.auxiliary_head[-1]
        target_first, target_last = self.power_head[0], self.power_head[-1]
        with torch.no_grad():
            target_first.weight.copy_(source_first.weight)
            target_first.bias.copy_(source_first.bias)
            target_last.weight[0].copy_(source_last.weight[0])
            target_last.bias[0].copy_(source_last.bias[0])


def build_stage1_model(
    config: dict,
    ldaps_dynamic_dim: int,
    gfs_dynamic_dim: int,
    ldaps_static,
    gfs_static,
    common_dim: int,
    group_dims: tuple[int, int, int],
) -> HubWindDistributionModel:
    model = HubWindDistributionModel(
        config,
        ldaps_dynamic_dim,
        gfs_dynamic_dim,
        ldaps_static,
        gfs_static,
        common_dim,
        group_dims,
        target_count=int(config.get("stage1", {}).get("target_count", 4)),
    )
    if config.get("stage1", {}).get("initialize_from_auxiliary", False):
        model.initialize_median_from_auxiliary_head()
    return model


def group_balanced_distribution_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: tuple[float, ...] = TARGET_LOSS_WEIGHTS,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Stage-1 prediction/target/mask shapes differ")
    active = min(prediction.shape[-1], len(weights))
    losses = []
    for target_index in range(active):
        per_group = []
        for group in range(3):
            valid = mask[..., group, target_index].bool()
            if torch.any(valid):
                per_group.append(nn.functional.smooth_l1_loss(
                    prediction[..., group, target_index][valid],
                    target[..., group, target_index][valid],
                ))
        if per_group:
            losses.append(float(weights[target_index]) * torch.stack(per_group).mean())
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).sum()
