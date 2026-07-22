from __future__ import annotations

from experiments.exp08_scada_hubwind_pretraining.src.stage2_model import HubWindPowerModel
from .quantile_head import MonotoneQuantileHead


VARIANT_INDICES = {"q_a_exp04": (), "q_b_hubwind": (0, 1), "q_c_calibrated": (0, 1, 2, 3, 4)}


class QuantilePowerModel(HubWindPowerModel):
    def __init__(self, *args, variant: str, **kwargs):
        if variant not in VARIANT_INDICES: raise ValueError(f"unknown quantile variant: {variant}")
        super().__init__(*args, hub_feature_indices=VARIANT_INDICES[variant], **kwargs)
        hidden = int(self.power_head[0].in_features)
        self.power_head = MonotoneQuantileHead(hidden)


def build_quantile_power_model(config, data, variant: str):
    return QuantilePowerModel(
        config, data.train_inputs.ldaps.shape[-1], data.train_inputs.gfs.shape[-1],
        data.ldaps_static, data.gfs_static, data.common_dim, data.group_dims, variant=variant,
    )
