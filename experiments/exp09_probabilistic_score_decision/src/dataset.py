from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_NAMES = ("ldaps_dynamic", "gfs_dynamic", "engineered_common", "engineered_group", "crossfitted_hubwind")
HUB_FEATURES = ("predicted_hub_ws_median", "predicted_hub_ws_mean", "predicted_hub_ws_std",
                "predicted_hub_ws_iqr", "stage1_seed_std", "forecast_minus_predicted_hub_ws",
                "stage1_fallback_indicator")


def assert_input_contract(names=INPUT_NAMES) -> None:
    forbidden = ("scada_actual", "power_target", "target_lag")
    matches = [name for name in names if any(value in name.lower() for value in forbidden)]
    if matches:
        raise ValueError(f"forbidden probabilistic inputs: {matches}")


class QuantileDataset(Dataset):
    def __init__(self, hidden: np.ndarray, target=None, mask=None) -> None:
        assert_input_contract()
        self.hidden = torch.as_tensor(hidden, dtype=torch.float32)
        shape = hidden.shape[:-1]
        m = np.zeros(shape, bool) if mask is None else np.asarray(mask, bool)
        y = np.zeros(shape, np.float32) if target is None else np.asarray(target, np.float32).copy()
        y[~m] = 0
        self.target, self.mask = torch.from_numpy(y), torch.from_numpy(m)

    def __len__(self): return len(self.hidden)
    def __getitem__(self, index): return self.hidden[index], self.target[index], self.mask[index]
