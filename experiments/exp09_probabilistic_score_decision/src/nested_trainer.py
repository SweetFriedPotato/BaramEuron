from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.exp02_daily_tcn_scada_aux.src.trainer import seed_everything
from .dataset import QuantileDataset
from .quantile_loss import quantile_training_loss


@dataclass
class QuantileTrainingResult:
    prediction: np.ndarray
    best_epoch: int
    best_loss: float


def assert_inner_precedes_outer(inner_quarters: list[str], outer_quarter: str) -> None:
    import pandas as pd
    if not inner_quarters or max(pd.Period(q, freq="Q") for q in inner_quarters) >= pd.Period(outer_quarter, freq="Q"):
        raise ValueError("inner calibration/selection must strictly precede the outer quarter")


def train_quantile_head(model, train_hidden, train_y, train_mask, valid_hidden, valid_y, valid_mask,
                        seed: int, checkpoint: Path, epochs: int = 50, patience: int = 8) -> QuantileTrainingResult:
    seed_everything(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    loader = DataLoader(QuantileDataset(train_hidden, train_y, train_mask), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for hidden, target, mask in loader:
            hidden, target, mask = hidden.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True); prediction = model(hidden)
            total, _ = quantile_training_loss(prediction, target, mask); total.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            vp = model(torch.as_tensor(valid_hidden, dtype=torch.float32, device=device))
            loss = float(quantile_training_loss(vp, torch.as_tensor(valid_y, device=device),
                                                torch.as_tensor(valid_mask, device=device))[0].cpu())
        if loss < best - 1e-8: best, state, best_epoch, stale = loss, copy.deepcopy(model.state_dict()), epoch, 0
        else: stale += 1
        if stale >= patience: break
    if state is None: raise RuntimeError("quantile training produced no checkpoint")
    model.load_state_dict(state); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "seed": seed, "best_epoch": best_epoch, "best_loss": best}, checkpoint)
    model.eval()
    with torch.no_grad(): prediction = model(torch.as_tensor(valid_hidden, dtype=torch.float32, device=device)).cpu().numpy()
    return QuantileTrainingResult(prediction, best_epoch, best)
