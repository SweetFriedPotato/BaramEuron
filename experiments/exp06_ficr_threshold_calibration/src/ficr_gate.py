"""Small target-free Exp03/raw blend gates trained on rolling OOF only."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from baram.constants import TARGETS
from experiments.exp05_cross_group_transfer.src.nested_rolling import ORDERED_QUARTERS, assert_nested_order
from experiments.exp05_cross_group_transfer.src.oof_contract import score_prediction


GATE_FEATURES = [
    "group_1", "group_2", "group_3", "exp03_cf", "raw_cf", "model_disagreement_cf",
    "lead_normalized", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "raw_source_gate", "raw_source_gate_available", "raw_seed_std_cf", "raw_seed_std_available",
]


def build_gate_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    for group_id in (1, 2, 3):
        out[f"group_{group_id}"] = out["group_id"].eq(group_id).astype(float)
    out["exp03_cf"] = out["exp03_prediction"] / out["capacity_kwh"]
    out["raw_cf"] = out["raw_prediction"] / out["capacity_kwh"]
    out["model_disagreement_cf"] = out["raw_cf"].sub(out["exp03_cf"]).abs()
    out["lead_normalized"] = (out["lead_time_h"] - 12.0) / 23.0
    hour = pd.to_datetime(out["forecast_kst_dtm"]).dt.hour.to_numpy(dtype=float)
    month = pd.to_datetime(out["forecast_kst_dtm"]).dt.month.to_numpy(dtype=float)
    out["hour_sin"], out["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    out["month_sin"], out["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    # Per-quarter gates/std were not persisted by Exp04. Neutral values preserve schema without leakage.
    out["raw_source_gate"] = 0.5; out["raw_source_gate_available"] = 0.0
    out["raw_seed_std_cf"] = 0.0; out["raw_seed_std_available"] = 0.0
    if not np.isfinite(out[GATE_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("gate features contain NaN/inf")
    forbidden = [name for name in GATE_FEATURES if "target" in name.lower() or "scada" in name.lower() or "lag" in name.lower()]
    if forbidden:
        raise ValueError(f"forbidden gate inputs: {forbidden}")
    return out, list(GATE_FEATURES)


class BlendGate(nn.Module):
    def __init__(self, input_dim: int, kind: str = "linear", hidden: int = 8) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif kind == "mlp":
            self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        else:
            raise ValueError(kind)
        final = self.network if isinstance(self.network, nn.Linear) else self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(np.log(0.4 / 0.6)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(values)).squeeze(-1)


@dataclass
class FittedGate:
    kind: str
    network: BlendGate
    mean: np.ndarray
    scale: np.ndarray
    feature_columns: list[str]
    lambda_prior: float


def _temporal_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    indexed = frame.reset_index(drop=True).reset_index(names="position")
    pairs_left, pairs_right = [], []
    for _, part in indexed.sort_values(["target", "issue_kst_dtm", "forecast_kst_dtm"]).groupby(
        ["target", "issue_kst_dtm"], sort=False
    ):
        if len(part) < 2:
            continue
        pairs_left.extend(part["position"].to_numpy()[:-1]); pairs_right.extend(part["position"].to_numpy()[1:])
    return np.asarray(pairs_left, dtype=int), np.asarray(pairs_right, dtype=int)


def fit_gate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    lambda_prior: float,
    kind: str = "linear",
    seed: int = 42,
) -> FittedGate:
    torch.manual_seed(seed); np.random.seed(seed)
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0); scale = values.std(axis=0); scale[scale < 1e-6] = 1.0
    x = torch.from_numpy((values-mean)/scale)
    target_cf = torch.from_numpy((frame["y_true_kwh"]/frame["capacity_kwh"]).to_numpy(dtype=np.float32))
    exp_cf = torch.from_numpy((frame["exp03_prediction"]/frame["capacity_kwh"]).to_numpy(dtype=np.float32))
    raw_cf = torch.from_numpy((frame["raw_prediction"]/frame["capacity_kwh"]).to_numpy(dtype=np.float32))
    groups = torch.from_numpy(frame["group_id"].to_numpy(dtype=np.int64)-1)
    official = target_cf >= 0.10
    left, right = _temporal_pairs(frame)
    left_t, right_t = torch.from_numpy(left), torch.from_numpy(right)
    model = BlendGate(len(feature_columns), kind, int(config.get("hidden", 8)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    for _ in range(int(config["epochs"])):
        optimizer.zero_grad(); weight = model(x)
        prediction = exp_cf + weight * (raw_cf-exp_cf)
        error = torch.abs(prediction-target_cf)
        nmae_parts, ficr_parts, group_weights = [], [], []
        for group in range(3):
            mask = official & groups.eq(group)
            if not torch.any(mask):
                continue
            nmae_parts.append(error[mask].mean())
            soft_reward = 0.25*torch.sigmoid((0.06-error[mask])/float(config["temperature"])) \
                + 0.75*torch.sigmoid((0.08-error[mask])/float(config["temperature"]))
            ficr_parts.append(1.0-(soft_reward*target_cf[mask]).sum()/target_cf[mask].sum().clamp_min(1e-8))
            group_weights.append(weight[mask].mean())
        main = torch.stack(nmae_parts).mean() + float(config["lambda_ficr"])*torch.stack(ficr_parts).mean()
        prior = (weight-0.4).square().mean()
        spread = torch.stack(group_weights).var(unbiased=False) if len(group_weights) > 1 else prior*0
        temporal = (weight[left_t]-weight[right_t]).square().mean() if len(left) else prior*0
        l2 = sum(parameter.square().mean() for parameter in model.parameters())
        loss = main + float(lambda_prior)*(prior+spread) \
            + float(config["temporal_smoothness"])*temporal + float(config["l2"])*l2
        loss.backward(); optimizer.step()
    return FittedGate(kind, model.eval(), mean, scale, feature_columns, float(lambda_prior))


def apply_gate(model: FittedGate, frame: pd.DataFrame, output_column: str = "gate_prediction") -> pd.DataFrame:
    out = frame.copy()
    values = out[model.feature_columns].to_numpy(dtype=np.float32)
    with torch.no_grad():
        weight = model.network(torch.from_numpy((values-model.mean)/model.scale)).numpy()
    out["gate_raw_weight"] = weight
    out[output_column] = out["exp03_prediction"] + weight*(out["raw_prediction"]-out["exp03_prediction"])
    return out


def nested_gate_selection(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    kind: str = "linear",
) -> tuple[pd.DataFrame, pd.DataFrame, FittedGate]:
    quarters = [quarter for quarter in ORDERED_QUARTERS if quarter in set(frame["quarter"])]
    parts, rows, selected = [], [], []
    for outer_index, quarter in enumerate(quarters):
        evaluation = frame.loc[frame["quarter"].eq(quarter)].copy()
        if outer_index < 2:
            evaluation["gate_raw_weight"] = float(config["fallback_weight"])
            evaluation["gate_prediction"] = evaluation["global_blend_prediction"]
            rows.append({"evaluation_quarter": quarter, "status": "fallback_insufficient_history",
                         "lambda_prior": np.nan, "mean_raw_weight": config["fallback_weight"]})
            parts.append(evaluation); continue
        fit_quarters = quarters[:outer_index]; assert_nested_order(fit_quarters, quarter)
        inner_train = frame.loc[frame["quarter"].isin(fit_quarters[:-1])]
        inner_valid = frame.loc[frame["quarter"].eq(fit_quarters[-1])]
        best = None
        for prior in config["lambda_prior"]:
            candidate = fit_gate(inner_train, feature_columns, config, float(prior), kind)
            predicted = apply_gate(candidate, inner_valid, "inner_prediction")
            score = score_prediction(predicted, "inner_prediction")["total_score"]
            key = (score, -abs(float(prior)-0.05))
            if best is None or key > best[0]:
                best = (key, float(prior), score)
        prior = best[1]; selected.append(prior)
        model = fit_gate(frame.loc[frame["quarter"].isin(fit_quarters)], feature_columns, config, prior, kind)
        evaluation = apply_gate(model, evaluation)
        metric = score_prediction(evaluation, "gate_prediction")
        rows.append({
            "evaluation_quarter": quarter, "status": "nested_selected",
            "fit_quarters": repr(fit_quarters), "inner_validation_quarter": fit_quarters[-1],
            "lambda_prior": prior, "inner_score": best[2], **metric,
            "mean_raw_weight": float(evaluation["gate_raw_weight"].mean()),
            "std_raw_weight": float(evaluation["gate_raw_weight"].std()),
        })
        parts.append(evaluation)
    nested = pd.concat(parts, ignore_index=True)
    final_prior = Counter(selected).most_common(1)[0][0] if selected else 0.05
    final_model = fit_gate(frame, feature_columns, config, final_prior, kind)
    return nested, pd.DataFrame(rows), final_model
