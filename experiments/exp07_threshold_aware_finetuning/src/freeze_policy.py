"""Explicit conservative trainability policies for Exp03 and Exp04 models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True)
class FreezeManifest:
    policy: str
    model_family: str
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int

    def to_dict(self) -> dict:
        return asdict(self)


def _family(model: nn.Module) -> str:
    if hasattr(model, "power_heads") and hasattr(model, "input_projection"):
        return "exp03_tcn"
    if hasattr(model, "power_head") and hasattr(model, "ldaps_encoder"):
        return "exp04_raw"
    raise TypeError(f"unsupported model for Exp07 freeze policy: {type(model).__name__}")


def _head_prefix(family: str) -> str:
    return "power_heads" if family == "exp03_tcn" else "power_head"


def _last_block_prefix(family: str) -> str:
    return "temporal.3" if family == "exp03_tcn" else "temporal.temporal.3"


def _last_layer_norm_prefix(model: nn.Module) -> str | None:
    names = [name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    return names[-1] if names else None


def apply_freeze_policy(model: nn.Module, policy: str = "head_only") -> FreezeManifest:
    family = _family(model)
    if policy not in {"head_only", "last_block"}:
        raise ValueError(f"unsupported freeze policy: {policy}")
    head = _head_prefix(family)
    block = _last_block_prefix(family)
    norm = _last_layer_norm_prefix(model)
    for name, parameter in model.named_parameters():
        trainable = name.startswith(head)
        if policy == "last_block":
            trainable = trainable or name.startswith(block)
            trainable = trainable or (norm is not None and name.startswith(norm))
        parameter.requires_grad_(trainable)
    trainable_names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    frozen_names = tuple(name for name, value in model.named_parameters() if not value.requires_grad)
    if not trainable_names:
        raise AssertionError("freeze policy left no trainable parameters")
    if any(name.startswith(("aux_heads", "auxiliary_head")) for name in trainable_names):
        raise AssertionError("auxiliary head must remain frozen")
    if policy == "head_only" and any(not name.startswith(head) for name in trainable_names):
        raise AssertionError("head-only policy opened a non-head parameter")
    return FreezeManifest(
        policy=policy,
        model_family=family,
        trainable_names=trainable_names,
        frozen_names=frozen_names,
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        total_parameters=sum(value.numel() for value in model.parameters()),
    )


def optimizer_groups(
    model: nn.Module,
    policy: str,
    *,
    head_learning_rate: float,
    block_learning_rate: float | None = None,
) -> list[dict]:
    family = _family(model)
    head = _head_prefix(family)
    head_parameters = [value for name, value in model.named_parameters() if value.requires_grad and name.startswith(head)]
    groups = [{"params": head_parameters, "lr": float(head_learning_rate), "name": "power_head"}]
    other = [value for name, value in model.named_parameters() if value.requires_grad and not name.startswith(head)]
    if policy == "last_block" and other:
        if block_learning_rate is None:
            raise ValueError("last-block policy requires block_learning_rate")
        groups.append({"params": other, "lr": float(block_learning_rate), "name": "last_block_and_norm"})
    return groups

