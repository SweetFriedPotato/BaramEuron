"""Checkpoint transfer and constrained Stage-2 unfreezing policies."""

from __future__ import annotations

import json
from pathlib import Path

import torch


HEAD_PREFIXES = ("power_head.", "auxiliary_head.", "hub_retention_head.", "hub_encoder.")


def load_matching_encoder_weights(model: torch.nn.Module, checkpoint: Path | dict) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if isinstance(checkpoint, Path) else checkpoint
    source = payload.get("state_dict", payload)
    destination = model.state_dict()
    loaded, skipped = [], []
    for name, value in source.items():
        if name.startswith(HEAD_PREFIXES) or name not in destination or destination[name].shape != value.shape:
            skipped.append(name)
            continue
        destination[name] = value.detach().clone()
        loaded.append(name)
    model.load_state_dict(destination)
    if not any(name.startswith("ldaps_encoder.") for name in loaded):
        raise ValueError("checkpoint did not transfer the Exp04 raw-grid encoder")
    return {"loaded": loaded, "skipped": skipped, "source_kind": "Exp03/Exp04 champion component"}


def load_stage1_retention_head(model: torch.nn.Module, checkpoint: Path | dict) -> dict:
    """Initialize the joint model's retained hub-wind head from Stage 1."""
    if getattr(model, "hub_retention_head", None) is None:
        raise ValueError("joint Stage-2 model does not expose a hub retention head")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if isinstance(checkpoint, Path) else checkpoint
    source = payload.get("state_dict", payload)
    destination = model.state_dict()
    loaded = []
    for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
        source_name = f"power_head.{suffix}"
        destination_name = f"hub_retention_head.{suffix}"
        if source_name not in source or destination_name not in destination:
            raise ValueError(f"missing Stage-1 retention parameter: {source_name}")
        if source[source_name].shape != destination[destination_name].shape:
            raise ValueError(f"retention parameter shape mismatch: {source_name}")
        destination[destination_name] = source[source_name].detach().clone()
        loaded.append(destination_name)
    model.load_state_dict(destination)
    return {"loaded": loaded, "source_kind": "Stage-1 hub-wind distribution head"}


def load_stage1_from_exp04(model: torch.nn.Module, checkpoint: Path | dict, *, auxiliary_init: bool = False) -> dict:
    """Load all shape-compatible Exp04 representation weights and optional auxiliary head."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if isinstance(checkpoint, Path) else checkpoint
    source = payload.get("state_dict", payload)
    destination = model.state_dict()
    loaded, skipped = [], []
    for name, value in source.items():
        if name.startswith("power_head.") or name not in destination or destination[name].shape != value.shape:
            skipped.append(name)
            continue
        destination[name] = value.detach().clone()
        loaded.append(name)
    model.load_state_dict(destination)
    if auxiliary_init:
        model.initialize_median_from_auxiliary_head()
    return {"loaded": loaded, "skipped": skipped, "auxiliary_head_initialization": bool(auxiliary_init)}


def _last_temporal_prefix(model: torch.nn.Module) -> str:
    count = len(model.temporal.temporal)
    if count == 0:
        raise ValueError("temporal encoder has no residual blocks")
    return f"temporal.temporal.{count - 1}."


def apply_transfer_policy(model: torch.nn.Module, variant: str, epoch: int = 0) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False
    allowed = ["power_head."]
    if variant == "pretrained_encoder":
        allowed.append(_last_temporal_prefix(model))
    elif variant in {"explicit_hubwind", "distribution_hubwind"}:
        allowed.extend(["hub_encoder.", "final_projection."])
        if epoch >= 5:
            allowed.extend([_last_temporal_prefix(model), "fusion.gate."])
    elif variant == "joint_finetune":
        allowed.extend([
            "hub_encoder.", "final_projection.", "hub_retention_head.",
            _last_temporal_prefix(model), "fusion.gate.",
        ])
    else:
        raise ValueError(f"unknown transfer policy: {variant}")
    trainable = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in allowed):
            parameter.requires_grad = True
            trainable.append(name)
    forbidden_spatial = [name for name in trainable if name.startswith(("ldaps_encoder.", "gfs_encoder."))]
    if forbidden_spatial:
        raise ValueError(f"raw spatial encoder unfreeze is prohibited: {forbidden_spatial}")
    return {
        "variant": variant,
        "epoch": int(epoch),
        "freeze_first_epochs": 5 if variant in {"explicit_hubwind", "distribution_hubwind"} else 0,
        "trainable": trainable,
        "raw_spatial_fully_frozen": True,
    }


def optimizer_groups(model: torch.nn.Module, variant: str) -> list[dict]:
    head, encoder = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head if name.startswith(("power_head.", "hub_retention_head.")) else encoder).append(parameter)
    if variant == "joint_finetune":
        groups = []
        if encoder:
            groups.append({"params": encoder, "lr": 1e-5, "name": "encoder"})
        if head:
            groups.append({"params": head, "lr": 1e-4, "name": "head"})
        return groups
    return [{"params": [*encoder, *head], "lr": 1e-4, "name": "selected_parameters"}]


def write_transfer_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
