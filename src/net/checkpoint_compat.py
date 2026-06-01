from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

MAPPER_KEY_PREFIXES = ("mapper.",)
MAPPER_KEY_INFIX = ".mapper."


@dataclass
class CompatibilityReport:
    checkpoint_path: str
    source_targets: list[str] = field(default_factory=list)
    target_clone_sources: dict[str, str] = field(default_factory=dict)
    exact_copied: list[str] = field(default_factory=list)
    cloned_keys: list[str] = field(default_factory=list)
    remapped_keys: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "source_targets": list(self.source_targets),
            "target_clone_sources": dict(self.target_clone_sources),
            "exact_copied": list(self.exact_copied),
            "cloned_keys": list(self.cloned_keys),
            "remapped_keys": list(self.remapped_keys),
            "skipped_keys": list(self.skipped_keys),
        }


def strip_mapper_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(MAPPER_KEY_PREFIXES) and MAPPER_KEY_INFIX not in key
    }


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a model state_dict mapping.")
    return strip_mapper_keys(state_dict)


def initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> CompatibilityReport:
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    source_state = load_checkpoint_state_dict(checkpoint_path)
    target_state = model.state_dict()
    report = CompatibilityReport(checkpoint_path=checkpoint_path)
    report.source_targets = _discover_source_targets(source_state)
    clone_sources = _resolve_target_clone_sources(model, report.source_targets)
    report.target_clone_sources = clone_sources

    staged_state: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            staged_state[key] = value
            report.exact_copied.append(key)
            continue
        report.skipped_keys.append(key)

    for dst_target, src_target in clone_sources.items():
        if src_target == dst_target:
            continue
        src_prefix = f"heads.{src_target}."
        dst_prefix = f"heads.{dst_target}."
        for key, value in source_state.items():
            if not key.startswith(src_prefix):
                continue
            new_key = dst_prefix + key[len(src_prefix) :]
            if new_key in staged_state:
                continue
            target_value = target_state.get(new_key)
            if target_value is None:
                continue
            if target_value.shape == value.shape:
                staged_state[new_key] = value
                report.cloned_keys.append(new_key)

    _apply_embedding_remaps(model, source_state, staged_state, report)
    _apply_linear_module_remaps(model, source_state, staged_state, report)
    model.load_state_dict(staged_state, strict=False)
    return report


def _discover_source_targets(state_dict: dict[str, torch.Tensor]) -> list[str]:
    targets: set[str] = set()
    for key in state_dict:
        if not key.startswith("heads."):
            continue
        parts = key.split(".")
        if len(parts) >= 2:
            targets.add(parts[1])
    return sorted(targets)


def _resolve_target_clone_sources(
    model: nn.Module,
    source_targets: list[str],
) -> dict[str, str]:
    cfg_targets = list(getattr(getattr(model, "cfg", None), "matrix_targets", []))
    if not cfg_targets:
        return {}
    preferred_source = "hamiltonian" if "hamiltonian" in source_targets else None
    if preferred_source is None and source_targets:
        preferred_source = source_targets[0]
    clone_sources: dict[str, str] = {}
    for target in cfg_targets:
        if target in source_targets:
            clone_sources[target] = target
        elif preferred_source is not None:
            clone_sources[target] = preferred_source
    return clone_sources


def _apply_embedding_remaps(
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    node_key = "node_enc.elem_emb.weight"
    if node_key in source_state and node_key not in staged_state:
        src = source_state[node_key]
        dst = model.state_dict().get(node_key)
        if dst is not None:
            remapped = _copy_common_embedding_rows(src, dst)
            staged_state[node_key] = remapped
            report.remapped_keys.append(node_key)

    edge_key = "edge_enc.edge_emb.weight"
    if edge_key in source_state and edge_key not in staged_state:
        src = source_state[edge_key]
        dst = model.state_dict().get(edge_key)
        if dst is not None:
            remapped = _copy_common_embedding_rows(src, dst)
            staged_state[edge_key] = remapped
            report.remapped_keys.append(edge_key)


def _copy_common_embedding_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    out = dst.detach().clone()
    rows = min(src.shape[0], dst.shape[0])
    cols = min(src.shape[1], dst.shape[1])
    if rows > 0 and cols > 0:
        out[:rows, :cols] = src[:rows, :cols].to(dtype=out.dtype)
    if dst.shape[0] > rows and rows > 0:
        fill = src[:rows, :cols].mean(dim=0, keepdim=True).to(dtype=out.dtype)
        out[rows:, :cols] = fill.expand(dst.shape[0] - rows, cols)
    return out


def _apply_linear_module_remaps(
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    target_modules = dict(model.named_modules())
    for module_name, module in target_modules.items():
        if not module_name:
            continue
        if isinstance(module, nn.Linear):
            _remap_linear(module_name, model, source_state, staged_state, report)


def _remap_linear(
    module_name: str,
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    weight_key = f"{module_name}.weight"
    bias_key = f"{module_name}.bias"
    target_state = model.state_dict()
    if (
        weight_key in staged_state
        or weight_key not in source_state
        or weight_key not in target_state
    ):
        return
    src = source_state[weight_key]
    dst = target_state[weight_key]
    remapped = dst.detach().clone()
    rows = min(src.shape[0], dst.shape[0])
    cols = min(src.shape[1], dst.shape[1])
    remapped[:rows, :cols] = src[:rows, :cols].to(dtype=remapped.dtype)
    staged_state[weight_key] = remapped
    report.remapped_keys.append(weight_key)
    if (
        bias_key in source_state
        and bias_key in target_state
        and bias_key not in staged_state
        and source_state[bias_key].ndim == 1
        and target_state[bias_key].ndim == 1
    ):
        bias = target_state[bias_key].detach().clone()
        n = min(source_state[bias_key].shape[0], bias.shape[0])
        bias[:n] = source_state[bias_key][:n].to(dtype=bias.dtype)
        staged_state[bias_key] = bias
        report.remapped_keys.append(bias_key)
