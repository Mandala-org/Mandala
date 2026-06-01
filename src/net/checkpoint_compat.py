from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from e3nn.o3 import FullyConnectedTensorProduct, Linear

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config, SeparateWeightTensorProduct
from net.e3mlp_variants import ScaledLinear

if TYPE_CHECKING:
    from net.e3gnn import E3GNN


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
    source_num_species: int | None = None
    source_head_pair_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "source_targets": list(self.source_targets),
            "target_clone_sources": dict(self.target_clone_sources),
            "exact_copied": list(self.exact_copied),
            "cloned_keys": list(self.cloned_keys),
            "remapped_keys": list(self.remapped_keys),
            "skipped_keys": list(self.skipped_keys),
            "source_num_species": self.source_num_species,
            "source_head_pair_mode": self.source_head_pair_mode,
        }


def strip_mapper_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(MAPPER_KEY_PREFIXES) and MAPPER_KEY_INFIX not in key
    }


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a mapping.")
    return checkpoint


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a model state_dict mapping.")
    return strip_mapper_keys(state_dict)


def initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> CompatibilityReport:
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    checkpoint = load_checkpoint_payload(checkpoint_path)
    source_state = load_checkpoint_state_dict(checkpoint_path)
    target_state = model.state_dict()
    report = CompatibilityReport(checkpoint_path=checkpoint_path)
    report.source_targets = _discover_source_targets(source_state)
    report.source_num_species = _infer_source_num_species(source_state)
    source_cfg = _extract_source_config(checkpoint)
    report.source_head_pair_mode = getattr(source_cfg, "head_pair_mode", None)
    print(
        "[compat] initialize_model_from_checkpoint "
        f"checkpoint_path={checkpoint_path} "
        f"source_targets={report.source_targets} "
        f"source_head_pair_mode={report.source_head_pair_mode} "
        f"source_num_species={report.source_num_species}"
    )
    clone_sources = _resolve_target_clone_sources(model, report.source_targets)
    report.target_clone_sources = clone_sources
    print(f"[compat] target_clone_sources={clone_sources}")

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
    source_model = _build_source_model_skeleton(
        source_cfg=source_cfg,
        source_state=source_state,
    )
    if source_model is not None:
        print("[compat] source model skeleton built successfully")
        _apply_equivariant_module_remaps(
            source_model=source_model,
            target_model=model,
            source_state=source_state,
            staged_state=staged_state,
            report=report,
        )
        _apply_shared_projector_initialization(
            source_model=source_model,
            target_model=model,
            source_state=source_state,
            staged_state=staged_state,
            report=report,
        )
    else:
        print(
            "[compat] source model skeleton could not be built; skipping module-level remaps"
        )
    print(
        "[compat] load summary "
        f"exact={len(report.exact_copied)} "
        f"cloned={len(report.cloned_keys)} "
        f"remapped={len(report.remapped_keys)} "
        f"skipped={len(report.skipped_keys)}"
    )
    model.load_state_dict(staged_state, strict=False)
    return report


def _extract_source_config(checkpoint: dict[str, Any]) -> Config:
    hyper = checkpoint.get("hyper_parameters", {})
    cfg_payload = hyper.get("cfg", None)
    if isinstance(cfg_payload, Config):
        return cfg_payload
    cfg = Config()
    if cfg_payload is None:
        return cfg
    if is_dataclass(cfg_payload):
        cfg_data = _safe_dataclass_to_dict(cfg_payload)
    elif isinstance(cfg_payload, dict):
        cfg_data = dict(cfg_payload)
    else:
        cfg_data = dict(vars(cfg_payload))
    for key, value in cfg_data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if not hasattr(cfg, "head_pair_mode"):
        cfg.head_pair_mode = "split"
    print(
        "[compat] extracted source config "
        f"head_pair_mode={getattr(cfg, 'head_pair_mode', None)} "
        f"hidden_irreps={getattr(cfg, 'hidden_irreps', None)} "
        f"l_max={getattr(cfg, 'l_max', None)} "
        f"hidden_base_dim={getattr(cfg, 'hidden_base_dim', None)}"
    )
    return cfg


def _infer_source_num_species(source_state: dict[str, torch.Tensor]) -> int | None:
    node_key = "node_enc.elem_emb.weight"
    if node_key not in source_state:
        return None
    return int(source_state[node_key].shape[0])


def _build_source_model_skeleton(
    source_cfg: Config,
    source_state: dict[str, torch.Tensor],
) -> "E3GNN" | None:
    num_species = _infer_source_num_species(source_state)
    if num_species is None or num_species <= 0:
        return None
    dummy_orbitals = {f"E{i}": "1x0e" for i in range(num_species)}
    dummy_cfg = _copy_config(source_cfg)
    dummy_cfg.train_on_energy = False
    dummy_cfg.train_on_num_electrons = False
    dummy_cfg.enable_energy = False
    dummy_cfg.enable_num_electrons = False
    dummy_cfg.verbosity = 0
    try:
        from net.e3gnn import E3GNN

        mapper = BlockIrrepMapper(
            OrbitalIrrepConfig.from_dict(dummy_orbitals),
            diagonal=False,
            device="cpu",
            dtype=dummy_cfg.dtype,
        )
        return E3GNN(mapper=mapper, cfg=dummy_cfg)
    except Exception as exc:
        print(f"[compat] failed to build source model skeleton: {exc}")
        return None


def _copy_config(cfg: Config) -> Config:
    new_cfg = Config()
    if is_dataclass(cfg):
        cfg_data = _safe_dataclass_to_dict(cfg)
    else:
        cfg_data = dict(vars(cfg))
    for key, value in cfg_data.items():
        if hasattr(new_cfg, key):
            setattr(new_cfg, key, value)
    if not hasattr(new_cfg, "head_pair_mode"):
        new_cfg.head_pair_mode = "split"
    return new_cfg


def _safe_dataclass_to_dict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_info in fields(obj):
        if not hasattr(obj, field_info.name):
            if field_info.default is not MISSING:
                out[field_info.name] = field_info.default
            elif field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
                out[field_info.name] = field_info.default_factory()  # type: ignore[misc]
            continue
        out[field_info.name] = getattr(obj, field_info.name)
    return out


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
            print(
                f"[compat] remapped embedding {node_key} "
                f"src_shape={tuple(src.shape)} dst_shape={tuple(dst.shape)}"
            )

    edge_key = "edge_enc.edge_emb.weight"
    if edge_key in source_state and edge_key not in staged_state:
        src = source_state[edge_key]
        dst = model.state_dict().get(edge_key)
        if dst is not None:
            remapped = _copy_common_embedding_rows(src, dst)
            staged_state[edge_key] = remapped
            report.remapped_keys.append(edge_key)
            print(
                f"[compat] remapped embedding {edge_key} "
                f"src_shape={tuple(src.shape)} dst_shape={tuple(dst.shape)}"
            )


def _copy_common_embedding_rows(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    out = dst.detach().clone()
    rows = min(src.shape[0], dst.shape[0])
    cols = min(src.shape[1], dst.shape[1])
    if rows > 0 and cols > 0:
        out[:rows, :cols] = src[:rows, :cols].to(dtype=out.dtype)
    if dst.shape[0] > rows and rows > 0:
        fill = src[:rows, :cols].mean(dim=0, keepdim=True).to(dtype=out.dtype)
        out[rows:, :cols] = fill.expand(dst.shape[0] - rows, cols)
    return out


def _apply_equivariant_module_remaps(
    *,
    source_model: "E3GNN",
    target_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    src_modules = dict(source_model.named_modules())
    dst_modules = dict(target_model.named_modules())
    print(
        "[compat] applying equivariant module remaps "
        f"source_modules={len(src_modules)} target_modules={len(dst_modules)}"
    )
    for module_name, dst_module in dst_modules.items():
        if not module_name or module_name not in src_modules:
            continue
        src_module = src_modules[module_name]
        if isinstance(dst_module, nn.Linear) and isinstance(src_module, nn.Linear):
            _remap_linear(module_name, target_model, source_state, staged_state, report)
            continue
        if isinstance(dst_module, ScaledLinear) and isinstance(
            src_module, ScaledLinear
        ):
            _remap_e3_linear(
                module_name=f"{module_name}.linear",
                src_module=src_module.linear,
                dst_module=dst_module.linear,
                target_model=target_model,
                source_state=source_state,
                staged_state=staged_state,
                report=report,
            )
            continue
        if isinstance(dst_module, Linear) and isinstance(src_module, Linear):
            _remap_e3_linear(
                module_name=module_name,
                src_module=src_module,
                dst_module=dst_module,
                target_model=target_model,
                source_state=source_state,
                staged_state=staged_state,
                report=report,
            )
            continue
        if isinstance(dst_module, FullyConnectedTensorProduct) and isinstance(
            src_module, FullyConnectedTensorProduct
        ):
            _remap_fctp(
                module_name=module_name,
                src_module=src_module,
                dst_module=dst_module,
                target_model=target_model,
                source_state=source_state,
                staged_state=staged_state,
                report=report,
            )
            continue
        if isinstance(dst_module, SeparateWeightTensorProduct) and isinstance(
            src_module, SeparateWeightTensorProduct
        ):
            _remap_separate_weight_tp(
                module_name=module_name,
                src_module=src_module,
                dst_module=dst_module,
                target_model=target_model,
                source_state=source_state,
                staged_state=staged_state,
                report=report,
            )


def _remap_linear(
    module_name: str,
    target_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    weight_key = f"{module_name}.weight"
    bias_key = f"{module_name}.bias"
    target_state = target_model.state_dict()
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


def _remap_e3_linear(
    *,
    module_name: str,
    src_module: Linear,
    dst_module: Linear,
    target_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    weight_key = f"{module_name}.weight"
    target_state = target_model.state_dict()
    if (
        weight_key in staged_state
        or weight_key not in source_state
        or weight_key not in target_state
    ):
        return
    src_weight = source_state[weight_key]
    dst_weight = target_state[weight_key].detach().clone()
    matched = False
    src_views = list(src_module.weight_views(weight=src_weight, yield_instruction=True))
    for _, dst_ins, dst_view in dst_module.weight_views(
        weight=dst_weight, yield_instruction=True
    ):
        for _, src_ins, src_view in src_views:
            if (
                src_module.irreps_in[src_ins.i_in].ir
                == dst_module.irreps_in[dst_ins.i_in].ir
                and src_module.irreps_out[src_ins.i_out].ir
                == dst_module.irreps_out[dst_ins.i_out].ir
            ):
                _copy_overlap(src_view, dst_view)
                matched = True
                break
    if matched:
        staged_state[weight_key] = dst_weight
        report.remapped_keys.append(weight_key)
        print(f"[compat] remapped e3 linear {weight_key}")


def _remap_fctp(
    *,
    module_name: str,
    src_module: FullyConnectedTensorProduct,
    dst_module: FullyConnectedTensorProduct,
    target_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    weight_key = f"{module_name}.weight"
    target_state = target_model.state_dict()
    if (
        weight_key in staged_state
        or weight_key not in source_state
        or weight_key not in target_state
    ):
        return
    src_weight = source_state[weight_key]
    dst_weight = target_state[weight_key].detach().clone()
    matched = False
    src_views = list(src_module.weight_views(weight=src_weight, yield_instruction=True))
    for _, dst_ins, dst_view in dst_module.weight_views(
        weight=dst_weight, yield_instruction=True
    ):
        for _, src_ins, src_view in src_views:
            if (
                src_module.irreps_in1[src_ins.i_in1].ir
                == dst_module.irreps_in1[dst_ins.i_in1].ir
                and src_module.irreps_in2[src_ins.i_in2].ir
                == dst_module.irreps_in2[dst_ins.i_in2].ir
                and src_module.irreps_out[src_ins.i_out].ir
                == dst_module.irreps_out[dst_ins.i_out].ir
                and src_ins.connection_mode == dst_ins.connection_mode
            ):
                _copy_overlap(src_view, dst_view)
                matched = True
                break
    if matched:
        staged_state[weight_key] = dst_weight
        report.remapped_keys.append(weight_key)
        print(f"[compat] remapped tensor product {weight_key}")


def _remap_separate_weight_tp(
    *,
    module_name: str,
    src_module: SeparateWeightTensorProduct,
    dst_module: SeparateWeightTensorProduct,
    target_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    src_pairs = _separate_weight_param_pairs(src_module, source_state, module_name)
    dst_pairs = _separate_weight_param_pairs(
        dst_module, target_model.state_dict(), module_name
    )
    if not src_pairs or not dst_pairs:
        return
    matched_any = False
    for dst_idx, dst_ins, dst_w1, dst_w2 in dst_pairs:
        best_match = None
        for src_idx, src_ins, src_w1, src_w2 in src_pairs:
            if (
                src_module.tp.irreps_in1[src_ins.i_in1].ir
                == dst_module.tp.irreps_in1[dst_ins.i_in1].ir
                and src_module.tp.irreps_in2[src_ins.i_in2].ir
                == dst_module.tp.irreps_in2[dst_ins.i_in2].ir
                and src_module.tp.irreps_out[src_ins.i_out].ir
                == dst_module.tp.irreps_out[dst_ins.i_out].ir
                and src_ins.connection_mode == dst_ins.connection_mode
            ):
                best_match = (src_w1, src_w2)
                break
        if best_match is None:
            continue
        src_w1, src_w2 = best_match
        new_w1 = dst_w1.detach().clone()
        new_w2 = dst_w2.detach().clone()
        _copy_overlap(src_w1, new_w1)
        _copy_overlap(src_w2, new_w2)
        staged_state[f"{module_name}.weights1.{dst_idx}"] = new_w1
        staged_state[f"{module_name}.weights2.{dst_idx}"] = new_w2
        report.remapped_keys.append(f"{module_name}.weights1.{dst_idx}")
        report.remapped_keys.append(f"{module_name}.weights2.{dst_idx}")
        print(
            f"[compat] remapped separate weight tp "
            f"{module_name}.weights1.{dst_idx} and {module_name}.weights2.{dst_idx}"
        )
        matched_any = True
    if matched_any:
        return


def _apply_shared_projector_initialization(
    *,
    source_model: "E3GNN",
    target_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    target_heads = getattr(target_model, "heads", None)
    source_heads = getattr(source_model, "heads", None)
    if target_heads is None or source_heads is None:
        return
    for target_name, target_head in target_heads.items():
        source_target_name = report.target_clone_sources.get(target_name, target_name)
        if source_target_name not in source_heads:
            continue
        source_head = source_heads[source_target_name]
        if getattr(target_head, "head_pair_mode", "split") != "shared_conditioned":
            continue
        if getattr(source_head, "head_pair_mode", "split") != "split":
            continue
        print(
            "[compat] initializing shared projector from split head "
            f"target={target_name} source_target={source_target_name}"
        )
        _init_shared_proj_from_split(
            source_head=source_head,
            target_head=target_head,
            source_state=source_state,
            staged_state=staged_state,
            report=report,
            branch_name="diag",
        )
        _init_shared_proj_from_split(
            source_head=source_head,
            target_head=target_head,
            source_state=source_state,
            staged_state=staged_state,
            report=report,
            branch_name="offdiag",
        )
        if (
            getattr(target_head, "shared_shifted_self_proj", None) is not None
            and getattr(source_head, "shifted_self_projs", None) is not None
        ):
            _init_shared_proj_from_split(
                source_head=source_head,
                target_head=target_head,
                source_state=source_state,
                staged_state=staged_state,
                report=report,
                branch_name="shifted_self",
            )


def _init_shared_proj_from_split(
    *,
    source_head: nn.Module,
    target_head: nn.Module,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
    branch_name: str,
) -> None:
    src_proj_dict = getattr(source_head, f"{branch_name}_projs", None)
    dst_shared_proj = getattr(target_head, f"shared_{branch_name}_proj", None)
    if src_proj_dict is None or dst_shared_proj is None:
        return
    pair_keys = list(getattr(source_head, "pair_keys", []))
    if not pair_keys:
        return
    source_matrix_name = _head_matrix_name(source_head)
    target_matrix_name = _head_matrix_name(target_head)
    print(
        "[compat] shared projector aggregation "
        f"branch={branch_name} n_source_pairs={len(pair_keys)} "
        f"source_matrix={source_matrix_name} target_matrix={target_matrix_name}"
    )
    _aggregate_module_group(
        src_modules=[src_proj_dict[key] for key in pair_keys if key in src_proj_dict],
        src_prefixes=[
            f"heads.{source_matrix_name}.{branch_name}_projs.{key}"
            for key in pair_keys
            if key in src_proj_dict
        ],
        dst_module=dst_shared_proj,
        dst_prefix=f"heads.{target_matrix_name}.shared_{branch_name}_proj",
        source_state=source_state,
        staged_state=staged_state,
        target_state=target_head.state_dict(prefix=f"heads.{target_matrix_name}."),
        report=report,
    )


def _aggregate_module_group(
    *,
    src_modules: list[nn.Module],
    src_prefixes: list[str],
    dst_module: nn.Module,
    dst_prefix: str,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    if not src_modules:
        return
    # Only aggregate the child modules we know how to align.
    for rel_name, dst_child in dst_module.named_modules():
        if rel_name == "":
            continue
        full_dst_name = f"{dst_prefix}.{rel_name}"
        if isinstance(dst_child, nn.Linear):
            _aggregate_plain_linear(
                src_modules=src_modules,
                src_prefixes=src_prefixes,
                rel_name=rel_name,
                full_dst_name=full_dst_name,
                source_state=source_state,
                staged_state=staged_state,
                target_state=target_state,
                report=report,
            )
        elif isinstance(dst_child, ScaledLinear):
            _aggregate_e3_linear(
                src_modules=src_modules,
                src_prefixes=src_prefixes,
                rel_name=rel_name,
                full_dst_name=full_dst_name,
                dst_linear=dst_child.linear,
                source_state=source_state,
                staged_state=staged_state,
                target_state=target_state,
                report=report,
            )
        elif isinstance(dst_child, Linear):
            _aggregate_e3_linear(
                src_modules=src_modules,
                src_prefixes=src_prefixes,
                rel_name=rel_name,
                full_dst_name=full_dst_name,
                dst_linear=dst_child,
                source_state=source_state,
                staged_state=staged_state,
                target_state=target_state,
                report=report,
            )


def _aggregate_plain_linear(
    *,
    src_modules: list[nn.Module],
    src_prefixes: list[str],
    rel_name: str,
    full_dst_name: str,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    weight_key = f"{full_dst_name}.weight"
    if weight_key in staged_state or weight_key not in target_state:
        return
    acc = target_state[weight_key].detach().clone().zero_()
    count = 0
    for src_module, src_prefix in zip(src_modules, src_prefixes):
        src_child = dict(src_module.named_modules()).get(rel_name)
        src_weight_key = f"{src_prefix}.{rel_name}.weight"
        if not isinstance(src_child, nn.Linear) or src_weight_key not in source_state:
            continue
        tmp = acc.detach().clone().zero_()
        _copy_overlap(source_state[src_weight_key], tmp)
        acc += tmp
        count += 1
    if count > 0:
        staged_state[weight_key] = acc / float(count)
        report.remapped_keys.append(weight_key)
        print(
            f"[compat] aggregated shared plain linear {weight_key} from {count} sources"
        )


def _aggregate_e3_linear(
    *,
    src_modules: list[nn.Module],
    src_prefixes: list[str],
    rel_name: str,
    full_dst_name: str,
    dst_linear: Linear,
    source_state: dict[str, torch.Tensor],
    staged_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    report: CompatibilityReport,
) -> None:
    weight_key = f"{full_dst_name}.weight"
    if weight_key in staged_state or weight_key not in target_state:
        return
    acc = target_state[weight_key].detach().clone().zero_()
    count = 0
    for src_module, src_prefix in zip(src_modules, src_prefixes):
        src_child = dict(src_module.named_modules()).get(rel_name)
        src_weight_key = f"{src_prefix}.{rel_name}.weight"
        if not isinstance(src_child, Linear) or src_weight_key not in source_state:
            continue
        tmp = acc.detach().clone().zero_()
        _copy_matching_e3_linear_weights(
            src_module=src_child,
            dst_module=dst_linear,
            src_weight=source_state[src_weight_key],
            dst_weight=tmp,
        )
        acc += tmp
        count += 1
    if count > 0:
        staged_state[weight_key] = acc / float(count)
        report.remapped_keys.append(weight_key)
        print(f"[compat] aggregated shared e3 linear {weight_key} from {count} sources")


def _head_matrix_name(head: nn.Module) -> str:
    info = getattr(head, "info", None)
    if isinstance(info, dict):
        return str(info.get("matrix", "hamiltonian"))
    return "hamiltonian"


def _copy_matching_e3_linear_weights(
    *,
    src_module: Linear,
    dst_module: Linear,
    src_weight: torch.Tensor,
    dst_weight: torch.Tensor,
) -> None:
    src_views = list(src_module.weight_views(weight=src_weight, yield_instruction=True))
    for _, dst_ins, dst_view in dst_module.weight_views(
        weight=dst_weight, yield_instruction=True
    ):
        for _, src_ins, src_view in src_views:
            if (
                src_module.irreps_in[src_ins.i_in].ir
                == dst_module.irreps_in[dst_ins.i_in].ir
                and src_module.irreps_out[src_ins.i_out].ir
                == dst_module.irreps_out[dst_ins.i_out].ir
            ):
                _copy_overlap(src_view, dst_view)
                break


def _separate_weight_param_pairs(
    module: SeparateWeightTensorProduct,
    state_dict: dict[str, torch.Tensor],
    module_name: str,
) -> list[tuple[int, Any, torch.Tensor, torch.Tensor]]:
    out: list[tuple[int, Any, torch.Tensor, torch.Tensor]] = []
    for idx, ins in enumerate(module.tp.instructions):
        key1 = f"{module_name}.weights1.{idx}"
        key2 = f"{module_name}.weights2.{idx}"
        if key1 not in state_dict or key2 not in state_dict:
            continue
        out.append((idx, ins, state_dict[key1], state_dict[key2]))
    return out


def _copy_overlap(src: torch.Tensor, dst: torch.Tensor) -> None:
    sizes = tuple(min(a, b) for a, b in zip(src.shape, dst.shape))
    if any(size <= 0 for size in sizes):
        return
    src_slices = tuple(slice(0, size) for size in sizes)
    dst_slices = tuple(slice(0, size) for size in sizes)
    dst[dst_slices] = src[src_slices].to(dtype=dst.dtype)
