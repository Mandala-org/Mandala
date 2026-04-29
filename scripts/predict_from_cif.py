#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.structure_inference import (  # noqa: E402
    build_model_input_from_structure,
    load_orbital_cfg_from_reference_info,
    load_structure_from_cif,
)
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mandala checkpoint inference on a CIF structure."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cif-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-info-path",
        type=Path,
        default=None,
        help="OpenMX-style info file used to recover the orbital basis.",
    )
    parser.add_argument(
        "--orbital-set",
        type=str,
        default=None,
        help="YAML/JSON mapping like '{Si: 3s3p2d1f}'. Used if no reference info is provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Also save the prepared model input tensors for debugging.",
    )
    return parser.parse_args()


def _patch_config_unpickling() -> None:
    if getattr(Config, "_mandala_legacy_unpickle_patch", False):
        return

    def __setstate__(self, state: Any) -> None:
        state_map: dict[str, Any] = {}
        if isinstance(state, tuple) and len(state) == 2:
            dict_state, slot_state = state
            if isinstance(dict_state, dict):
                state_map.update(dict_state)
            if isinstance(slot_state, dict):
                state_map.update(slot_state)
        elif isinstance(state, dict):
            state_map.update(state)

        defaults = Config()
        for name in Config.__dataclass_fields__:
            if name in state_map:
                object.__setattr__(self, name, state_map[name])
            else:
                object.__setattr__(self, name, getattr(defaults, name))

    Config.__setstate__ = __setstate__  # type: ignore[attr-defined]
    Config._mandala_legacy_unpickle_patch = True  # type: ignore[attr-defined]


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    _patch_config_unpickling()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    return checkpoint


def _restore_config(checkpoint: dict[str, Any]) -> Config:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    cfg = hyper_parameters.get("cfg")
    if not isinstance(cfg, Config):
        raise ValueError(
            "Checkpoint does not contain a Config instance in hyper_parameters['cfg']."
        )
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.dtype, str):
        cfg.dtype = getattr(torch, cfg.dtype)
    if isinstance(cfg.matrix_targets, str):
        cfg.matrix_targets = [cfg.matrix_targets]
    return cfg


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def _resolve_orbital_cfg(args: argparse.Namespace, cfg: Config) -> OrbitalIrrepConfig:
    if args.reference_info_path is not None:
        return load_orbital_cfg_from_reference_info(
            args.reference_info_path, dtype=cfg.dtype
        )
    if args.orbital_set is not None:
        payload = yaml.safe_load(args.orbital_set)
        if not isinstance(payload, dict):
            raise ValueError(
                "--orbital-set must parse to a mapping like '{Si: 3s3p2d1f}'"
            )
        return OrbitalIrrepConfig.from_dict(payload)
    raise ValueError(
        "Need either --reference-info-path or --orbital-set to define the orbital basis."
    )


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cpu_copy(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [_cpu_copy(val) for val in value]
        return type(value)(copied)
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def main() -> None:
    args = setup_argparse()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = _restore_config(checkpoint)
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None

    device = _resolve_device(args.device)
    orbital_cfg = _resolve_orbital_cfg(args, cfg)
    mapper = BlockIrrepMapper(
        orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=cfg.dtype,
    )
    atoms, positions, box = load_structure_from_cif(
        args.cif_path,
        dtype=cfg.dtype,
        device=device,
    )
    x = build_model_input_from_structure(
        atoms=atoms,
        positions=positions,
        box=box,
        cfg=cfg,
        mapper=mapper,
    )

    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions_irreps = model(x)
        predictions_matrix = {
            name: predictions_irreps[name].to_blocks(model.mapper)
            for name in cfg.matrix_targets
        }

    structure_payload = {
        "atoms": list(atoms),
        "positions": positions.detach().cpu(),
        "box": box.detach().cpu() if box is not None else None,
        "orbital_cfg": orbital_cfg.to_dict(),
        "checkpoint": str(args.checkpoint),
        "cif_path": str(args.cif_path),
        "matrix_targets": list(cfg.matrix_targets),
    }
    torch.save(structure_payload, output_dir / "structure_metadata.pt")
    if args.save_input:
        torch.save(_cpu_copy(x), output_dir / "model_input.pt")

    for name, block_matrix in predictions_matrix.items():
        block_matrix.save(output_dir / f"{name}.pt")

    print("checkpoint:", args.checkpoint)
    print("cif_path:", args.cif_path)
    print("device:", device)
    print("output_dir:", output_dir)
    print("matrix_targets:", cfg.matrix_targets)
    print("saved_files:", sorted(path.name for path in output_dir.iterdir()))


if __name__ == "__main__":
    main()
