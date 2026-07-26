#!/usr/bin/env python
"""Evaluate absolute and population-consistent relative metrics on a validation split.

This script is intended for the paper checkpoints on rosi.  It restores the
checkpoint, reconstructs the original deterministic validation split, and
uses the same validation step as training.  Relative matrix errors divide the
mean validation MAE by the mean absolute reference-matrix element over that
same validation population.  The band-energy ratio analogously uses the mean
absolute reference band energy over the same structures.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.block_matrix import BlockMatrix  # noqa: E402
from net.checkpoint_compat import (
    load_checkpoint_payload,
    strip_mapper_keys,
)  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from scripts.check_compat_reload_equivalence import (  # noqa: E402
    _build_dataset_bundle,
    _evaluate_shared_step,
    _move_to_device,
)
from scripts.evaluate_checkpoint_materials import (  # noqa: E402
    _patch_config_unpickling,
    _restore_config,
)
from utils.units import HARTREE_TO_EV  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute validation MAEs and relative errors whose numerators and "
            "denominators use the same validation population."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        required=True,
        choices=["silicon_scales", "siox", "ZnCuSnSeS"],
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scales", default="[1]")
    parser.add_argument("--num-train-per-scale", type=int, default=80)
    parser.add_argument("--num-val-per-scale", type=int, default=20)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-split-seed", type=int, default=42)
    parser.add_argument("--snapshot-cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-device", default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--convention", default="e3nn")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _matrix_mean_abs(matrix: BlockMatrix) -> float:
    absolute_sum = 0.0
    scalar_count = 0
    for blocks in matrix.pair_blocks.values():
        absolute_sum += float(torch.sum(torch.abs(blocks)).detach().cpu().item())
        scalar_count += int(blocks.numel())
    if scalar_count == 0:
        raise ValueError("Cannot normalize by an empty reference matrix.")
    return absolute_sum / scalar_count


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().reshape(-1)
        if tensor.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return float(tensor.item())
    return float(value)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    _patch_config_unpickling()
    checkpoint = load_checkpoint_payload(checkpoint_path)
    cfg = deepcopy(_restore_config(checkpoint))
    cfg.snapshot_cache_dir = (
        None
        if args.snapshot_cache_dir is None
        else str(args.snapshot_cache_dir.expanduser().resolve())
    )
    cfg.dataset_device = args.dataset_device
    cfg.data_split_seed = int(args.data_split_seed)
    cfg.verbosity = 0

    _train_ds, val_ds, mapper = _build_dataset_bundle(args, cfg)
    if len(val_ds) == 0:
        raise RuntimeError("The reconstructed validation split is empty.")

    device = torch.device(args.device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = E3GNN(mapper=mapper, cfg=deepcopy(cfg)).to(device)
    model.load_state_dict(strip_mapper_keys(state_dict), strict=True)
    model.eval()

    metric_values: dict[str, list[float]] = {}
    reference_values: dict[str, list[float]] = {}
    energy_reference_per_atom_ev: list[float] = []

    for batch_idx in tqdm(range(len(val_ds)), desc="Validation structures"):
        x, y = val_ds[batch_idx]
        batch = (_move_to_device(x, device), _move_to_device(y, device))
        _loss, metrics = _evaluate_shared_step(model, batch, batch_idx=batch_idx)

        for matrix_name in ("hamiltonian", "density", "overlap"):
            metric_key = f"val/{matrix_name}_mae"
            target = y.get(matrix_name)
            if metric_key not in metrics or not isinstance(target, BlockMatrix):
                continue
            metric_values.setdefault(metric_key, []).append(float(metrics[metric_key]))
            magnitude = _matrix_mean_abs(target)
            if matrix_name == "hamiltonian":
                magnitude *= HARTREE_TO_EV
            reference_values.setdefault(matrix_name, []).append(magnitude)

        for metric_key in (
            "val/energy_mae",
            "val/energy_mae_gt_hamiltonian",
            "val/energy_mae_gt_density",
        ):
            if metric_key in metrics:
                atom_count = int(x["positions"].shape[0])
                metric_values.setdefault(metric_key, []).append(
                    float(metrics[metric_key]) / atom_count
                )
        if y.get("energy") is not None:
            atom_count = int(x["positions"].shape[0])
            energy_reference_per_atom_ev.append(
                abs(_scalar(y["energy"])) * HARTREE_TO_EV / atom_count
            )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "dataset_kind": args.dataset_kind,
        "validation_structures": len(val_ds),
        "data_split_seed": args.data_split_seed,
        "definition": (
            "ratio of the mean per-structure validation MAE to the mean absolute "
            "reference magnitude over the same validation structures"
        ),
        "metrics": {},
    }

    for matrix_name in ("hamiltonian", "density", "overlap"):
        metric_key = f"val/{matrix_name}_mae"
        if metric_key not in metric_values:
            continue
        mae = sum(metric_values[metric_key]) / len(metric_values[metric_key])
        denominator = sum(reference_values[matrix_name]) / len(
            reference_values[matrix_name]
        )
        payload["metrics"][matrix_name] = {
            "mae": mae,
            "mean_absolute_reference": denominator,
            "relative_mae": mae / denominator,
            "relative_mae_percent": 100.0 * mae / denominator,
            "unit": "eV" if matrix_name == "hamiltonian" else "dimensionless",
        }

    if energy_reference_per_atom_ev:
        denominator = sum(energy_reference_per_atom_ev) / len(
            energy_reference_per_atom_ev
        )
        for metric_key in (
            "val/energy_mae",
            "val/energy_mae_gt_hamiltonian",
            "val/energy_mae_gt_density",
        ):
            if metric_key not in metric_values:
                continue
            mae = sum(metric_values[metric_key]) / len(metric_values[metric_key])
            payload["metrics"][metric_key] = {
                "mae_ev_per_atom": mae,
                "mean_absolute_reference_ev_per_atom": denominator,
                "relative_mae": mae / denominator,
                "relative_mae_percent": 100.0 * mae / denominator,
            }

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
