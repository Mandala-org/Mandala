#!/usr/bin/env python
"""Export OpenMX-compatible predictions using only a checkpoint and structure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from net.e3gnn import E3GNN
from net.evaluation import (
    load_checkpoint,
    restore_config,
)
from ase.io import read as ase_read
from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_writer import export_model_openmx
from net.common import get_torch_dtype


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--structure",
        type=Path,
        required=True,
        help="XYZ/extxyz structure; periodic inputs require lattice metadata.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("openmx",), default="openmx")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    checkpoint = load_checkpoint(args.checkpoint)
    cfg = restore_config(checkpoint)
    if not cfg.matrix_targets or set(cfg.matrix_targets) - {
        "hamiltonian",
        "overlap",
        "density",
    }:
        raise ValueError("Checkpoint must predict a nonempty subset of H/S/D matrices.")
    cfg.verbosity = 0
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None
    orbital_spec = checkpoint.get("hyper_parameters", {}).get("orbital_cfg")
    if orbital_spec is None:
        raise ValueError(
            "Checkpoint lacks orbital_cfg metadata; re-save the loaded model with its mapper orbital configuration."
        )
    structure = ase_read(args.structure)
    if any(structure.pbc) and not all(structure.pbc):
        raise ValueError("Only fully periodic or nonperiodic structures are supported.")
    lattice = structure.cell.array if all(structure.pbc) else None
    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict(orbital_spec), dtype=get_torch_dtype(cfg.dtype)
    )
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    stats = export_model_openmx(
        model,
        atoms=structure.get_chemical_symbols(),
        positions=structure.get_positions(),
        lattice=lattice,
        output_path=args.output,
    )
    print(f"output: {args.output}")
    print(f"blocks written: {dict(stats.blocks_written)}")


if __name__ == "__main__":
    main()
