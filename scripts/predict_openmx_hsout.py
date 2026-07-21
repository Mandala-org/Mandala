#!/usr/bin/env python
"""Predict Si H/S/D matrices and export an OpenMX-compatible ``HS_pred.out``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.basis_converter import OpenMXE3NNConverter  # noqa: E402
from data.openmx_writer import write_openmx_hsout_prediction  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from net.evaluation import (  # noqa: E402
    build_mapper,
    load_checkpoint,
    predict_snapshot,
    restore_config,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/Si_perturbed/hdo_finetune/best_model.pt"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data/small/Si_perturbed_scale1_078"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def symmetrize_predictions(
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """Return Hermitian real-space H/S/D predictions."""

    return {
        name: 0.5 * (matrix + matrix.transpose())
        for name, matrix in predictions.items()
    }


def load_hsd_prediction(
    *,
    checkpoint_path: Path,
    matrix_path: Path,
    info_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Snapshot, Any]:
    """Load the H/S/D model and predict one OpenMX snapshot at its saved cutoff."""

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = restore_config(checkpoint)
    required = {"hamiltonian", "overlap", "density"}
    if set(cfg.matrix_targets) != required:
        raise ValueError(
            f"Checkpoint must predict exactly H/S/D; got {cfg.matrix_targets!r}"
        )
    cfg.verbosity = 0
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None

    reference = Snapshot.from_openmx(
        matrix_path=matrix_path,
        info_path=info_path,
        convention="e3nn",
        symmetrize_density=True,
        cfg=cfg,
    )
    mapper = build_mapper(reference, cfg)
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    predictions = predict_snapshot(model, reference, physical=True)
    predictions = symmetrize_predictions(predictions)
    return predictions, reference, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--matrix-path", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix_path or args.snapshot_dir / "HS.out"
    info_path = args.info_path or args.snapshot_dir / "Si.out"
    output_path = args.output or args.snapshot_dir / "HS_pred.out"
    device = _device(args.device)

    predictions, reference, cfg = load_hsd_prediction(
        checkpoint_path=args.checkpoint,
        matrix_path=matrix_path,
        info_path=info_path,
        device=device,
    )
    converter = OpenMXE3NNConverter(reference.hamiltonian.orbital_cfg)
    predictions_openmx = {
        name: converter.matrix_to_openmx(matrix.to("cpu"))
        for name, matrix in predictions.items()
    }
    stats = write_openmx_hsout_prediction(
        matrix_path,
        output_path,
        predictions_openmx,
        zero_fill_missing=True,
    )
    exported = Snapshot.from_openmx(
        matrix_path=output_path,
        info_path=info_path,
        convention="e3nn",
        symmetrize_density=True,
        cfg=cfg,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"snapshot: {matrix_path}")
    print(f"output: {output_path}")
    print(f"device: {device}")
    print(f"model cutoff radius: {cfg.cutoff_radius} Ang")
    print(f"blocks written: {dict(stats.blocks_written)}")
    print(
        f"zero-filled blocks outside prediction support: {dict(stats.zero_filled_blocks)}"
    )
    print(f"exported band energy: {float(exported.get_energy()):.12f} Ha")
    print(
        "exported electron count: " f"{float(exported.get_number_of_electrons()):.12f}"
    )


if __name__ == "__main__":
    main()
