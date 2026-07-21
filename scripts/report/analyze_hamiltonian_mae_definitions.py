#!/usr/bin/env python

"""Audit Hamiltonian MAE definitions in existing evaluation bundles.

The training/W&B metric is an element-weighted MAE over real-space sparse
blocks.  The legacy showcase inventory instead used the sampled dense
correlation payload, whose BlockMatrix.to_dense() input has already summed
periodic images.  This script reports both quantities without rerunning a
model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.block_matrix import BlockMatrix  # noqa: E402
from utils.units import HARTREE_TO_EV  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Evaluation bundle directories; defaults to all bundles below eval_outputs.",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def _discover(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.rglob("hamiltonian_block_error_metrics.pt")
        if (path.parent / "pred_hamiltonian.pt").exists()
    )


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def audit_bundle(path: Path) -> dict[str, Any]:
    block_payload = _load(path / "hamiltonian_block_error_metrics.pt")
    pred = BlockMatrix.load(path / "pred_hamiltonian.pt")
    pair_keys = block_payload["pair_key"]
    edge_maes = block_payload["abs_mae"]
    if len(pair_keys) != len(edge_maes):
        raise ValueError(f"Mismatched pair-key and edge-MAE lengths in {path}")

    weighted_abs_sum = 0.0
    scalar_count = 0
    for pair_key, edge_mae in zip(pair_keys, edge_maes):
        rows, cols = pred.orbital_cfg.block_dims(str(pair_key))
        count = int(rows * cols)
        weighted_abs_sum += float(edge_mae) * count
        scalar_count += count
    sparse_mae_ha = weighted_abs_sum / scalar_count

    sampled_dense_mae_ha = None
    correlation_path = path / "hamiltonian_correlation.pt"
    if correlation_path.exists():
        correlation = _load(correlation_path)
        sampled_pred = correlation.get("pred")
        sampled_target = correlation.get("target")
        if isinstance(sampled_pred, torch.Tensor) and isinstance(
            sampled_target, torch.Tensor
        ):
            sampled_dense_mae_ha = float(
                torch.mean(
                    torch.abs(
                        sampled_pred.to(torch.float64)
                        - sampled_target.to(torch.float64)
                    )
                ).item()
            )

    sparse_mae_ev = sparse_mae_ha * HARTREE_TO_EV
    sampled_dense_mae_ev = (
        None if sampled_dense_mae_ha is None else sampled_dense_mae_ha * HARTREE_TO_EV
    )
    return {
        "evaluation_dir": str(path.relative_to(REPO_ROOT)),
        "real_space_block_mae_ha": sparse_mae_ha,
        "real_space_block_mae_ev": sparse_mae_ev,
        "real_space_scalar_count": scalar_count,
        "sampled_gamma_folded_dense_mae_ha": sampled_dense_mae_ha,
        "sampled_gamma_folded_dense_mae_ev": sampled_dense_mae_ev,
        "dense_to_block_ratio": (
            None
            if sampled_dense_mae_ev is None
            else sampled_dense_mae_ev / sparse_mae_ev
        ),
        "matched_edges": int(block_payload.get("matched_edges", len(edge_maes))),
        "missing_edges": int(block_payload.get("missing_in_pred", 0)),
    }


def main() -> None:
    args = parse_args()
    paths = args.paths or _discover(REPO_ROOT / "eval_outputs")
    rows = [audit_bundle(path.resolve()) for path in paths]

    for row in rows:
        dense = row["sampled_gamma_folded_dense_mae_ev"]
        ratio = row["dense_to_block_ratio"]
        print(
            f"{row['evaluation_dir']}: "
            f"block={row['real_space_block_mae_ev']:.9g} eV, "
            f"sampled_dense={dense:.9g} eV, ratio={ratio:.4f}"
            if dense is not None and ratio is not None
            else f"{row['evaluation_dir']}: "
            f"block={row['real_space_block_mae_ev']:.9g} eV, sampled_dense=n/a"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys()) if rows else []
            )
            if rows:
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
