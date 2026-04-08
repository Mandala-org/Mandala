#!/usr/bin/env python3
"""Verify PySCF rotational equivariance on the H2O original/rotated pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.snapshot import Snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-npz",
        type=Path,
        default=Path("data/pyscf_baseline/results/h2o_original_rks_openmx_like.npz"),
    )
    parser.add_argument(
        "--rotated-npz",
        type=Path,
        default=Path("data/pyscf_baseline/results/h2o_rotated_rks_openmx_like.npz"),
    )
    parser.add_argument("--original-json", type=Path, default=None)
    parser.add_argument("--rotated-json", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pyscf_baseline/results/equivariance"),
    )
    parser.add_argument("--matrix-atol", type=float, default=1e-3)
    parser.add_argument("--vector-atol", type=float, default=2e-4)
    parser.add_argument("--energy-atol", type=float, default=5e-5)
    return parser.parse_args()


def _resolve_json(npz_path: Path, json_path: Path | None) -> Path:
    if json_path is not None:
        return json_path
    return npz_path.with_suffix(".json")


def _nearest_rotation_from_boxes(
    box_orig: torch.Tensor, box_rot: torch.Tensor
) -> torch.Tensor:
    transform = torch.linalg.solve(box_orig, box_rot)
    u, _, vh = torch.linalg.svd(transform)
    rot_t = u @ vh
    if torch.det(rot_t) < 0:
        u[:, -1] *= -1.0
        rot_t = u @ vh
    return rot_t.T


def _max_block_diff(a: Any, b: Any) -> float:
    diff = a - b
    return max(float(block.abs().max().item()) for block in diff.pair_blocks.values())


def _tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _tensor_mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _pyscf_energy(snapshot: Snapshot) -> float:
    pyscf_meta = snapshot.info["pyscf"]
    if "total_energy_hartree" in pyscf_meta:
        return float(pyscf_meta["total_energy_hartree"])
    if "total_energy_ev" in pyscf_meta:
        return float(pyscf_meta["total_energy_ev"])
    raise KeyError("Could not find total energy in PySCF metadata.")


def _pyscf_energy_unit(snapshot: Snapshot) -> str:
    pyscf_meta = snapshot.info["pyscf"]
    if "total_energy_hartree" in pyscf_meta:
        return "hartree"
    if "total_energy_ev" in pyscf_meta:
        return "eV"
    return "unknown"


def _edge_count(snapshot: Snapshot) -> int:
    return sum(
        int(edges.shape[1]) for edges in snapshot.hamiltonian.pair_edges.values()
    )


def _edge_count_by_key(snapshot: Snapshot) -> dict[str, int]:
    return {
        key: int(edges.shape[1])
        for key, edges in snapshot.hamiltonian.pair_edges.items()
    }


def _plot_dense_panel(
    name: str,
    original: torch.Tensor,
    rotated_ref: torch.Tensor,
    rotated_calc: torch.Tensor,
    output_path: Path,
) -> None:
    diff = rotated_ref - rotated_calc
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    mats = [
        ("Original", original),
        ("Rotated", rotated_ref),
        ("Original Rotated", rotated_calc),
        (f"Difference\nMAE={mae:.3e}\nMaxAE={max_abs:.3e}", diff),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for ax, (title, tensor) in zip(axes.flat, mats):
        array = tensor.detach().cpu().numpy()
        im = ax.imshow(array, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("AO index")
        ax.set_ylabel("AO index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{name} | MAE={mae:.3e} | MaxAE={max_abs:.3e}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_hh_block_norm_vs_distance(
    snapshot: Snapshot,
    output_path: Path,
    *,
    pair_key: str = "H-H",
) -> None:
    if snapshot.positions is None or snapshot.box is None:
        raise RuntimeError("Snapshot must contain positions and box for distance plots")
    if pair_key not in snapshot.hamiltonian.pair_blocks:
        raise RuntimeError(f"Snapshot does not contain pair key '{pair_key}'")

    edges = snapshot.hamiltonian.pair_edges[pair_key]
    blocks = snapshot.hamiltonian.pair_blocks[pair_key]
    sx, sy, sz, src, dst = edges
    edge_shift = torch.stack([sx, sy, sz], dim=-1).to(snapshot.positions)
    delta = (
        snapshot.positions[dst] - snapshot.positions[src] + edge_shift @ snapshot.box
    )
    distances = torch.linalg.norm(delta, dim=-1).detach().cpu().numpy()
    norms = (
        torch.linalg.norm(blocks.reshape(blocks.shape[0], -1), dim=-1)
        .detach()
        .cpu()
        .numpy()
    )
    norms = np.clip(norms, 1e-12, None)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(distances, norms, s=36, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Distance (Angstrom)")
    ax.set_ylabel("H-H block L2 magnitude")
    ax.set_title(
        f"H-H Hamiltonian block L2 magnitude vs distance (edges={len(distances)})"
    )
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snap_orig = Snapshot.from_pyscf(
        npz_path=args.original_npz,
        json_path=_resolve_json(args.original_npz, args.original_json),
        convention="e3nn",
    )
    snap_rot = Snapshot.from_pyscf(
        npz_path=args.rotated_npz,
        json_path=_resolve_json(args.rotated_npz, args.rotated_json),
        convention="e3nn",
    )

    if snap_orig.box is None or snap_rot.box is None:
        raise RuntimeError("Both snapshots must contain periodic boxes")
    if snap_orig.positions is None or snap_rot.positions is None:
        raise RuntimeError("Both snapshots must contain positions")

    R = _nearest_rotation_from_boxes(snap_orig.box, snap_rot.box)
    snap_rot_calc = snap_orig.rotate(R)

    force_available = snap_orig.forces is not None and snap_rot.forces is not None
    stress_available = snap_orig.stress is not None and snap_rot.stress is not None
    energy_unit = _pyscf_energy_unit(snap_orig)
    energy_orig = _pyscf_energy(snap_orig)
    energy_rot = _pyscf_energy(snap_rot)

    metrics = {
        "rotation_matrix": R.tolist(),
        "edge_count_total": _edge_count(snap_orig),
        "edge_count_by_key": _edge_count_by_key(snap_orig),
        "energy_unit": energy_unit,
        "energy_original": energy_orig,
        "energy_rotated": energy_rot,
        "energy_delta": abs(energy_orig - energy_rot),
        "positions_max_abs": _tensor_max_abs(
            snap_rot_calc.positions, snap_rot.positions
        ),
        "box_max_abs": _tensor_max_abs(snap_rot_calc.box, snap_rot.box),
        "forces_available": force_available,
        "stress_available": stress_available,
        "forces_max_abs": (
            _tensor_max_abs(snap_rot_calc.forces, snap_rot.forces)
            if force_available
            else None
        ),
        "stress_max_abs": (
            _tensor_max_abs(snap_rot_calc.stress, snap_rot.stress)
            if stress_available
            else None
        ),
        "hamiltonian_mae": _tensor_mae(
            snap_rot_calc.hamiltonian.to_dense(), snap_rot.hamiltonian.to_dense()
        ),
        "hamiltonian_max_abs": _max_block_diff(
            snap_rot_calc.hamiltonian, snap_rot.hamiltonian
        ),
        "overlap_mae": _tensor_mae(
            snap_rot_calc.overlap.to_dense(), snap_rot.overlap.to_dense()
        ),
        "overlap_max_abs": _max_block_diff(snap_rot_calc.overlap, snap_rot.overlap),
        "density_mae": _tensor_mae(
            snap_rot_calc.density.to_dense(), snap_rot.density.to_dense()
        ),
        "density_max_abs": _max_block_diff(snap_rot_calc.density, snap_rot.density),
    }

    _plot_dense_panel(
        "Hamiltonian",
        snap_orig.hamiltonian.to_dense(),
        snap_rot.hamiltonian.to_dense(),
        snap_rot_calc.hamiltonian.to_dense(),
        args.output_dir / "hamiltonian_equivariance.png",
    )
    _plot_dense_panel(
        "Overlap",
        snap_orig.overlap.to_dense(),
        snap_rot.overlap.to_dense(),
        snap_rot_calc.overlap.to_dense(),
        args.output_dir / "overlap_equivariance.png",
    )
    _plot_dense_panel(
        "Density",
        snap_orig.density.to_dense(),
        snap_rot.density.to_dense(),
        snap_rot_calc.density.to_dense(),
        args.output_dir / "density_equivariance.png",
    )
    _plot_hh_block_norm_vs_distance(
        snap_orig,
        args.output_dir / "hh_hamiltonian_block_l2_vs_distance.png",
    )

    report = {
        "inputs": {
            "original_npz": str(args.original_npz),
            "rotated_npz": str(args.rotated_npz),
            "original_json": str(_resolve_json(args.original_npz, args.original_json)),
            "rotated_json": str(_resolve_json(args.rotated_npz, args.rotated_json)),
        },
        "metrics": metrics,
        "plots": {
            "hamiltonian": str(args.output_dir / "hamiltonian_equivariance.png"),
            "overlap": str(args.output_dir / "overlap_equivariance.png"),
            "density": str(args.output_dir / "density_equivariance.png"),
            "hh_hamiltonian_block_l2_vs_distance": str(
                args.output_dir / "hh_hamiltonian_block_l2_vs_distance.png"
            ),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    failures = []
    if metrics["energy_delta"] > args.energy_atol:
        failures.append(
            f"energy delta {metrics['energy_delta']:.3e} > {args.energy_atol:.3e}"
        )
    for name in (
        "positions_max_abs",
        "box_max_abs",
        "forces_max_abs",
        "stress_max_abs",
    ):
        if metrics[name] is not None and metrics[name] > args.vector_atol:
            failures.append(f"{name} {metrics[name]:.3e} > {args.vector_atol:.3e}")
    for name in ("hamiltonian_max_abs", "overlap_max_abs", "density_max_abs"):
        if metrics[name] > args.matrix_atol:
            failures.append(f"{name} {metrics[name]:.3e} > {args.matrix_atol:.3e}")

    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Equivariance check failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
