#!/usr/bin/env python3
"""Analyze shift-resolved graph size for PySCF artifacts and compare to OpenMX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.snapshot import Snapshot


def _resolve_json(npz_path: Path, json_path: Path | None) -> Path:
    if json_path is not None:
        return json_path
    return npz_path.with_suffix(".json")


def _edge_counts(snapshot: Snapshot) -> dict[str, int]:
    return {k: int(v.shape[1]) for k, v in snapshot.hamiltonian.pair_edges.items()}


def _total_edges(snapshot: Snapshot) -> int:
    return sum(_edge_counts(snapshot).values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyscf-npz",
        type=Path,
        default=Path(
            "data/pyscf_baseline/results/h2o_original_rks_openmx_like_prod_k444.npz"
        ),
    )
    parser.add_argument("--pyscf-json", type=Path, default=None)
    parser.add_argument(
        "--openmx-matrix",
        type=Path,
        default=Path("data/small/H2O/original/H2O.matrix"),
    )
    parser.add_argument(
        "--openmx-info",
        type=Path,
        default=Path("data/small/H2O/original/H2O.info.out"),
    )
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pyscf_json = _resolve_json(args.pyscf_npz, args.pyscf_json)
    if not args.pyscf_npz.exists():
        raise FileNotFoundError(
            f"Missing PySCF NPZ: {args.pyscf_npz}. "
            "Run data/pyscf_baseline/run_h2o_original_kmesh_match_openmx.sh first."
        )
    if not pyscf_json.exists():
        raise FileNotFoundError(
            f"Missing PySCF JSON: {pyscf_json}. "
            "The calculation likely did not finish successfully."
        )

    with np.load(args.pyscf_npz, allow_pickle=False) as npz_payload:
        shifts = np.asarray(npz_payload["shifts"], dtype=np.int64)
        nshift = int(shifts.shape[0])

    snap_pyscf = Snapshot.from_pyscf(
        npz_path=args.pyscf_npz,
        json_path=pyscf_json,
        convention="e3nn",
        cutoff_radius=None,
    )
    snap_openmx = Snapshot.from_openmx(
        matrix_path=args.openmx_matrix,
        info_path=args.openmx_info,
        convention="e3nn",
        cutoff_radius=None,
    )

    pyscf_edges_by_key = _edge_counts(snap_pyscf)
    openmx_edges_by_key = _edge_counts(snap_openmx)
    pyscf_total_edges = _total_edges(snap_pyscf)
    openmx_total_edges = _total_edges(snap_openmx)

    num_atoms = len(snap_pyscf.hamiltonian.atoms)
    expected_dense_shift_edges = nshift * num_atoms * num_atoms

    report = {
        "inputs": {
            "pyscf_npz": str(args.pyscf_npz),
            "pyscf_json": str(pyscf_json),
            "openmx_matrix": str(args.openmx_matrix),
            "openmx_info": str(args.openmx_info),
        },
        "pyscf": {
            "num_atoms": num_atoms,
            "nshift": nshift,
            "edge_count_total": pyscf_total_edges,
            "edge_count_by_key": pyscf_edges_by_key,
            "max_distance_angstrom": float(snap_pyscf.max_distance()),
            "expected_dense_shift_edges": int(expected_dense_shift_edges),
        },
        "openmx": {
            "edge_count_total": openmx_total_edges,
            "edge_count_by_key": openmx_edges_by_key,
            "max_distance_angstrom": float(snap_openmx.max_distance()),
        },
        "comparison": {
            "edge_count_delta": int(pyscf_total_edges - openmx_total_edges),
            "edge_count_ratio": float(pyscf_total_edges / openmx_total_edges),
            "distance_delta_angstrom": float(
                snap_pyscf.max_distance() - snap_openmx.max_distance()
            ),
        },
    }

    print(json.dumps(report, indent=2))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
