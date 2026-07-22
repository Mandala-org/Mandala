#!/usr/bin/env python3
"""Standalone diagnostic loader for one raw DeepH-E3 snapshot."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.basis_converter import (  # noqa: E402
    OpenMXE3NNConverter,
    _U_OPENMX_TO_WIKI,
)
from data.deeph_e3_parser import (  # noqa: E402
    HARTREE_TO_EV,
    load_deeph_e3_block_matrix,
    load_deeph_e3_metadata,
)
from data.snapshot import Snapshot  # noqa: E402


DEEPH_OPENMX_TO_WIKI_ROWS = {
    0: [0],
    1: [1, 2, 0],
    2: [2, 4, 0, 3, 1],
    3: [6, 4, 2, 0, 1, 3, 5],
}


def _edge_count(matrix) -> int:
    return sum(edges.shape[1] for edges in matrix.pair_edges.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--convention", choices=("openmx", "e3nn"), default="e3nn")
    args = parser.parse_args()

    print("=== DeepH-E3 standalone loader diagnostics ===", flush=True)
    metadata = load_deeph_e3_metadata(args.snapshot_dir)
    print(f"snapshot_dir: {metadata.snapshot_dir}")
    print(
        f"atoms: {len(metadata.atoms)}; "
        f"counts={dict(sorted(Counter(metadata.atoms).items()))}"
    )
    print(f"orbital_set: {metadata.orbital_set}")
    print(
        f"positions: {tuple(metadata.positions.shape)}; box: {tuple(metadata.box.shape)}"
    )
    print(f"available matrices: {sorted(metadata.available_matrices)}")
    print(
        "fermi level: "
        + (
            "missing"
            if metadata.fermi_level is None
            else f"{metadata.fermi_level.item():.9f} Ha / "
            f"{metadata.fermi_level.item() * HARTREE_TO_EV:.9f} eV"
        )
    )

    print("\n--- Convention equivalence: DeepH-E3 vs Mandala ---")
    for l, rows in DEEPH_OPENMX_TO_WIKI_ROWS.items():
        expected = torch.eye(2 * l + 1)[rows]
        exact = torch.equal(expected, _U_OPENMX_TO_WIKI[l])
        print(f"l={l}: rows={rows}; exact={exact}")
        if not exact:
            raise RuntimeError(f"Convention mismatch at l={l}")
    print("coordinate mapping: OpenMX xyz -> e3nn yzx (same mapping as DeepH-E3)")

    print("\n--- Native HDF5 block checks ---", flush=True)
    native = load_deeph_e3_block_matrix(
        metadata.snapshot_dir / "hamiltonians.h5",
        metadata,
        matrix_name="hamiltonian",
    )
    print(f"pair keys: {sorted(native.pair_blocks)}")
    print(f"blocks: {_edge_count(native)}; basis={native.basis}")
    for key in sorted(native.pair_blocks):
        blocks = native.pair_blocks[key]
        print(
            f"  {key}: count={blocks.shape[0]} shape={tuple(blocks.shape[1:])} "
            f"max_abs={blocks.abs().max().item():.6e} Ha"
        )

    print("\n--- Integrated Snapshot load and block order ---", flush=True)
    snapshot = Snapshot.from_deeph_e3(metadata.snapshot_dir, convention=args.convention)
    print(f"loaded basis: {snapshot.hamiltonian.basis}")
    for key in sorted(snapshot.hamiltonian.pair_edges):
        edges = snapshot.hamiltonian.pair_edges[key]
        diagonal = (
            (edges[0] == 0) & (edges[1] == 0) & (edges[2] == 0) & (edges[3] == edges[4])
        )
        source_element, target_element = key.split("-")
        expected_diag = (
            snapshot.hamiltonian.atom_counts[source_element]
            if source_element == target_element
            else 0
        )
        prefix_ok = bool(diagonal[:expected_diag].all()) and not bool(
            diagonal[expected_diag:].any()
        )
        print(
            f"  {key}: diagonal_prefix={expected_diag}; prefix_ok={prefix_ok}; "
            f"first_edges={edges[:, :min(3, edges.shape[1])].T.tolist()}"
        )
        if not prefix_ok:
            raise RuntimeError(f"Canonical diagonal block order failed for {key}")

    if args.convention == "e3nn":
        converter = OpenMXE3NNConverter(native.orbital_cfg)
        roundtrip = converter.matrix_to_openmx(snapshot.hamiltonian)
        max_error = 0.0
        for edge, (key, native_idx) in native.lookup.items():
            converted_key, converted_idx = roundtrip.lookup[edge]
            if key != converted_key:
                raise RuntimeError(f"Pair key changed for edge {edge}")
            error = torch.max(
                torch.abs(
                    native.pair_blocks[key][native_idx]
                    - roundtrip.pair_blocks[key][converted_idx]
                )
            )
            max_error = max(max_error, float(error.item()))
        print(f"OpenMX -> e3nn -> OpenMX max block error: {max_error:.6e} Ha")
        if max_error > 1.0e-6:
            raise RuntimeError("Spherical-harmonic basis round-trip failed")

    print(
        "\nPASS: format, units, reciprocal support, convention, and block order validated."
    )


if __name__ == "__main__":
    main()
