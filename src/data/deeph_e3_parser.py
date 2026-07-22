"""Strict loader for the public DeepH-E3 HDF5 dataset format.

DeepH-E3 stores OpenMX matrix blocks under ``[Rx, Ry, Rz, i, j]`` HDF5
keys. Atom indices are one-based and matrix values are in eV. The blocks are
still in OpenMX's real-spherical-harmonic order; conversion to Mandala's e3nn
basis must therefore happen exactly once after parsing.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from ase.data import chemical_symbols

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix

HARTREE_TO_EV = 27.2113845
REQUIRED_FILES = (
    "hamiltonians.h5",
    "element.dat",
    "orbital_types.dat",
    "site_positions.dat",
    "lat.dat",
    "info.json",
)


class DeepHE3ParseError(RuntimeError):
    """Raised when a DeepH-E3 snapshot violates the documented format."""


@dataclass(frozen=True)
class DeepHE3Metadata:
    snapshot_dir: Path
    atoms: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    orbital_types: tuple[tuple[int, ...], ...]
    orbital_set: dict[str, list[str]]
    positions: torch.Tensor
    box: torch.Tensor
    fermi_level: torch.Tensor | None
    is_spinful: bool
    available_matrices: frozenset[str]

    def as_snapshot_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            elements=list(self.atoms),
            orbital_set=self.orbital_set,
            fermi_level=self.fermi_level,
            is_spinful=self.is_spinful,
            available_matrices=self.available_matrices,
            source_format="deeph_e3",
            source_energy_unit="eV",
        )


def is_deeph_e3_snapshot(matrix_path: str | Path, info_path: str | Path) -> bool:
    matrix_path = Path(matrix_path)
    info_path = Path(info_path)
    return (
        matrix_path.name == "hamiltonians.h5"
        and info_path.name == "info.json"
        and matrix_path.parent == info_path.parent
        and all((matrix_path.parent / name).is_file() for name in REQUIRED_FILES)
    )


def _load_2d_text(path: Path, *, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(np.loadtxt(path, dtype=dtype))
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise DeepHE3ParseError(f"Expected a 2D array in {path}, got {array.shape}")
    return array


def _load_orbital_types(path: Path) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            tokens = raw_line.split()
            if not tokens:
                raise DeepHE3ParseError(f"Empty orbital row at {path}:{line_number}")
            try:
                row = tuple(int(token) for token in tokens)
            except ValueError as exc:
                raise DeepHE3ParseError(
                    f"Non-integer orbital type at {path}:{line_number}"
                ) from exc
            rows.append(row)
    return tuple(rows)


def _orbital_set_from_atoms(
    atoms: tuple[str, ...], orbital_types: tuple[tuple[int, ...], ...]
) -> dict[str, list[str]]:
    by_element: dict[str, tuple[int, ...]] = {}
    for atom, atom_orbitals in zip(atoms, orbital_types, strict=True):
        if any(l < 0 or l > 3 for l in atom_orbitals):
            raise DeepHE3ParseError(
                f"Element {atom} uses orbital angular momentum outside Mandala's "
                f"supported l=0..3 range: {atom_orbitals}"
            )
        previous = by_element.setdefault(atom, atom_orbitals)
        if previous != atom_orbitals:
            raise DeepHE3ParseError(
                f"Atoms of element {atom} have inconsistent orbital order: "
                f"{previous} versus {atom_orbitals}"
            )

    result: dict[str, list[str]] = {}
    for atom, ordered_ls in by_element.items():
        counts = Counter(ordered_ls)
        # Mandala's block basis groups orbital copies by l. Reject unusual
        # interleaving instead of silently changing the raw block order.
        grouped = tuple(l for l in sorted(counts) for _ in range(counts[l]))
        if ordered_ls != grouped:
            raise DeepHE3ParseError(
                f"Element {atom} has non-grouped orbital order {ordered_ls}; "
                f"Mandala requires {grouped}."
            )
        result[atom] = [
            f"{counts[l]}x{l}{'e' if l % 2 == 0 else 'o'}" for l in sorted(counts)
        ]
    return result


def load_deeph_e3_metadata(
    snapshot_dir: str | Path, *, dtype: torch.dtype = torch.float32
) -> DeepHE3Metadata:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (snapshot_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DeepH-E3 snapshot {snapshot_dir} is missing required files: {missing}"
        )

    atomic_numbers_np = np.atleast_1d(
        np.loadtxt(snapshot_dir / "element.dat", dtype=np.int64)
    )
    atomic_numbers = tuple(int(z) for z in atomic_numbers_np.tolist())
    if not atomic_numbers:
        raise DeepHE3ParseError(f"No atoms found in {snapshot_dir / 'element.dat'}")
    invalid_z = [z for z in atomic_numbers if z <= 0 or z >= len(chemical_symbols)]
    if invalid_z:
        raise DeepHE3ParseError(f"Invalid atomic numbers: {invalid_z}")
    atoms = tuple(chemical_symbols[z] for z in atomic_numbers)

    orbital_types = _load_orbital_types(snapshot_dir / "orbital_types.dat")
    if len(orbital_types) != len(atoms):
        raise DeepHE3ParseError(
            "orbital_types.dat row count does not match element.dat: "
            f"{len(orbital_types)} != {len(atoms)}"
        )
    orbital_set = _orbital_set_from_atoms(atoms, orbital_types)

    positions_raw = _load_2d_text(snapshot_dir / "site_positions.dat", dtype=np.float64)
    if positions_raw.shape == (3, len(atoms)):
        positions_raw = positions_raw.T
    elif positions_raw.shape != (len(atoms), 3):
        raise DeepHE3ParseError(
            "site_positions.dat must have shape (3,N) as written by DeepH-E3 "
            f"or (N,3), got {positions_raw.shape}"
        )
    # DeepH-E3 loads lat.dat with .T; Mandala stores lattice vectors as rows.
    lat_raw = np.asarray(np.loadtxt(snapshot_dir / "lat.dat"), dtype=np.float64)
    if lat_raw.shape != (3, 3):
        raise DeepHE3ParseError(f"lat.dat must have shape (3,3), got {lat_raw.shape}")
    box_raw = lat_raw.T

    with (snapshot_dir / "info.json").open() as handle:
        info_payload = json.load(handle)
    is_spinful = bool(info_payload.get("isspinful", False))
    if is_spinful:
        raise DeepHE3ParseError(
            "Spinful/SOC DeepH-E3 snapshots are not supported by Mandala's current "
            "real-valued orbital-block model."
        )
    fermi_ev = info_payload.get("fermi_level")
    fermi_level = (
        None
        if fermi_ev is None
        else torch.tensor(float(fermi_ev) / HARTREE_TO_EV, dtype=dtype)
    )
    available = {"hamiltonian"}
    if (snapshot_dir / "overlaps.h5").is_file():
        available.add("overlap")
    if (snapshot_dir / "density_matrixs.h5").is_file():
        available.add("density")

    return DeepHE3Metadata(
        snapshot_dir=snapshot_dir,
        atoms=atoms,
        atomic_numbers=atomic_numbers,
        orbital_types=orbital_types,
        orbital_set=orbital_set,
        positions=torch.tensor(positions_raw, dtype=dtype),
        box=torch.tensor(box_raw, dtype=dtype),
        fermi_level=fermi_level,
        is_spinful=is_spinful,
        available_matrices=frozenset(available),
    )


def _parse_hdf5_key(raw_key: str, *, num_atoms: int) -> tuple[int, int, int, int, int]:
    try:
        values = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError) as exc:
        raise DeepHE3ParseError(f"Invalid DeepH-E3 HDF5 key: {raw_key!r}") from exc
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise DeepHE3ParseError(
            f"DeepH-E3 HDF5 key must be [Rx,Ry,Rz,i,j], got {raw_key!r}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise DeepHE3ParseError(f"HDF5 key contains non-integers: {raw_key!r}")
    sx, sy, sz, src_one, dst_one = (int(value) for value in values)
    if not 1 <= src_one <= num_atoms or not 1 <= dst_one <= num_atoms:
        raise DeepHE3ParseError(
            f"DeepH-E3 atom indices are one-based; key out of range: {raw_key!r}"
        )
    return sx, sy, sz, src_one - 1, dst_one - 1


def load_deeph_e3_block_matrix(
    path: str | Path,
    metadata: DeepHE3Metadata,
    *,
    matrix_name: str,
    dtype: torch.dtype = torch.float32,
    validate_hermitian: bool = True,
) -> BlockMatrix:
    path = Path(path)
    orbital_cfg = OrbitalIrrepConfig.from_dict(metadata.orbital_set)
    entries: dict[tuple[int, int, int, int, int], torch.Tensor] = {}
    with h5py.File(path, "r") as handle:
        for raw_key, dataset in handle.items():
            edge = _parse_hdf5_key(raw_key, num_atoms=len(metadata.atoms))
            if edge in entries:
                raise DeepHE3ParseError(f"Duplicate edge {edge} in {path}")
            sx, sy, sz, src, dst = edge
            pair_key = f"{metadata.atoms[src]}-{metadata.atoms[dst]}"
            expected_shape = orbital_cfg.block_dims(pair_key)
            array = np.asarray(dataset)
            if np.iscomplexobj(array):
                raise DeepHE3ParseError(
                    f"Complex block {raw_key} in {path}; spinful matrices are unsupported."
                )
            if array.shape != expected_shape:
                raise DeepHE3ParseError(
                    f"Block {raw_key} in {path} has shape {array.shape}, expected "
                    f"{expected_shape} for {pair_key}. This usually indicates wrong "
                    "atom indexing or orbital order."
                )
            block = torch.tensor(array, dtype=dtype)
            if matrix_name == "hamiltonian":
                block = block / HARTREE_TO_EV
            entries[edge] = block

    if not entries:
        raise DeepHE3ParseError(f"No matrix blocks found in {path}")
    for atom_idx in range(len(metadata.atoms)):
        edge = (0, 0, 0, atom_idx, atom_idx)
        if edge not in entries:
            raise DeepHE3ParseError(f"Missing zero-shift self block {edge} in {path}")
    for edge, block in entries.items():
        sx, sy, sz, src, dst = edge
        reverse = (-sx, -sy, -sz, dst, src)
        if reverse not in entries:
            raise DeepHE3ParseError(
                f"Edge {edge} in {path} has no reciprocal {reverse}"
            )
        if validate_hermitian and not torch.allclose(
            block, entries[reverse].T, rtol=1.0e-5, atol=1.0e-8
        ):
            error = float(torch.max(torch.abs(block - entries[reverse].T)).item())
            raise DeepHE3ParseError(
                f"Reciprocal blocks {edge} and {reverse} are not transposes in {path}; "
                f"max_abs_error={error:.6e}"
            )

    grouped: dict[str, list[tuple[tuple[int, int, int, int, int], torch.Tensor]]] = {}
    for edge, block in entries.items():
        src, dst = edge[3], edge[4]
        pair_key = f"{metadata.atoms[src]}-{metadata.atoms[dst]}"
        grouped.setdefault(pair_key, []).append((edge, block))

    pair_blocks: dict[str, torch.Tensor] = {}
    pair_edges: dict[str, torch.Tensor] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}
    for pair_key in sorted(grouped):
        ordered = sorted(grouped[pair_key], key=lambda item: item[0])
        pair_blocks[pair_key] = torch.stack([block for _, block in ordered])
        pair_edges[pair_key] = torch.tensor(
            [edge for edge, _ in ordered], dtype=torch.long
        ).T.contiguous()
        for local_idx, (edge, _) in enumerate(ordered):
            lookup[edge] = (pair_key, local_idx)

    return BlockMatrix(
        atoms=metadata.atoms,
        atom_counts=Counter(metadata.atoms),
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis="openmx",
    )


def zero_matrix_on_support(matrix: BlockMatrix) -> BlockMatrix:
    """Create an explicit infrastructure placeholder on identical edge support."""
    return matrix._replace_pair_blocks(
        {key: torch.zeros_like(blocks) for key, blocks in matrix.pair_blocks.items()},
        basis=matrix.basis,
    )
