"""Export Mandala/OpenMX snapshots to DeepH-E3's processed HDF5 format."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from ase.data import atomic_numbers

from data.deeph_e3_parser import (
    HARTREE_TO_EV,
    load_deeph_e3_block_matrix,
    load_deeph_e3_metadata,
)
from data.snapshot import Snapshot


@dataclass(frozen=True)
class DeepHE3ExportSummary:
    """Numerical summary returned after one snapshot is exported and checked."""

    source_dir: str
    output_dir: str
    num_atoms: int
    num_hamiltonian_blocks: int
    num_overlap_blocks: int
    num_density_blocks: int
    hamiltonian_roundtrip_max_abs_ha: float
    overlap_roundtrip_max_abs: float | None
    density_roundtrip_max_abs: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ordered_edges(matrix) -> list[tuple[tuple[int, int, int, int, int], torch.Tensor]]:
    entries: list[tuple[tuple[int, int, int, int, int], torch.Tensor]] = []
    for pair_key in sorted(matrix.pair_edges):
        edges = matrix.pair_edges[pair_key]
        blocks = matrix.pair_blocks[pair_key]
        if edges.ndim != 2 or edges.shape[0] != 5:
            raise ValueError(
                f"Matrix edge tensor for {pair_key} must have shape (5,E), got "
                f"{tuple(edges.shape)}"
            )
        if edges.shape[1] != blocks.shape[0]:
            raise ValueError(
                f"Matrix edge/block count mismatch for {pair_key}: "
                f"{edges.shape[1]} != {blocks.shape[0]}"
            )
        for local_idx in range(edges.shape[1]):
            edge = tuple(int(value) for value in edges[:, local_idx].tolist())
            entries.append((edge, blocks[local_idx]))
    entries.sort(key=lambda item: item[0])
    return entries


def _write_hdf5_matrix(path: Path, matrix, *, scale: float) -> int:
    if matrix.basis != "openmx":
        raise ValueError(
            f"DeepH-E3 export requires OpenMX-basis blocks, got {matrix.basis!r}"
        )
    entries = _ordered_edges(matrix)
    with h5py.File(path, "w") as handle:
        for (sx, sy, sz, src, dst), block in entries:
            # DeepH-E3 stores one-based atom indices in JSON-list HDF5 keys.
            key = str([sx, sy, sz, src + 1, dst + 1])
            # Promote before unit conversion. OpenMX parsing currently yields
            # float32 blocks; multiplying those in float32 would add avoidable
            # ~1e-7 Ha round-trip error even though HDF5 is written as float64.
            array = np.asarray(block.detach().cpu().numpy(), dtype=np.float64)
            handle.create_dataset(key, data=array * float(scale))
    return len(entries)


def _orbital_rows(snapshot: Snapshot) -> list[list[int]]:
    cfg = snapshot.hamiltonian.orbital_cfg
    rows: list[list[int]] = []
    for atom in snapshot.hamiltonian.atoms:
        if atom not in cfg.element_to_irreps:
            raise ValueError(f"No orbital configuration found for element {atom}")
        row: list[int] = []
        for multiplicity, irrep in cfg.element_to_irreps[atom]:
            if irrep.p != (-1) ** irrep.l:
                raise ValueError(
                    f"Element {atom} has non-orbital parity {irrep}; DeepH-E3's "
                    "orbital_types.dat cannot represent it"
                )
            row.extend([int(irrep.l)] * int(multiplicity))
        if not row:
            raise ValueError(f"Element {atom} has no orbitals")
        rows.append(row)
    return rows


def _write_text_metadata(output_dir: Path, snapshot: Snapshot) -> None:
    atoms = tuple(snapshot.hamiltonian.atoms)
    positions = snapshot.positions
    box = snapshot.box
    if positions is None or tuple(positions.shape) != (len(atoms), 3):
        raise ValueError(
            f"Snapshot positions must have shape ({len(atoms)},3), got "
            f"{None if positions is None else tuple(positions.shape)}"
        )
    if box is None or tuple(box.shape) != (3, 3):
        raise ValueError(
            f"Snapshot box must have shape (3,3), got "
            f"{None if box is None else tuple(box.shape)}"
        )
    if snapshot.info is None or snapshot.info.fermi_level is None:
        raise ValueError("OpenMX snapshot does not provide a Fermi level")

    try:
        numbers = np.asarray([atomic_numbers[atom] for atom in atoms], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown chemical element {exc.args[0]!r}") from exc

    # DeepH-E3 writes lattice vectors and Cartesian positions as columns.
    lat = box.detach().cpu().numpy().astype(np.float64, copy=False).T
    site_positions = positions.detach().cpu().numpy().astype(np.float64, copy=False).T
    reciprocal_lat = 2.0 * np.pi * np.linalg.inv(lat).T

    np.savetxt(output_dir / "element.dat", numbers, fmt="%d")
    np.savetxt(output_dir / "lat.dat", lat, fmt="%.17g")
    np.savetxt(output_dir / "rlat.dat", reciprocal_lat, fmt="%.17g")
    np.savetxt(output_dir / "site_positions.dat", site_positions, fmt="%.17g")
    with (output_dir / "orbital_types.dat").open("w") as handle:
        for row in _orbital_rows(snapshot):
            handle.write(" ".join(str(value) for value in row) + "\n")

    shifts = sorted(
        {
            tuple(int(value) for value in edge[:3])
            for edge, _ in _ordered_edges(snapshot.hamiltonian)
        }
    )
    if not shifts:
        raise ValueError("Hamiltonian contains no periodic shifts")
    np.savetxt(output_dir / "R_list.dat", np.asarray(shifts), fmt="%d")

    fermi_ha = float(snapshot.info.fermi_level.detach().cpu().item())
    with (output_dir / "info.json").open("w") as handle:
        json.dump(
            {"fermi_level": fermi_ha * HARTREE_TO_EV, "isspinful": False},
            handle,
            indent=4,
            sort_keys=True,
        )
        handle.write("\n")


def _matrix_roundtrip_max_abs(reference, loaded) -> float:
    reference_edges = set(reference.lookup)
    loaded_edges = set(loaded.lookup)
    if reference_edges != loaded_edges:
        missing = sorted(reference_edges - loaded_edges)[:5]
        extra = sorted(loaded_edges - reference_edges)[:5]
        raise ValueError(
            "DeepH-E3 round-trip changed matrix support: "
            f"missing={missing} extra={extra}"
        )
    max_abs = 0.0
    for edge in reference_edges:
        ref_key, ref_idx = reference.lookup[edge]
        got_key, got_idx = loaded.lookup[edge]
        if ref_key != got_key:
            raise ValueError(
                f"DeepH-E3 round-trip changed pair key for {edge}: "
                f"{ref_key} != {got_key}"
            )
        error = torch.max(
            torch.abs(
                reference.pair_blocks[ref_key][ref_idx]
                - loaded.pair_blocks[got_key][got_idx]
            )
        )
        max_abs = max(max_abs, float(error.item()))
    return max_abs


def _validate_export(
    output_dir: Path,
    snapshot: Snapshot,
    *,
    include_overlap: bool,
    include_density: bool,
) -> tuple[float, float | None, float | None]:
    metadata = load_deeph_e3_metadata(output_dir, dtype=torch.float64)
    atoms = tuple(snapshot.hamiltonian.atoms)
    if metadata.atoms != atoms:
        raise ValueError(
            f"Atom order changed during export: {metadata.atoms} != {atoms}"
        )
    if not torch.equal(metadata.positions, snapshot.positions.cpu()):
        error = torch.max(torch.abs(metadata.positions - snapshot.positions.cpu()))
        raise ValueError(f"Position round-trip mismatch: max_abs={float(error):.6e}")
    if not torch.equal(metadata.box, snapshot.box.cpu()):
        error = torch.max(torch.abs(metadata.box - snapshot.box.cpu()))
        raise ValueError(f"Lattice round-trip mismatch: max_abs={float(error):.6e}")

    loaded_h = load_deeph_e3_block_matrix(
        output_dir / "hamiltonians.h5",
        metadata,
        matrix_name="hamiltonian",
        dtype=torch.float64,
    )
    h_error = _matrix_roundtrip_max_abs(snapshot.hamiltonian, loaded_h)

    overlap_error = None
    if include_overlap:
        loaded_s = load_deeph_e3_block_matrix(
            output_dir / "overlaps.h5",
            metadata,
            matrix_name="overlap",
            dtype=torch.float64,
        )
        overlap_error = _matrix_roundtrip_max_abs(snapshot.overlap, loaded_s)

    density_error = None
    if include_density:
        loaded_d = load_deeph_e3_block_matrix(
            output_dir / "density_matrixs.h5",
            metadata,
            matrix_name="density",
            dtype=torch.float64,
        )
        density_error = _matrix_roundtrip_max_abs(snapshot.density, loaded_d)

    tolerance = 1.0e-12
    errors = [h_error]
    errors.extend(
        error for error in (overlap_error, density_error) if error is not None
    )
    if any(error > tolerance for error in errors):
        raise ValueError(
            f"DeepH-E3 round-trip exceeded {tolerance:.1e}: errors={errors}"
        )
    return h_error, overlap_error, density_error


def export_openmx_snapshot_to_deeph_e3(
    matrix_path: str | Path,
    info_path: str | Path,
    output_dir: str | Path,
    *,
    include_overlap: bool = True,
    include_density: bool = False,
    validate: bool = True,
) -> DeepHE3ExportSummary:
    """Convert one OpenMX snapshot and atomically publish the checked result."""

    matrix_path = Path(matrix_path).expanduser().resolve()
    info_path = Path(info_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not matrix_path.is_file():
        raise FileNotFoundError(f"OpenMX matrix file not found: {matrix_path}")
    if not info_path.is_file():
        raise FileNotFoundError(f"OpenMX info file not found: {info_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output snapshot already exists: {output_dir}")

    # Keep raw OpenMX orbital order and full real-space support. DeepH-E3 uses
    # the same basis, while its HDF5 Hamiltonian values are expressed in eV.
    snapshot = Snapshot.from_openmx(
        matrix_path,
        info_path,
        convention="openmx",
        symmetrize_density=False,
        dtype=torch.float64,
    )
    temp_dir = output_dir.with_name(
        f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_text_metadata(temp_dir, snapshot)
        num_h = _write_hdf5_matrix(
            temp_dir / "hamiltonians.h5",
            snapshot.hamiltonian,
            scale=HARTREE_TO_EV,
        )
        num_s = 0
        if include_overlap:
            num_s = _write_hdf5_matrix(
                temp_dir / "overlaps.h5", snapshot.overlap, scale=1.0
            )
        num_d = 0
        if include_density:
            num_d = _write_hdf5_matrix(
                temp_dir / "density_matrixs.h5", snapshot.density, scale=1.0
            )

        if validate:
            h_error, s_error, d_error = _validate_export(
                temp_dir,
                snapshot,
                include_overlap=include_overlap,
                include_density=include_density,
            )
        else:
            h_error, s_error, d_error = float("nan"), None, None

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return DeepHE3ExportSummary(
        source_dir=str(matrix_path.parent),
        output_dir=str(output_dir),
        num_atoms=len(snapshot.hamiltonian.atoms),
        num_hamiltonian_blocks=num_h,
        num_overlap_blocks=num_s,
        num_density_blocks=num_d,
        hamiltonian_roundtrip_max_abs_ha=h_error,
        overlap_roundtrip_max_abs=s_error,
        density_roundtrip_max_abs=d_error,
    )
