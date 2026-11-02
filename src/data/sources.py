"""One format boundary for raw OpenMX, processed HDF5 and PySCF snapshots.

Source format describes storage, never a neural architecture. DatasetFactory
and every training configuration use this same discovery/metadata/load path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnapshotSource:
    matrix_path: Path
    info_path: Path
    format: str


def openmx_info_path(directory):
    directory = Path(directory)
    for name in ("Si.out", "SiO2.out", "ZnCuSeS.out", "info.dat", "info.txt"):
        if (directory / name).is_file():
            return directory / name
    candidates = sorted(p for p in directory.glob("*.out") if p.name != "HS.out")
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous OpenMX metadata under {directory}; provide info_path"
        )
    return candidates[0] if candidates else None


def resolve_snapshot_source(matrix_path, info_path=None, *, matrix_files=None):
    from data.deeph_e3_parser import is_deeph_e3_snapshot

    matrix = Path(matrix_path)
    if matrix.is_dir():
        if info_path is not None:
            raise ValueError("A snapshot directory must not also specify info_path")
        native = matrix / (matrix_files or {}).get("hamiltonian", "hamiltonians.h5")
        if (matrix / "HS.out").is_file() and native.is_file():
            raise ValueError(
                f"Ambiguous raw/processed snapshot: {matrix}; provide files"
            )
        if native.is_file():
            matrix, info_path = native, matrix / "info.json"
        elif (matrix / "HS.out").is_file():
            info_path = openmx_info_path(matrix)
            matrix = matrix / "HS.out"
        else:
            candidates = sorted(matrix.glob("*.npz"))
            if len(candidates) == 1:
                matrix = candidates[0]
                info_path = matrix.with_suffix(".json")
                if not info_path.is_file():
                    metadata = sorted(matrix.parent.glob("*.json"))
                    if len(metadata) != 1:
                        raise ValueError(
                            f"Ambiguous/missing PySCF metadata under {matrix.parent}"
                        )
                    info_path = metadata[0]
    if info_path is None:
        raise FileNotFoundError(f"Cannot resolve snapshot metadata: {matrix}")
    info = Path(info_path)
    # Explicit pairs retain the factory's lazy-registration behavior. Actual
    # parsers validate file availability when the dataset is created.
    if is_deeph_e3_snapshot(matrix, info):
        kind = "periodic_hdf5"
    elif matrix.suffix == ".h5":
        raise ValueError(f"Incomplete periodic HDF5 snapshot: {matrix.parent}")
    elif matrix.suffix == ".npz" or info.suffix == ".json":
        kind = "pyscf"
    else:
        kind = "openmx"
    return SnapshotSource(matrix, info, kind)


def load_source_metadata(source, *, dtype, options=None):
    options = options or {}
    if source.format == "periodic_hdf5":
        from data.deeph_e3_parser import load_deeph_e3_metadata

        return load_deeph_e3_metadata(
            source.info_path.parent,
            dtype=dtype,
            matrix_files=options.get("matrix_files"),
            source_basis=options.get("source_basis"),
            source_cartesian_axes=options.get("source_cartesian_axes"),
        )
    if source.format == "pyscf":
        from data.pyscf_baseline_parser import load_pyscf_metadata

        return load_pyscf_metadata(source.info_path)
    from data.openmx_info_parser import parse_info_out

    return parse_info_out(source.info_path, dtype)


def load_source_snapshot(source, *, cfg, convention, dtype):
    from data.snapshot import Snapshot

    if source.format == "periodic_hdf5":
        return Snapshot.from_deeph_e3(
            source.info_path.parent,
            convention=convention,
            dtype=dtype,
            cfg=cfg,
            matrix_files=cfg.data.get("matrix_files"),
            source_basis=cfg.data.get("source_basis"),
            source_cartesian_axes=cfg.data.get("source_cartesian_axes"),
            validate_hermitian=cfg.data.get("validate_hermitian", True),
            validate_support=cfg.data.get("validate_support", True),
            canonicalize=cfg.data.get("canonicalize_edges", True),
            symmetrize_density=cfg.symmetrize_density_targets,
        )
    if source.format == "pyscf":
        return Snapshot.from_pyscf(
            npz_path=source.matrix_path,
            json_path=source.info_path,
            convention=convention,
            dtype=dtype,
            cfg=cfg,
        )
    return Snapshot.from_openmx(
        matrix_path=source.matrix_path,
        info_path=source.info_path,
        convention=convention,
        dtype=dtype,
        cfg=cfg,
        symmetrize_density=cfg.symmetrize_density_targets,
    )
