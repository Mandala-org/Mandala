"""Loader/parser utilities for PySCF baseline Hamiltonian artifacts.

Expected input comes from ``data/pyscf_baseline/calc_pyscf_baseline.py``:
- ``<run>.json``
- ``<run>.npz``
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot

__all__ = [
    "PySCFBaselineData",
    "PySCFMetadataInfo",
    "load_pyscf_metadata",
    "load_pyscf_baseline",
    "load_pyscf_snapshot",
]


def _resolve_json_path(npz_path: Path, json_path: str | Path | None) -> Path | None:
    if json_path is not None:
        p = Path(json_path)
        return p
    candidate = npz_path.with_suffix(".json")
    return candidate if candidate.exists() else None


def _load_metadata(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {path}")
    return json.loads(path.read_text())


@dataclass(slots=True)
class PySCFMetadataInfo:
    """Lightweight metadata container for dataset/factory integration."""

    metadata: Dict[str, Any]
    orbital_set: Dict[str, str]


def load_pyscf_metadata(path: str | Path) -> PySCFMetadataInfo:
    """Load PySCF JSON metadata and expose the orbital set like InfoOutData."""
    path_obj = Path(path)
    metadata = _load_metadata(path_obj)
    orbital_set = metadata.get("settings", {}).get("orbital_set")
    if orbital_set is None:
        orbital_set = metadata.get("openmx_reference", {}).get("orbital_set")
    if orbital_set is None:
        raise ValueError(f"Could not infer orbital_set from metadata JSON: {path_obj}")
    return PySCFMetadataInfo(metadata=metadata, orbital_set=dict(orbital_set))


def _extract_atoms(
    metadata: Dict[str, Any], atoms: Iterable[str] | None
) -> Tuple[str, ...]:
    if atoms is not None:
        ret = tuple(str(a) for a in atoms)
        if not ret:
            raise ValueError("atoms override is empty")
        return ret

    rows = metadata.get("snapshot", {}).get("atoms")
    if not rows:
        raise ValueError(
            "Could not infer atoms. Provide `atoms=` or include metadata JSON."
        )

    ret = tuple(str(row["element"]) for row in rows)
    if not ret:
        raise ValueError("No atoms found in metadata")
    return ret


def _extract_orbital_cfg(
    metadata: Dict[str, Any],
    orbital_cfg: OrbitalIrrepConfig | None,
    orbital_set: Dict[str, str] | None,
) -> OrbitalIrrepConfig:
    if orbital_cfg is not None:
        return orbital_cfg

    if orbital_set is not None:
        return OrbitalIrrepConfig.from_dict(orbital_set)

    json_orbital_set = metadata.get("settings", {}).get("orbital_set")
    if json_orbital_set is None:
        json_orbital_set = metadata.get("openmx_reference", {}).get("orbital_set")
    if json_orbital_set is None:
        raise ValueError(
            "Could not infer orbital_set. Provide `orbital_set=`/`orbital_cfg=` or metadata JSON."
        )
    return OrbitalIrrepConfig.from_dict(json_orbital_set)


def _extract_positions(
    metadata: Dict[str, Any],
    npz_payload: np.lib.npyio.NpzFile,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor | None:
    if "positions_angstrom" in npz_payload:
        arr = np.asarray(npz_payload["positions_angstrom"], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    rows = metadata.get("snapshot", {}).get("atoms")
    if not rows:
        return None
    pos = [
        [float(r["x_angstrom"]), float(r["y_angstrom"]), float(r["z_angstrom"])]
        for r in rows
    ]
    return torch.tensor(pos, dtype=dtype, device=device)


def _extract_box(
    metadata: Dict[str, Any],
    npz_payload: np.lib.npyio.NpzFile,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor | None:
    if "box_angstrom" in npz_payload:
        arr = np.asarray(npz_payload["box_angstrom"], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    box = metadata.get("snapshot", {}).get("box_angstrom")
    if box is None:
        return None
    return torch.tensor(box, dtype=dtype, device=device)


def _extract_forces(
    metadata: Dict[str, Any],
    npz_payload: np.lib.npyio.NpzFile,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor | None:
    if "forces_ev_per_angstrom" in npz_payload:
        arr = np.asarray(npz_payload["forces_ev_per_angstrom"], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)
    if "forces_hartree_per_bohr" in npz_payload:
        arr = np.asarray(npz_payload["forces_hartree_per_bohr"], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    forces = metadata.get("pyscf", {}).get("forces_ev_per_angstrom")
    if forces is not None:
        return torch.tensor(forces, dtype=dtype, device=device)
    forces = metadata.get("pyscf", {}).get("forces_hartree_per_bohr")
    if forces is None:
        return None
    return torch.tensor(forces, dtype=dtype, device=device)


def _extract_stress(
    metadata: Dict[str, Any],
    npz_payload: np.lib.npyio.NpzFile,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor | None:
    if "stress_ev_per_angstrom3" in npz_payload:
        arr = np.asarray(npz_payload["stress_ev_per_angstrom3"], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)
    if "stress_hartree_per_angstrom3" in npz_payload:
        arr = np.asarray(npz_payload["stress_hartree_per_angstrom3"], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    stress = metadata.get("pyscf", {}).get("stress_ev_per_angstrom3")
    if stress is not None:
        return torch.tensor(stress, dtype=dtype, device=device)
    stress = metadata.get("pyscf", {}).get("stress_hartree_per_angstrom3")
    if stress is None:
        return None
    return torch.tensor(stress, dtype=dtype, device=device)


def _pick_hamiltonian(npz_payload: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("hamiltonian_ao", "fock_ao", "hcore_ao"):
        if key in npz_payload:
            return np.asarray(npz_payload[key], dtype=np.float32)
    raise KeyError(
        "No Hamiltonian found in NPZ. Expected one of: hamiltonian_ao, fock_ao, hcore_ao."
    )


def _validate_dense_square(name: str, matrix: torch.Tensor) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be rank-2; got shape {tuple(matrix.shape)}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square; got shape {tuple(matrix.shape)}")


def _validate_shifted(name: str, matrix: torch.Tensor, nshift: int, nao: int) -> None:
    if matrix.ndim != 3:
        raise ValueError(f"{name} must be rank-3; got shape {tuple(matrix.shape)}")
    if matrix.shape[0] != nshift or matrix.shape[1] != nao or matrix.shape[2] != nao:
        raise ValueError(
            f"{name} shape mismatch: expected ({nshift}, {nao}, {nao}), got {tuple(matrix.shape)}"
        )


def _atom_offsets(
    atoms: Tuple[str, ...], orbital_cfg: OrbitalIrrepConfig
) -> torch.Tensor:
    dims = [orbital_cfg.block_dims(f"{el}-{el}")[0] for el in atoms]
    offsets = [0]
    for d in dims[:-1]:
        offsets.append(offsets[-1] + int(d))
    return torch.tensor(offsets, dtype=torch.long)


def _block_matrix_from_shifted_dense(
    shifted: torch.Tensor,
    shifts: torch.Tensor,
    atoms: Tuple[str, ...],
    orbital_cfg: OrbitalIrrepConfig,
    *,
    basis: str,
) -> BlockMatrix:
    offsets = _atom_offsets(atoms, orbital_cfg)
    edge_blocks: dict[tuple[int, int, int, int, int], torch.Tensor] = {}

    def _register_edge(
        edge: tuple[int, int, int, int, int], block: torch.Tensor
    ) -> None:
        existing = edge_blocks.get(edge)
        if existing is None:
            edge_blocks[edge] = block
            return
        if not torch.allclose(existing, block, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Inconsistent duplicate PySCF block for edge {edge}")

    for s_idx in range(int(shifts.shape[0])):
        sx, sy, sz = [int(v) for v in shifts[s_idx].tolist()]
        mat = shifted[s_idx]  # (nao,nao)

        for i, el_i in enumerate(atoms):
            di = orbital_cfg.block_dims(f"{el_i}-{el_i}")[0]
            r0 = int(offsets[i])
            for j, el_j in enumerate(atoms):
                dj = orbital_cfg.block_dims(f"{el_j}-{el_j}")[0]
                c0 = int(offsets[j])
                blk = mat[r0 : r0 + di, c0 : c0 + dj].clone()
                edge = (sx, sy, sz, i, j)
                _register_edge(edge, blk)
                reverse_edge = (-sx, -sy, -sz, j, i)
                _register_edge(reverse_edge, blk.T.clone())

    pair_blocks: dict[str, list[torch.Tensor]] = {}
    pair_edges: dict[str, list[list[int]]] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}
    for edge, blk in sorted(edge_blocks.items()):
        sx, sy, sz, i, j = edge
        key = f"{atoms[i]}-{atoms[j]}"
        if key not in pair_blocks:
            pair_blocks[key] = []
            pair_edges[key] = []
        local_idx = len(pair_blocks[key])
        pair_blocks[key].append(blk)
        pair_edges[key].append([sx, sy, sz, i, j])
        lookup[edge] = (key, local_idx)

    pair_blocks_t = {k: torch.stack(v, dim=0) for k, v in pair_blocks.items()}
    pair_edges_t = {
        k: torch.tensor(v, dtype=torch.long, device=shifted.device).t()
        for k, v in pair_edges.items()
    }

    all_edges = set()
    for edges in pair_edges_t.values():
        for edge in edges.t().tolist():
            all_edges.add(tuple(edge))
    total_edges = sum(int(edges.shape[1]) for edges in pair_edges_t.values())
    if len(all_edges) != total_edges:
        raise ValueError("Duplicate edges found while rebuilding PySCF shift blocks")
    if len(lookup) != total_edges:
        raise ValueError("Lookup size does not match PySCF edge count")
    for i, atom in enumerate(atoms):
        key = f"{atom}-{atom}"
        if key not in pair_blocks_t:
            raise ValueError(f"Missing self-edge key {key} in PySCF block matrix")
        if (0, 0, 0, i, i) not in lookup:
            raise ValueError(f"Missing self-edge lookup entry for atom {i}")
    for sx, sy, sz, i, j in lookup:
        if (-sx, -sy, -sz, j, i) not in lookup:
            raise ValueError(
                "PySCF shift graph is not symmetric for edge " f"{(sx, sy, sz, i, j)}"
            )

    return BlockMatrix(
        atoms=atoms,
        atom_counts=Counter(atoms),
        pair_blocks=pair_blocks_t,
        pair_edges=pair_edges_t,
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis=basis,
    )


@dataclass(slots=True)
class PySCFBaselineData:
    """In-memory representation of a parsed PySCF baseline artifact set."""

    npz_path: Path
    json_path: Path | None
    metadata: Dict[str, Any]
    atoms: Tuple[str, ...]
    orbital_cfg: OrbitalIrrepConfig
    hamiltonian_ao: torch.Tensor
    overlap_ao: torch.Tensor
    density_ao: torch.Tensor
    shifts: torch.Tensor | None
    hamiltonian_shifted: torch.Tensor | None
    overlap_shifted: torch.Tensor | None
    density_shifted: torch.Tensor | None
    positions: torch.Tensor | None
    box: torch.Tensor | None
    forces: torch.Tensor | None
    stress: torch.Tensor | None

    def to_snapshot(self, *, basis: str = "pyscf") -> Snapshot:
        """Convert parsed matrices to a project-native ``Snapshot``."""
        if self.shifts is None:
            raise ValueError("PBC-only loader requires `shifts` in the NPZ payload.")
        if self.hamiltonian_shifted is None:
            raise ValueError(
                "PBC-only loader requires `hamiltonian_shifted` in the NPZ payload."
            )
        if self.overlap_shifted is None:
            raise ValueError(
                "PBC-only loader requires `overlap_shifted` in the NPZ payload."
            )
        if self.density_shifted is None:
            raise ValueError(
                "PBC-only loader requires `density_shifted` in the NPZ payload."
            )

        ham = _block_matrix_from_shifted_dense(
            self.hamiltonian_shifted,
            self.shifts,
            self.atoms,
            self.orbital_cfg,
            basis=basis,
        )
        ovl = _block_matrix_from_shifted_dense(
            self.overlap_shifted,
            self.shifts,
            self.atoms,
            self.orbital_cfg,
            basis=basis,
        )
        den = _block_matrix_from_shifted_dense(
            self.density_shifted,
            self.shifts,
            self.atoms,
            self.orbital_cfg,
            basis=basis,
        )
        return Snapshot(
            hamiltonian=ham,
            overlap=ovl,
            density=den,
            positions=self.positions,
            forces=self.forces,
            box=self.box,
            stress=self.stress,
            matrix_path=self.npz_path,
            info_path=self.json_path,
            info=self.metadata,
        )


def load_pyscf_baseline(
    npz_path: str | Path,
    *,
    json_path: str | Path | None = None,
    atoms: Iterable[str] | None = None,
    orbital_set: Dict[str, str] | None = None,
    orbital_cfg: OrbitalIrrepConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> PySCFBaselineData:
    """Load PySCF baseline files into a structured dataclass."""
    npz_path_obj = Path(npz_path)
    if not npz_path_obj.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path_obj}")

    json_path_obj = _resolve_json_path(npz_path_obj, json_path)
    metadata = _load_metadata(json_path_obj)
    atoms_tuple = _extract_atoms(metadata, atoms)
    orb_cfg = _extract_orbital_cfg(metadata, orbital_cfg, orbital_set)

    with np.load(npz_path_obj, allow_pickle=False) as npz_payload:
        ham_arr = _pick_hamiltonian(npz_payload)
        if "overlap_ao" not in npz_payload:
            raise KeyError("Missing overlap_ao in NPZ payload")
        if "dm_ao" not in npz_payload:
            raise KeyError("Missing dm_ao in NPZ payload")

        overlap_arr = np.asarray(npz_payload["overlap_ao"], dtype=np.float32)
        density_arr = np.asarray(npz_payload["dm_ao"], dtype=np.float32)
        shifts_arr = (
            np.asarray(npz_payload["shifts"], dtype=np.int64)
            if "shifts" in npz_payload
            else None
        )
        ham_shift_arr = (
            np.asarray(npz_payload["hamiltonian_shifted"], dtype=np.float32)
            if "hamiltonian_shifted" in npz_payload
            else None
        )
        ovl_shift_arr = (
            np.asarray(npz_payload["overlap_shifted"], dtype=np.float32)
            if "overlap_shifted" in npz_payload
            else None
        )
        den_shift_arr = (
            np.asarray(npz_payload["density_shifted"], dtype=np.float32)
            if "density_shifted" in npz_payload
            else None
        )
        positions = _extract_positions(
            metadata, npz_payload, dtype=dtype, device=device
        )
        box = _extract_box(metadata, npz_payload, dtype=dtype, device=device)
        forces = _extract_forces(metadata, npz_payload, dtype=dtype, device=device)
        stress = _extract_stress(metadata, npz_payload, dtype=dtype, device=device)

    ham_t = torch.as_tensor(ham_arr, dtype=dtype, device=device)
    overlap_t = torch.as_tensor(overlap_arr, dtype=dtype, device=device)
    density_t = torch.as_tensor(density_arr, dtype=dtype, device=device)
    shifts_t = (
        torch.as_tensor(shifts_arr, dtype=torch.long, device=device)
        if shifts_arr is not None
        else None
    )
    ham_shift_t = (
        torch.as_tensor(ham_shift_arr, dtype=dtype, device=device)
        if ham_shift_arr is not None
        else None
    )
    ovl_shift_t = (
        torch.as_tensor(ovl_shift_arr, dtype=dtype, device=device)
        if ovl_shift_arr is not None
        else None
    )
    den_shift_t = (
        torch.as_tensor(den_shift_arr, dtype=dtype, device=device)
        if den_shift_arr is not None
        else None
    )

    _validate_dense_square("hamiltonian_ao", ham_t)
    _validate_dense_square("overlap_ao", overlap_t)
    _validate_dense_square("dm_ao", density_t)

    if ham_t.shape != overlap_t.shape or ham_t.shape != density_t.shape:
        raise ValueError(
            "hamiltonian_ao, overlap_ao, and dm_ao shapes differ: "
            f"{tuple(ham_t.shape)}, {tuple(overlap_t.shape)}, {tuple(density_t.shape)}"
        )

    expected_dim = sum(orb_cfg.block_dims(f"{el}-{el}")[0] for el in atoms_tuple)
    actual_dim = int(ham_t.shape[0])
    if expected_dim != actual_dim:
        raise ValueError(
            f"AO dimension mismatch for atoms/orbital_cfg: expected {expected_dim}, got {actual_dim}"
        )

    if shifts_t is None:
        raise ValueError("PBC-only baseline requires `shifts` in the NPZ payload.")
    if shifts_t.ndim != 2 or shifts_t.shape[1] != 3:
        raise ValueError(f"shifts must have shape (N,3), got {tuple(shifts_t.shape)}")
    nshift = int(shifts_t.shape[0])
    if ham_shift_t is None or ovl_shift_t is None or den_shift_t is None:
        raise ValueError(
            "Shift metadata is incomplete: expected hamiltonian_shifted, overlap_shifted, density_shifted."
        )
    _validate_shifted("hamiltonian_shifted", ham_shift_t, nshift, actual_dim)
    _validate_shifted("overlap_shifted", ovl_shift_t, nshift, actual_dim)
    _validate_shifted("density_shifted", den_shift_t, nshift, actual_dim)

    if positions is not None and positions.shape[0] != len(atoms_tuple):
        raise ValueError(
            "positions length does not match atoms: "
            f"{positions.shape[0]} vs {len(atoms_tuple)}"
        )
    if forces is not None and positions is None:
        raise ValueError("forces were present but positions are missing")
    if forces is not None and forces.shape != positions.shape:
        raise ValueError(
            "forces shape does not match positions: "
            f"{tuple(forces.shape)} vs {tuple(positions.shape)}"
        )
    if box is not None and tuple(box.shape) != (3, 3):
        raise ValueError(f"box must have shape (3,3), got {tuple(box.shape)}")
    if stress is not None and tuple(stress.shape) != (3, 3):
        raise ValueError(f"stress must have shape (3,3), got {tuple(stress.shape)}")

    return PySCFBaselineData(
        npz_path=npz_path_obj,
        json_path=json_path_obj,
        metadata=metadata,
        atoms=atoms_tuple,
        orbital_cfg=orb_cfg,
        hamiltonian_ao=ham_t,
        overlap_ao=overlap_t,
        density_ao=density_t,
        shifts=shifts_t,
        hamiltonian_shifted=ham_shift_t,
        overlap_shifted=ovl_shift_t,
        density_shifted=den_shift_t,
        positions=positions,
        box=box,
        forces=forces,
        stress=stress,
    )


def load_pyscf_snapshot(
    npz_path: str | Path,
    *,
    json_path: str | Path | None = None,
    atoms: Iterable[str] | None = None,
    orbital_set: Dict[str, str] | None = None,
    orbital_cfg: OrbitalIrrepConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    basis: str = "pyscf",
) -> Snapshot:
    """Convenience wrapper: parse artifacts and return a ``Snapshot``."""
    payload = load_pyscf_baseline(
        npz_path=npz_path,
        json_path=json_path,
        atoms=atoms,
        orbital_set=orbital_set,
        orbital_cfg=orbital_cfg,
        dtype=dtype,
        device=device,
    )
    return payload.to_snapshot(basis=basis)
