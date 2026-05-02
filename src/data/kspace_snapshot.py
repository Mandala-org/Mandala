from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from ase.cell import Cell
from ase.dft.kpoints import BandPath

from core.basis_converter import (
    _U_FHIAIMS_TO_WIKI,
    _U_OPENMX_TO_WIKI,
    _U_PYSCF_TO_WIKI,
    _orbital_types_from_irreps,
)
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.periodic_fourier import (
    kspace_to_shiftspace_dense,
    shiftspace_to_kspace_dense,
    translation_shifts_for_kmesh,
)
from data.block_matrix import BlockMatrix

__all__ = [
    "BandStructure",
    "KSpaceMatrix",
    "KSpaceSnapshot",
    "build_band_path",
    "block_matrix_to_shiftspace_dense",
    "load_pyscf_kspace_snapshot",
]


_TO_E3NN = {
    "openmx": _U_OPENMX_TO_WIKI,
    "pyscf": _U_PYSCF_TO_WIKI,
    "fhi-aims": _U_FHIAIMS_TO_WIKI,
    "e3nn": None,
}


@dataclass(slots=True)
class BandStructure:
    eigenvalues: torch.Tensor
    kpoints_abs: torch.Tensor
    linear_k: torch.Tensor
    tick_positions: torch.Tensor
    tick_labels: list[str]
    fractional_kpoints: torch.Tensor | None = None
    fermi_level: torch.Tensor | None = None
    overlap_psd_cleanup: bool = False
    overlap_jitter: bool = False


def _fractional_to_cartesian_kpoints(
    fractional_kpoints: torch.Tensor,
    box: torch.Tensor,
) -> torch.Tensor:
    reciprocal = 2 * torch.pi * torch.linalg.inv(box).T
    return fractional_kpoints @ reciprocal


def _linear_k_axis(kpoints_abs: torch.Tensor) -> torch.Tensor:
    if kpoints_abs.ndim != 2 or kpoints_abs.shape[1] != 3:
        raise ValueError("kpoints_abs must have shape (Nk,3)")
    if kpoints_abs.shape[0] == 0:
        return torch.zeros(0, dtype=kpoints_abs.dtype, device=kpoints_abs.device)
    if kpoints_abs.shape[0] == 1:
        return torch.zeros(1, dtype=kpoints_abs.dtype, device=kpoints_abs.device)
    deltas = torch.linalg.norm(kpoints_abs[1:] - kpoints_abs[:-1], dim=-1)
    return torch.cat(
        [
            torch.zeros(1, dtype=kpoints_abs.dtype, device=kpoints_abs.device),
            torch.cumsum(deltas, dim=0),
        ]
    )


def _format_k_label(label: str) -> str:
    return r"$\Gamma$" if label == "G" else label


def build_band_path(
    box: torch.Tensor,
    *,
    path: str | None = None,
    special_points: dict[str, Sequence[float]] | None = None,
    npoints: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    cell = Cell(box.detach().cpu().numpy())
    if special_points is None:
        band_path = cell.bandpath(path=path, npoints=npoints)
    else:
        band_path = BandPath(
            cell, path=path, special_points=special_points
        ).interpolate(npoints=npoints)

    fractional_kpoints = torch.tensor(
        np.asarray(band_path.kpts),
        dtype=box.dtype,
        device=box.device,
    )
    kpoints_abs = torch.tensor(
        np.asarray(band_path.cartesian_kpts()),
        dtype=box.dtype,
        device=box.device,
    )
    linear_k_np, tick_positions_np, tick_labels_raw = band_path.get_linear_kpoint_axis()
    linear_k = torch.tensor(linear_k_np, dtype=box.dtype, device=box.device)
    tick_positions = torch.tensor(
        tick_positions_np,
        dtype=box.dtype,
        device=box.device,
    )
    tick_labels = [_format_k_label(str(label)) for label in tick_labels_raw]
    return fractional_kpoints, kpoints_abs, linear_k, tick_positions, tick_labels


def _generalized_eigenvalues_kspace(
    hamiltonian_k: torch.Tensor,
    overlap_k: torch.Tensor | None,
    *,
    psd_cleanup: bool = False,
    allow_jitter: bool = False,
) -> torch.Tensor:
    H = 0.5 * (hamiltonian_k + hamiltonian_k.transpose(-1, -2).conj())
    if overlap_k is None:
        return torch.linalg.eigvalsh(H)

    S = 0.5 * (overlap_k + overlap_k.transpose(-1, -2).conj())
    if psd_cleanup:
        evals_S, evecs_S = torch.linalg.eigh(S)
        evals_real = evals_S.real
        min_eval = float(torch.min(evals_real).item())
        max_eval = float(torch.max(evals_real).item())
        if min_eval < 1.0e-6 or not bool(torch.isfinite(evals_real).all()):
            print(
                f"[OVERLAP] k-space PSD projection needed: shape={tuple(S.shape)} "
                f"eig_min={min_eval:.6e} eig_max={max_eval:.6e} floor=1.0e-6"
            )
        finite = torch.isfinite(evals_real)
        if not bool(finite.all()):
            n_bad = int((~finite).sum().item())
            print(f"[OVERLAP] k-space replacing {n_bad} non-finite overlap eigenvalues")
            evals_real = torch.where(finite, evals_real, torch.zeros_like(evals_real))
        positive = evals_real[evals_real > 1.0e-6]
        if positive.numel() > 0:
            upper = float(torch.quantile(positive, 0.995).item()) * 10.0
            upper = max(upper, 1.0e-6)
        else:
            upper = 1.0e-5
        evals_real = torch.clamp(evals_real, min=1.0e-6, max=upper)
        S = (
            evecs_S
            @ torch.diag_embed(evals_real.to(dtype=evecs_S.dtype))
            @ evecs_S.transpose(-1, -2).conj()
        )
    n = S.shape[-1]
    eye = torch.eye(n, dtype=S.dtype, device=S.device)
    jitters = (0.0, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4) if allow_jitter else (0.0,)
    for jitter in jitters:
        try:
            S_reg = S if jitter == 0.0 else S + jitter * eye
            if jitter > 0.0:
                print(f"[OVERLAP] k-space retrying Cholesky with jitter={jitter:.1e}")
            L = torch.linalg.cholesky(S_reg)
            tmp = torch.linalg.solve(L, H)
            A = (
                torch.linalg.solve(L, tmp.transpose(-1, -2).conj())
                .transpose(-1, -2)
                .conj()
            )
            A = 0.5 * (A + A.transpose(-1, -2).conj())
            return torch.linalg.eigvalsh(A)
        except torch.linalg.LinAlgError:
            print(f"[OVERLAP] k-space Cholesky failed at jitter={jitter:.1e}")
            continue

    raise torch.linalg.LinAlgError(
        "k-space generalized eigensolve failed: overlap Cholesky did not succeed"
        + (" even after jitter retries." if allow_jitter else ".")
    )


def block_matrix_to_shiftspace_dense(
    mat: BlockMatrix,
    *,
    shifts: torch.Tensor,
) -> torch.Tensor:
    total_dim = sum(mat.orbital_cfg.block_dims(f"{el}-{el}")[0] for el in mat.atoms)
    device = next(iter(mat.pair_blocks.values())).device
    out = torch.zeros(
        shifts.shape[0],
        total_dim,
        total_dim,
        dtype=next(iter(mat.pair_blocks.values())).dtype,
        device=device,
    )
    offsets = _global_offsets(mat.atoms, mat.orbital_cfg, device=device)
    shift_to_idx = {tuple(map(int, s.tolist())): idx for idx, s in enumerate(shifts)}
    for key, edges in mat.pair_edges.items():
        blocks = mat.pair_blocks[key]
        for idx, edge in enumerate(edges.t().tolist()):
            sx, sy, sz, i, j = edge
            s_idx = shift_to_idx.get((int(sx), int(sy), int(sz)))
            if s_idx is None:
                continue
            di = mat.orbital_cfg.block_dims(f"{mat.atoms[i]}-{mat.atoms[i]}")[0]
            dj = mat.orbital_cfg.block_dims(f"{mat.atoms[j]}-{mat.atoms[j]}")[0]
            r0 = int(offsets[i])
            c0 = int(offsets[j])
            out[s_idx, r0 : r0 + di, c0 : c0 + dj] = blocks[idx]
    return out


def _global_offsets(
    atoms: tuple[str, ...], orbital_cfg: OrbitalIrrepConfig, *, device: torch.device
) -> torch.Tensor:
    dims = [orbital_cfg.block_dims(f"{el}-{el}")[0] for el in atoms]
    offsets = [0]
    for dim in dims[:-1]:
        offsets.append(offsets[-1] + int(dim))
    return torch.tensor(offsets, dtype=torch.long, device=device)


def _build_global_basis_transform(
    atoms: tuple[str, ...],
    orbital_cfg: OrbitalIrrepConfig,
    *,
    source: str,
    target: str,
    device: torch.device,
) -> torch.Tensor:
    if source == target:
        total_dim = sum(orbital_cfg.block_dims(f"{el}-{el}")[0] for el in atoms)
        return torch.eye(total_dim, dtype=torch.float32, device=device)
    if source not in _TO_E3NN or target not in _TO_E3NN:
        raise RuntimeError(f"Unsupported basis conversion: {source} -> {target}")

    blocks = []
    for el in atoms:
        irreps = orbital_cfg.element_to_irreps[el]
        orbital_types = _orbital_types_from_irreps(irreps)

        if source == "e3nn":
            mats = [_TO_E3NN[target][l].T for l in orbital_types]
        elif target == "e3nn":
            mats = [_TO_E3NN[source][l] for l in orbital_types]
        else:
            mats_src = [_TO_E3NN[source][l] for l in orbital_types]
            mats_tgt = [_TO_E3NN[target][l].T for l in orbital_types]
            mats = [v @ u for v, u in zip(mats_tgt, mats_src)]

        blocks.append(torch.block_diag(*mats).to(device))

    return torch.block_diag(*blocks)


def _axis_change_of_basis(
    source: str,
    target: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if source == "openmx" and target == "e3nn":
        return torch.eye(3, dtype=dtype, device=device)[[2, 0, 1]]
    if source == "e3nn" and target == "openmx":
        return torch.eye(3, dtype=dtype, device=device)[[1, 2, 0]]
    return None


def _dense_shift_to_block_matrix(
    shifted: torch.Tensor,
    shifts: torch.Tensor,
    atoms: tuple[str, ...],
    orbital_cfg: OrbitalIrrepConfig,
    *,
    basis: str,
) -> BlockMatrix:
    device = shifted.device
    offsets = _global_offsets(atoms, orbital_cfg, device=device)
    edge_blocks: dict[tuple[int, int, int, int, int], torch.Tensor] = {}

    def _register(edge: tuple[int, int, int, int, int], block: torch.Tensor) -> None:
        existing = edge_blocks.get(edge)
        if existing is None:
            edge_blocks[edge] = block
            return
        if not torch.allclose(existing, block, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Inconsistent duplicate block for edge {edge}")

    for s_idx in range(int(shifts.shape[0])):
        sx, sy, sz = [int(v) for v in shifts[s_idx].tolist()]
        mat = shifted[s_idx]
        for i, el_i in enumerate(atoms):
            di = orbital_cfg.block_dims(f"{el_i}-{el_i}")[0]
            r0 = int(offsets[i])
            for j, el_j in enumerate(atoms):
                dj = orbital_cfg.block_dims(f"{el_j}-{el_j}")[0]
                c0 = int(offsets[j])
                block = mat[r0 : r0 + di, c0 : c0 + dj].clone()
                edge = (sx, sy, sz, i, j)
                _register(edge, block)
                _register((-sx, -sy, -sz, j, i), block.transpose(-1, -2).clone())

    pair_blocks: dict[str, list[torch.Tensor]] = {}
    pair_edges: dict[str, list[list[int]]] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}
    for edge, block in sorted(edge_blocks.items()):
        sx, sy, sz, i, j = edge
        key = f"{atoms[i]}-{atoms[j]}"
        pair_blocks.setdefault(key, [])
        pair_edges.setdefault(key, [])
        local_idx = len(pair_blocks[key])
        pair_blocks[key].append(block)
        pair_edges[key].append([sx, sy, sz, i, j])
        lookup[edge] = (key, local_idx)

    return BlockMatrix(
        atoms=atoms,
        atom_counts=Counter(atoms),
        pair_blocks={k: torch.stack(v, dim=0) for k, v in pair_blocks.items()},
        pair_edges={
            k: torch.tensor(v, dtype=torch.long, device=device).t()
            for k, v in pair_edges.items()
        },
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis=basis,
    )


@dataclass(slots=True)
class KSpaceMatrix:
    atoms: tuple[str, ...]
    atom_counts: Dict[str, int]
    matrices_k: torch.Tensor
    kpoints_abs: torch.Tensor
    orbital_cfg: OrbitalIrrepConfig
    basis: str
    box: torch.Tensor
    kmesh: tuple[int, int, int] | None = None
    shifts: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.matrices_k.ndim != 3:
            raise ValueError(
                "matrices_k must have shape (Nk,nao,nao), "
                f"got {tuple(self.matrices_k.shape)}"
            )
        if self.matrices_k.shape[1] != self.matrices_k.shape[2]:
            raise ValueError("k-space matrices must be square in AO space")
        if self.kpoints_abs.ndim != 2 or self.kpoints_abs.shape[1] != 3:
            raise ValueError("kpoints_abs must have shape (Nk,3)")
        if self.kpoints_abs.shape[0] != self.matrices_k.shape[0]:
            raise ValueError("kpoint count must match leading matrix dimension")
        if self.box.shape != (3, 3):
            raise ValueError("box must have shape (3,3)")
        expected_dim = sum(
            self.orbital_cfg.block_dims(f"{el}-{el}")[0] for el in self.atoms
        )
        if int(self.matrices_k.shape[1]) != expected_dim:
            raise ValueError(
                "AO dimension mismatch for atoms/orbital_cfg: "
                f"expected {expected_dim}, got {self.matrices_k.shape[1]}"
            )
        if self.shifts is not None and (
            self.shifts.ndim != 2 or self.shifts.shape[1] != 3
        ):
            raise ValueError("shifts must have shape (Nshift,3)")

    def to(self, device: str | torch.device) -> "KSpaceMatrix":
        return KSpaceMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            matrices_k=self.matrices_k.to(device),
            kpoints_abs=self.kpoints_abs.to(device),
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
            box=self.box.to(device),
            kmesh=self.kmesh,
            shifts=self.shifts.to(device) if self.shifts is not None else None,
        )

    def _with_matrices(
        self, matrices_k: torch.Tensor, *, basis: str | None = None
    ) -> "KSpaceMatrix":
        return KSpaceMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            matrices_k=matrices_k,
            kpoints_abs=self.kpoints_abs,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis if basis is None else basis,
            box=self.box,
            kmesh=self.kmesh,
            shifts=self.shifts,
        )

    def _replace(
        self,
        *,
        matrices_k: torch.Tensor | None = None,
        kpoints_abs: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        basis: str | None = None,
        shifts: torch.Tensor | None = None,
    ) -> "KSpaceMatrix":
        return KSpaceMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            matrices_k=self.matrices_k if matrices_k is None else matrices_k,
            kpoints_abs=self.kpoints_abs if kpoints_abs is None else kpoints_abs,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis if basis is None else basis,
            box=self.box if box is None else box,
            kmesh=self.kmesh,
            shifts=self.shifts if shifts is None else shifts,
        )

    def change_basis(self, target: str) -> "KSpaceMatrix":
        if target == self.basis:
            return self
        U = _build_global_basis_transform(
            self.atoms,
            self.orbital_cfg,
            source=self.basis,
            target=target,
            device=self.matrices_k.device,
        ).to(self.matrices_k.dtype)
        mats = U @ self.matrices_k @ U.T

        cob = _axis_change_of_basis(
            self.basis,
            target,
            device=self.box.device,
            dtype=self.box.dtype,
        )
        box = self.box if cob is None else self.box @ cob
        kpoints_abs = self.kpoints_abs if cob is None else self.kpoints_abs @ cob
        return KSpaceMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            matrices_k=mats,
            kpoints_abs=kpoints_abs,
            orbital_cfg=self.orbital_cfg,
            basis=target,
            box=box,
            kmesh=self.kmesh,
            shifts=self.shifts,
        )

    def rotate(self, R: torch.Tensor) -> "KSpaceMatrix":
        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 rotation matrix")
        blocks = []
        for el in self.atoms:
            irr = self.orbital_cfg.element_to_irreps[el]
            blocks.append(
                irr.D_from_matrix(
                    R.to(device=self.matrices_k.device, dtype=torch.float32)
                )
            )
        U = torch.block_diag(*blocks).to(
            self.matrices_k.device, dtype=self.matrices_k.dtype
        )
        return self._replace(
            matrices_k=U @ self.matrices_k @ U.T,
            kpoints_abs=self.kpoints_abs
            @ R.T.to(self.kpoints_abs.device, self.kpoints_abs.dtype),
            box=self.box @ R.T.to(self.box.device, self.box.dtype),
        )

    def to_shiftspace_dense(
        self, *, shifts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifts_t = shifts
        if shifts_t is None:
            if self.shifts is not None:
                shifts_t = self.shifts
            elif self.kmesh is not None:
                shifts_t = translation_shifts_for_kmesh(
                    self.kmesh, device=self.matrices_k.device
                )
            else:
                raise ValueError(
                    "Need either shifts or kmesh to convert to shift space"
                )
        mats = kspace_to_shiftspace_dense(
            self.matrices_k,
            kpoints_abs=self.kpoints_abs,
            shifts=shifts_t,
            box=self.box,
        )
        return shifts_t, mats

    def to_block_matrix(self, *, shifts: torch.Tensor | None = None) -> BlockMatrix:
        shifts_t, mats = self.to_shiftspace_dense(shifts=shifts)
        return _dense_shift_to_block_matrix(
            mats,
            shifts_t,
            self.atoms,
            self.orbital_cfg,
            basis=self.basis,
        )

    def get_band_structure(
        self,
        *,
        overlap: "KSpaceMatrix | None" = None,
        fractional_kpoints: torch.Tensor | None = None,
        linear_k: torch.Tensor | None = None,
        tick_positions: torch.Tensor | None = None,
        tick_labels: Sequence[str] | None = None,
        fermi_level: torch.Tensor | None = None,
        chunk_size: int | None = None,
        psd_cleanup: bool = False,
        allow_jitter: bool = False,
    ) -> BandStructure:
        if overlap is not None:
            if overlap.matrices_k.shape != self.matrices_k.shape:
                raise ValueError(
                    "Hamiltonian and overlap must have matching k-space shapes."
                )
            if not torch.allclose(overlap.kpoints_abs, self.kpoints_abs):
                raise ValueError("Hamiltonian and overlap must share kpoints.")

        if (
            chunk_size is None
            or chunk_size <= 0
            or chunk_size >= self.matrices_k.shape[0]
        ):
            eigenvalues = _generalized_eigenvalues_kspace(
                self.matrices_k,
                None if overlap is None else overlap.matrices_k,
                psd_cleanup=psd_cleanup,
                allow_jitter=allow_jitter,
            )
        else:
            chunks = []
            for start in range(0, self.matrices_k.shape[0], int(chunk_size)):
                stop = min(start + int(chunk_size), self.matrices_k.shape[0])
                chunks.append(
                    _generalized_eigenvalues_kspace(
                        self.matrices_k[start:stop],
                        None if overlap is None else overlap.matrices_k[start:stop],
                        psd_cleanup=psd_cleanup,
                        allow_jitter=allow_jitter,
                    )
                )
            eigenvalues = torch.cat(chunks, dim=0)
        linear_axis = (
            _linear_k_axis(self.kpoints_abs)
            if linear_k is None
            else linear_k.to(
                device=self.kpoints_abs.device, dtype=self.kpoints_abs.dtype
            )
        )
        if tick_positions is None:
            tick_positions_t = torch.tensor(
                [float(linear_axis[0].item()), float(linear_axis[-1].item())],
                dtype=linear_axis.dtype,
                device=linear_axis.device,
            )
        else:
            tick_positions_t = tick_positions.to(
                device=linear_axis.device,
                dtype=linear_axis.dtype,
            )
        tick_labels_list = (
            [_format_k_label("G"), _format_k_label("X")]
            if tick_labels is None
            else [str(label) for label in tick_labels]
        )
        fractional = None
        if fractional_kpoints is not None:
            fractional = fractional_kpoints.to(
                device=self.kpoints_abs.device,
                dtype=self.kpoints_abs.dtype,
            )
        fermi = None
        if fermi_level is not None:
            fermi = fermi_level.to(
                device=eigenvalues.device,
                dtype=eigenvalues.real.dtype,
            )
        return BandStructure(
            eigenvalues=eigenvalues,
            kpoints_abs=self.kpoints_abs,
            linear_k=linear_axis,
            tick_positions=tick_positions_t,
            tick_labels=tick_labels_list,
            fractional_kpoints=fractional,
            fermi_level=fermi,
            overlap_psd_cleanup=bool(psd_cleanup),
            overlap_jitter=bool(allow_jitter),
        )

    @classmethod
    def from_shiftspace_dense(
        cls,
        matrices_shift: torch.Tensor,
        *,
        atoms: Sequence[str],
        orbital_cfg: OrbitalIrrepConfig,
        basis: str,
        kpoints_abs: torch.Tensor,
        box: torch.Tensor,
        shifts: torch.Tensor,
        kmesh: Sequence[int] | None = None,
    ) -> "KSpaceMatrix":
        mats_k = shiftspace_to_kspace_dense(
            matrices_shift,
            kpoints_abs=kpoints_abs,
            shifts=shifts,
            box=box,
        )
        atoms_t = tuple(str(a) for a in atoms)
        return cls(
            atoms=atoms_t,
            atom_counts=Counter(atoms_t),
            matrices_k=mats_k,
            kpoints_abs=kpoints_abs,
            orbital_cfg=orbital_cfg,
            basis=basis,
            box=box,
            kmesh=None if kmesh is None else tuple(int(v) for v in kmesh),
            shifts=shifts,
        )


@dataclass(slots=True)
class KSpaceSnapshot:
    hamiltonian: KSpaceMatrix
    overlap: KSpaceMatrix
    density: KSpaceMatrix
    positions: torch.Tensor | None = None
    forces: torch.Tensor | None = None
    box: torch.Tensor | None = None
    stress: torch.Tensor | None = None
    matrix_path: str | Path | None = None
    info_path: str | Path | None = None
    cutoff_radius: float | None = None
    cfg: Any = None
    info: Any = None

    def __post_init__(self) -> None:
        mats = (self.hamiltonian, self.overlap, self.density)
        first = mats[0]
        for other in mats[1:]:
            if other.atoms != first.atoms:
                raise ValueError("k-space matrices must share atoms")
            if other.orbital_cfg.to_dict() != first.orbital_cfg.to_dict():
                raise ValueError("k-space matrices must share orbital config")
            if other.basis != first.basis:
                raise ValueError("k-space matrices must share basis")
            if not torch.allclose(other.kpoints_abs, first.kpoints_abs):
                raise ValueError("k-space matrices must share kpoints")
        if self.box is not None and not torch.allclose(self.box, first.box):
            raise ValueError("Snapshot box and k-space matrix box must match")

    @property
    def basis(self) -> str:
        return self.hamiltonian.basis

    @property
    def atoms(self) -> tuple[str, ...]:
        return self.hamiltonian.atoms

    @property
    def orbital_cfg(self) -> OrbitalIrrepConfig:
        return self.hamiltonian.orbital_cfg

    @property
    def kpoints_abs(self) -> torch.Tensor:
        return self.hamiltonian.kpoints_abs

    @property
    def kmesh(self) -> tuple[int, int, int] | None:
        return self.hamiltonian.kmesh

    @property
    def shifts(self) -> torch.Tensor | None:
        return self.hamiltonian.shifts

    def change_basis(self, target: str) -> "KSpaceSnapshot":
        ham = self.hamiltonian.change_basis(target)
        ovl = self.overlap.change_basis(target)
        den = self.density.change_basis(target)

        box = self.box
        positions = self.positions
        forces = self.forces
        stress = self.stress
        if box is not None:
            cob = _axis_change_of_basis(
                self.basis, target, device=box.device, dtype=box.dtype
            )
            if cob is not None:
                box = box @ cob
                positions = positions @ cob if positions is not None else None
                forces = forces @ cob if forces is not None else None
                stress = (
                    cob.T @ stress @ cob
                    if stress is not None and tuple(stress.shape) == (3, 3)
                    else stress
                )

        return KSpaceSnapshot(
            ham,
            ovl,
            den,
            positions=positions,
            forces=forces,
            box=box,
            stress=stress,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
            cfg=self.cfg,
            info=self.info,
        )

    def to_e3nn(self) -> "KSpaceSnapshot":
        return self.change_basis("e3nn")

    def to_openmx(self) -> "KSpaceSnapshot":
        return self.change_basis("openmx")

    def to_pyscf(self) -> "KSpaceSnapshot":
        return self.change_basis("pyscf")

    def rotate(self, R: torch.Tensor) -> "KSpaceSnapshot":
        return KSpaceSnapshot(
            self.hamiltonian.rotate(R),
            self.overlap.rotate(R),
            self.density.rotate(R),
            positions=self.positions @ R.T if self.positions is not None else None,
            forces=self.forces @ R.T if self.forces is not None else None,
            box=self.box @ R.T if self.box is not None else None,
            stress=(
                R @ self.stress @ R.T
                if self.stress is not None and tuple(self.stress.shape) == (3, 3)
                else None
            ),
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
            cfg=self.cfg,
            info=self.info,
        )

    def to_shift_space(self):
        from data.snapshot import Snapshot

        ham = self.hamiltonian.to_block_matrix()
        ovl = self.overlap.to_block_matrix()
        den = self.density.to_block_matrix()
        snap = Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions,
            forces=self.forces,
            box=self.box,
            stress=self.stress,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
            cfg=self.cfg,
            info=self.info,
        )
        return snap

    def get_band_structure(
        self,
        *,
        fractional_kpoints: torch.Tensor | None = None,
        linear_k: torch.Tensor | None = None,
        tick_positions: torch.Tensor | None = None,
        tick_labels: Sequence[str] | None = None,
        chunk_size: int | None = None,
        psd_cleanup: bool = False,
        allow_jitter: bool = False,
    ) -> BandStructure:
        fermi_level = None
        if getattr(self.info, "fermi_level", None) is not None:
            fermi_level = self.info.fermi_level
        return self.hamiltonian.get_band_structure(
            overlap=self.overlap,
            fractional_kpoints=fractional_kpoints,
            linear_k=linear_k,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            fermi_level=fermi_level,
            chunk_size=chunk_size,
            psd_cleanup=psd_cleanup,
            allow_jitter=allow_jitter,
        )

    @classmethod
    def from_shift_space(
        cls,
        snapshot,
        *,
        kpoints_abs: torch.Tensor,
        kmesh: Sequence[int] | None = None,
        shifts: torch.Tensor | None = None,
    ) -> "KSpaceSnapshot":
        if snapshot.box is None:
            raise ValueError("Shift-space snapshot requires box to convert to k-space")
        shift_t = shifts
        if shift_t is None:
            if kmesh is None:
                raise ValueError("Need shifts or kmesh for shift->k conversion")
            shift_t = translation_shifts_for_kmesh(kmesh, device=snapshot.box.device)

        ham_shift = block_matrix_to_shiftspace_dense(
            snapshot.hamiltonian, shifts=shift_t
        )
        ovl_shift = block_matrix_to_shiftspace_dense(snapshot.overlap, shifts=shift_t)
        den_shift = block_matrix_to_shiftspace_dense(snapshot.density, shifts=shift_t)

        ham = KSpaceMatrix.from_shiftspace_dense(
            ham_shift,
            atoms=snapshot.density.atoms,
            orbital_cfg=snapshot.density.orbital_cfg,
            basis=snapshot.density.basis,
            kpoints_abs=kpoints_abs,
            box=snapshot.box,
            shifts=shift_t,
            kmesh=kmesh,
        )
        ovl = KSpaceMatrix.from_shiftspace_dense(
            ovl_shift,
            atoms=snapshot.density.atoms,
            orbital_cfg=snapshot.density.orbital_cfg,
            basis=snapshot.density.basis,
            kpoints_abs=kpoints_abs,
            box=snapshot.box,
            shifts=shift_t,
            kmesh=kmesh,
        )
        den = KSpaceMatrix.from_shiftspace_dense(
            den_shift,
            atoms=snapshot.density.atoms,
            orbital_cfg=snapshot.density.orbital_cfg,
            basis=snapshot.density.basis,
            kpoints_abs=kpoints_abs,
            box=snapshot.box,
            shifts=shift_t,
            kmesh=kmesh,
        )
        return cls(
            ham,
            ovl,
            den,
            positions=snapshot.positions,
            forces=snapshot.forces,
            box=snapshot.box,
            stress=snapshot.stress,
            matrix_path=snapshot.matrix_path,
            info_path=snapshot.info_path,
            cutoff_radius=snapshot.cutoff_radius,
            cfg=snapshot.cfg,
            info=snapshot.info,
        )

    @classmethod
    def from_pyscf(
        cls,
        npz_path: str | Path,
        *,
        json_path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        basis: str = "pyscf",
    ) -> "KSpaceSnapshot":
        from data.pyscf_baseline_parser import load_pyscf_baseline

        payload = load_pyscf_baseline(
            npz_path=npz_path,
            json_path=json_path,
            dtype=dtype,
            device=device,
        )
        if payload.box is None:
            raise ValueError("PySCF k-space loading requires box information")

        metadata = payload.metadata
        with np.load(npz_path, allow_pickle=False) as npz_payload:
            kpts_arr = (
                np.asarray(npz_payload["kpts_abs"], dtype=np.float64)
                if "kpts_abs" in npz_payload
                else None
            )
            kmesh_arr = (
                np.asarray(npz_payload["kmesh"], dtype=np.int64)
                if "kmesh" in npz_payload
                else None
            )
            ham_k_arr = (
                np.asarray(npz_payload["hamiltonian_k"])
                if "hamiltonian_k" in npz_payload
                else None
            )
            ovl_k_arr = (
                np.asarray(npz_payload["overlap_k"])
                if "overlap_k" in npz_payload
                else None
            )
            den_k_arr = (
                np.asarray(npz_payload["density_k"])
                if "density_k" in npz_payload
                else None
            )

        if kpts_arr is None:
            kpts = metadata.get("pyscf", {}).get("kpts_abs")
            if kpts is None:
                raise ValueError(
                    "PySCF artifacts are missing kpts_abs for k-space loading"
                )
            kpts_arr = np.asarray(kpts, dtype=np.float64)
        kpoints_abs = torch.tensor(kpts_arr, dtype=dtype, device=device)

        kmesh_raw = (
            kmesh_arr.tolist()
            if kmesh_arr is not None
            else metadata.get("settings", {}).get("kmesh")
        )
        kmesh = None if kmesh_raw is None else tuple(int(v) for v in kmesh_raw)

        if payload.shifts is None:
            if kmesh is None:
                raise ValueError(
                    "Need either shifts or kmesh for k-space reconstruction"
                )
            shifts = translation_shifts_for_kmesh(kmesh, device=device)
        else:
            shifts = payload.shifts

        if ham_k_arr is not None and ovl_k_arr is not None and den_k_arr is not None:
            ham = KSpaceMatrix(
                atoms=payload.atoms,
                atom_counts=Counter(payload.atoms),
                matrices_k=torch.as_tensor(ham_k_arr, device=device),
                kpoints_abs=kpoints_abs,
                orbital_cfg=payload.orbital_cfg,
                basis=basis,
                box=payload.box,
                kmesh=kmesh,
                shifts=shifts,
            )
            ovl = KSpaceMatrix(
                atoms=payload.atoms,
                atom_counts=Counter(payload.atoms),
                matrices_k=torch.as_tensor(ovl_k_arr, device=device),
                kpoints_abs=kpoints_abs,
                orbital_cfg=payload.orbital_cfg,
                basis=basis,
                box=payload.box,
                kmesh=kmesh,
                shifts=shifts,
            )
            den = KSpaceMatrix(
                atoms=payload.atoms,
                atom_counts=Counter(payload.atoms),
                matrices_k=torch.as_tensor(den_k_arr, device=device),
                kpoints_abs=kpoints_abs,
                orbital_cfg=payload.orbital_cfg,
                basis=basis,
                box=payload.box,
                kmesh=kmesh,
                shifts=shifts,
            )
        else:
            ham = KSpaceMatrix.from_shiftspace_dense(
                payload.hamiltonian_shifted,
                atoms=payload.atoms,
                orbital_cfg=payload.orbital_cfg,
                basis=basis,
                kpoints_abs=kpoints_abs,
                box=payload.box,
                shifts=shifts,
                kmesh=kmesh,
            )
            ovl = KSpaceMatrix.from_shiftspace_dense(
                payload.overlap_shifted,
                atoms=payload.atoms,
                orbital_cfg=payload.orbital_cfg,
                basis=basis,
                kpoints_abs=kpoints_abs,
                box=payload.box,
                shifts=shifts,
                kmesh=kmesh,
            )
            den = KSpaceMatrix.from_shiftspace_dense(
                payload.density_shifted,
                atoms=payload.atoms,
                orbital_cfg=payload.orbital_cfg,
                basis=basis,
                kpoints_abs=kpoints_abs,
                box=payload.box,
                shifts=shifts,
                kmesh=kmesh,
            )
        return cls(
            ham,
            ovl,
            den,
            positions=payload.positions,
            forces=payload.forces,
            box=payload.box,
            stress=payload.stress,
            matrix_path=payload.npz_path,
            info_path=payload.json_path,
            info=payload.metadata,
        )


def load_pyscf_kspace_snapshot(
    npz_path: str | Path,
    *,
    json_path: str | Path | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    basis: str = "pyscf",
) -> KSpaceSnapshot:
    return KSpaceSnapshot.from_pyscf(
        npz_path=npz_path,
        json_path=json_path,
        dtype=dtype,
        device=device,
        basis=basis,
    )
