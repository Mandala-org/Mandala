"""
snapshot.py
============

High-level container that bundles *all* block-sparse matrices (Hamiltonian,
Overlap, Density …) for **one structure** and exposes convenience physics
utilities.

Key features
------------
* Keeps three :class:`BlockMatrix` objects (`hamiltonian`, `overlap`,
  `density`) under a unified interface.  Access via ``snap[\"density\"]`` **or**
  attribute ``snap.density``.
* Upon construction **re-orders edges** for every element-pair by *ascending*
  L2-norm of the density blocks - this yields deterministic ordering and is
  helpful for compression / batching.
* Implements

    * ``get_number_of_electrons()``  ->  Tr(D·S)
    * ``get_energy()``               ->  Tr(D·H)

  using the highly-optimised
  :func:`core.sparse_math.trace_matmul_sparse_snap_vectorized`.
* Simple ``save(path)`` / ``load(path)`` round-trip based on the underlying
  Block snapshots' serialisation.
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Mapping
import torch
import h5py
from ase.io import read as ase_read
from ase import Atoms
from e3nn.o3 import Irreps

from core.sparse_math import (
    trace_matmul_sparse_block_matrix,
)
from data.block_matrix import BlockMatrix
from core.basis_converter import (
    OpenMXE3NNConverter,
    FHIaimsE3NNConverter,
    PySCFE3NNConverter,
)
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from core.periodic_fourier import shiftspace_to_kspace_dense

__all__ = ["Snapshot"]


def _compute_atom_image_shifts(
    reference_positions: torch.Tensor,
    target_positions: torch.Tensor,
    box: torch.Tensor,
    *,
    atol: float = 1.0e-4,
) -> torch.Tensor:
    if reference_positions.shape != target_positions.shape:
        raise ValueError(
            "reference_positions and target_positions must have the same shape"
        )
    if box.shape != torch.Size([3, 3]):
        raise ValueError(f"box must have shape (3, 3), got {tuple(box.shape)}")
    inv_box = torch.linalg.inv(box)
    delta_frac = (target_positions - reference_positions) @ inv_box
    atom_shifts = torch.round(delta_frac)
    residual = delta_frac - atom_shifts
    if float(torch.max(torch.abs(residual)).item()) > atol:
        raise RuntimeError(
            "Atom positions are not related by integer lattice shifts within tolerance "
            f"{atol}. max_residual={float(torch.max(torch.abs(residual)).item()):.6e}"
        )
    return atom_shifts.to(dtype=torch.long)


class Snapshot:
    """Bundle H, S, D for one configuration and provide physics helpers."""

    _ORBITAL_TO_L = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
        "h": 5,
        "i": 6,
        "k": 7,
        "l": 8,
        "m": 9,
    }

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        hamiltonian: BlockMatrix,
        overlap: BlockMatrix,
        density: BlockMatrix,
        *,
        positions: torch.Tensor | None = None,  # (N,3)
        forces: torch.Tensor | None = None,  # (N,3)
        box: torch.Tensor | None = None,  # (3,3)
        stress: torch.Tensor | None = None,  # (3,3) stress tensor (multiplicative)
        matrix_path=None,  # optional path to the source file
        info_path=None,  # optional path to the source info file
        cutoff_radius: float | None = None,  # optional cutoff radius for filtering
        cfg: Config | None = None,
        info: Any = None,
    ) -> None:
        # quick consistency sanity checks
        self._check_compatibility(hamiltonian, overlap, density)

        self._mats: Dict[str, BlockMatrix] = {
            "hamiltonian": hamiltonian,
            "overlap": overlap,
            "density": density,
        }

        self.positions = positions
        self.forces = forces
        self.box = box  # may be None for non-periodic test cases
        self.stress = stress

        self.matrix_path = matrix_path  # optional path to the source file
        self.info_path = info_path
        self.cutoff_radius = cutoff_radius

        self.hamiltonian = self._mats["hamiltonian"]
        self.overlap = self._mats["overlap"]
        self.density = self._mats["density"]

        self.cfg = cfg
        self.info = info

    # ---------------------------------------------------------------- compatibility
    @staticmethod
    def _check_compatibility(*mats: BlockMatrix) -> None:
        if not mats:
            return

        first = mats[0]
        for i, m in enumerate(mats[1:]):
            if m.atoms != first.atoms:
                raise ValueError(f"Matrices 0 and {i+1} must share the same atom list")

            if m.atom_counts.keys() != first.atom_counts.keys():
                raise ValueError(
                    f"Matrices 0 and {i+1} have different element sets in atom_counts"
                )

            for key in first.atom_counts:
                if m.atom_counts[key] != first.atom_counts[key]:
                    raise ValueError(
                        f"Matrices 0 and {i+1} have different counts for element {key}"
                    )

            if m.orbital_cfg.to_dict() != first.orbital_cfg.to_dict():
                raise ValueError(
                    f"Matrices 0 and {i+1} must share the same orbital config"
                )

            if m.basis != first.basis:
                raise ValueError(
                    f"Matrices 0 and {i+1} must use the same basis (openmx/e3nn/fhi-aims)"
                )

    # ---------------------------------------------------------------- edge ordering
    def canonicalize_edges(self) -> "Snapshot":
        """
        Return a new Snapshot with a canonical edge ordering for each key.
        The canonical order for each key is:
        1. Diagonal edges, sorted by their node index.
        2. Off-diagonal edges, sorted by the distance (ascending).
        """
        dists = self._edge_distances(self.density)
        order_dict = {}

        for key in self.density.pair_edges.keys():
            edges = self.density.pair_edges[key]  # (5, E)
            # edges rows: 0:sx, 1:sy, 2:sz, 3:src, 4:dst

            # Get distances for this key
            D = dists[key]  # (E,)

            num_edges = edges.shape[1]
            indices = torch.arange(num_edges, device=edges.device)

            # Identify diagonal edges: src == dst AND sx==0 AND sy==0 AND sz==0
            sx = edges[0]
            sy = edges[1]
            sz = edges[2]
            src = edges[3]
            dst = edges[4]

            is_diag = (src == dst) & (sx == 0) & (sy == 0) & (sz == 0)

            diag_indices = indices[is_diag]
            off_diag_indices = indices[~is_diag]

            # Sort diagonal indices by src
            diag_src = src[diag_indices]
            perm_diag = torch.argsort(diag_src)
            sorted_diag_indices = diag_indices[perm_diag]

            # Sort off-diagonal indices
            # Primary key: distance
            # Tie-breaker: sx, sy, sz, src, dst
            od_idx = off_diag_indices
            od_d = D[od_idx]
            od_edges = edges[:, od_idx]  # (5, E_od)

            # Move to CPU for sorting
            od_d_cpu = od_d.cpu().tolist()
            od_edges_cpu = od_edges.t().cpu().tolist()  # List of [sx, sy, sz, src, dst]
            od_idx_cpu = od_idx.cpu().tolist()

            # Combine into a list of tuples
            # (dist, sx, sy, sz, src, dst, original_idx)
            to_sort = []
            for i in range(len(od_idx_cpu)):
                row = od_edges_cpu[i]  # [sx, sy, sz, src, dst]
                to_sort.append(
                    (
                        od_d_cpu[i],
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        od_idx_cpu[i],
                    )
                )

            to_sort.sort()

            sorted_off_diag_indices = torch.tensor(
                [x[-1] for x in to_sort], device=edges.device, dtype=torch.long
            )

            # Concatenate
            final_indices = torch.cat([sorted_diag_indices, sorted_off_diag_indices])
            order_dict[key] = final_indices

        # Apply reordering
        new_ham = self.hamiltonian.reorder_edges(order_dict)
        new_ovl = self.overlap.reorder_edges(order_dict)
        new_den = self.density.reorder_edges(order_dict)

        return Snapshot(
            new_ham,
            new_ovl,
            new_den,
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

    def relabel_atom_images(self, atom_image_shifts: torch.Tensor) -> "Snapshot":
        """
        Relabel all matrix edge shifts to match a different per-atom periodic-image
        convention.
        """
        return Snapshot(
            self.hamiltonian.relabel_atom_images(atom_image_shifts),
            self.overlap.relabel_atom_images(atom_image_shifts),
            self.density.relabel_atom_images(atom_image_shifts),
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

    # ---------------------------------------------------------------- physics helpers
    def get_number_of_electrons(self) -> torch.Tensor:
        """Return *scalar* Tr(D·S)."""
        return trace_matmul_sparse_block_matrix(self.density, self.overlap)

    def get_energy(self) -> torch.Tensor:
        """Return *scalar* Tr(D·H)."""
        return trace_matmul_sparse_block_matrix(self.hamiltonian, self.density)

    def symmetrize_matrices(
        self,
        *,
        hamiltonian: bool = True,
        overlap: bool = True,
        density: bool = True,
    ) -> "Snapshot":
        """
        Return a new snapshot with selected matrices symmetrized by 0.5 * (M + M^T).
        """
        ham = (
            0.5 * (self.hamiltonian + self.hamiltonian.transpose())
            if hamiltonian
            else self.hamiltonian
        )
        ovl = (
            0.5 * (self.overlap + self.overlap.transpose()) if overlap else self.overlap
        )
        den = (
            0.5 * (self.density + self.density.transpose()) if density else self.density
        )
        return Snapshot(
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

    # ---------------------------------------------------------------- serialisation
    def _payload(self):
        return {
            "positions": self.positions.cpu() if self.positions is not None else None,
            "forces": self.forces.cpu() if self.forces is not None else None,
            "box": self.box.cpu() if self.box is not None else None,
            "stress": self.stress.cpu() if self.stress is not None else None,
            "matrix_path": self.matrix_path,
            "info_path": self.info_path,
            "cutoff_radius": self.cutoff_radius,
            "mats": {k: v._to_payload() for k, v in self._mats.items()},
        }

    def save(self, path: str | os.PathLike) -> None:
        torch.save(self._payload(), path)

    # helper to reconstruct one BlockMatrix from saved payload ----------
    @staticmethod
    def _matrix_from_payload(payload: Dict[str, Any], device="cpu") -> BlockMatrix:
        from core.orbital_irrep_config import OrbitalIrrepConfig
        from collections import Counter

        orb_cfg = OrbitalIrrepConfig.from_dict(payload["orbital_cfg"])

        pair_blocks = {k: v.to(device) for k, v in payload["pair_blocks"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup
        lookup = {}
        for key, edges in pair_edges.items():
            for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, idx)

        atoms = tuple(payload["atoms"])
        atom_counts = payload.get("atom_counts", Counter(atoms))

        return BlockMatrix(
            atoms=atoms,
            atom_counts=atom_counts,
            pair_blocks=pair_blocks,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=orb_cfg,
            basis=payload.get("basis", "openmx"),
        )

    # public classmethod ----------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike, *, device="cpu") -> "Snapshot":
        payload_top = torch.load(path, map_location="cpu", weights_only=False)
        mats = {
            name: cls._matrix_from_payload(pld, device)
            for name, pld in payload_top["mats"].items()
        }
        forces = payload_top.get("forces", None)
        if forces is not None:
            forces = forces.to(device)
        pos = payload_top.get("positions", None)
        if pos is not None:
            pos = pos.to(device)
        box = payload_top.get("box", None)
        if box is not None:
            box = box.to(device)
        stress = payload_top.get("stress", None)
        if stress is not None:
            stress = stress.to(device)
        return cls(
            mats["hamiltonian"],
            mats["overlap"],
            mats["density"],
            positions=pos,
            forces=forces,
            box=box,
            stress=stress,
            matrix_path=payload_top.get("matrix_path", None),
            info_path=payload_top.get("info_path", None),
            cutoff_radius=payload_top.get("cutoff_radius", None),
        )

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        if len(self.density.atoms) < 10:
            return (
                "Snapshot(\n"
                f"  atoms   = {''.join(self.density.atoms)}\n"
                f"  keys    = {sorted(self.density.keys())}\n"
                f"  basis   = {self.density.basis}\n"
                ")"
            )
        else:
            return (
                "Snapshot(\n"
                f"  atoms   = {len(self.density.atoms)} atoms\n"
                f"  keys    = {sorted(self.density.keys())[:10]} …\n"
                f"  basis   = {self.density.basis}\n"
                ")"
            )

    def _change_basis(self, target: str) -> "Snapshot":
        """
        Return a **new** snapshot in `target` basis
        ("openmx" | "e3nn" | "fhi-aims" | "pyscf").
        If already in that basis the current instance is returned unchanged.
        """
        if target not in {"openmx", "e3nn", "fhi-aims", "pyscf"}:
            raise ValueError("target must be 'openmx', 'e3nn', 'fhi-aims', or 'pyscf'")

        if self.density.basis == target:
            return self

        cfg = self.density.orbital_cfg
        any_block = next(iter(self.hamiltonian.pair_blocks.values()))
        device = any_block.device
        pos = self.positions
        forces = self.forces
        box = self.box
        stress = self.stress

        if self.density.basis == "openmx" and target == "e3nn":
            conv = OpenMXE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_e3nn(self.hamiltonian)
            ovl = conv.matrix_to_e3nn(self.overlap)
            den = conv.matrix_to_e3nn(self.density)

            cob_dtype = (
                pos.dtype if pos is not None else any_block.dtype
            )  # keep basis change in tensor dtype
            change_of_basis = torch.eye(3, dtype=cob_dtype, device=device)[[2, 0, 1]]
            pos = pos @ change_of_basis if pos is not None else None
            forces = forces @ change_of_basis if forces is not None else None
            box = box @ change_of_basis if box is not None else None
            stress = (
                change_of_basis.T @ stress @ change_of_basis
                if stress is not None and stress.shape == torch.Size([3, 3])
                else stress
            )
        elif self.density.basis == "e3nn" and target == "openmx":
            conv = OpenMXE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_openmx(self.hamiltonian)
            ovl = conv.matrix_to_openmx(self.overlap)
            den = conv.matrix_to_openmx(self.density)

            cob_dtype = (
                pos.dtype if pos is not None else any_block.dtype
            )  # keep basis change in tensor dtype
            change_of_basis = torch.eye(3, dtype=cob_dtype, device=device)[[1, 2, 0]]
            pos = pos @ change_of_basis if pos is not None else None
            forces = forces @ change_of_basis if forces is not None else None
            box = box @ change_of_basis if box is not None else None
            stress = (
                change_of_basis.T @ stress @ change_of_basis
                if stress is not None and stress.shape == torch.Size([3, 3])
                else stress
            )
        elif self.density.basis == "fhi-aims" and target == "e3nn":
            conv = FHIaimsE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_e3nn(self.hamiltonian)
            ovl = conv.matrix_to_e3nn(self.overlap)
            den = conv.matrix_to_e3nn(self.density)
        elif self.density.basis == "e3nn" and target == "fhi-aims":
            conv = FHIaimsE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_fhiaims(self.hamiltonian)
            ovl = conv.matrix_to_fhiaims(self.overlap)
            den = conv.matrix_to_fhiaims(self.density)
        elif self.density.basis == "pyscf" and target == "e3nn":
            conv = PySCFE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_e3nn(self.hamiltonian)
            ovl = conv.matrix_to_e3nn(self.overlap)
            den = conv.matrix_to_e3nn(self.density)
        elif self.density.basis == "e3nn" and target == "pyscf":
            conv = PySCFE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_pyscf(self.hamiltonian)
            ovl = conv.matrix_to_pyscf(self.overlap)
            den = conv.matrix_to_pyscf(self.density)
        else:
            raise RuntimeError("Unsupported basis conversion")

        return Snapshot(
            ham,
            ovl,
            den,
            positions=pos,
            forces=forces,
            box=box,
            stress=stress,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
            cfg=self.cfg,
            info=self.info,
        )

    # public façade --------------------------------------------------------
    def to_e3nn(self) -> "Snapshot":
        """Return a (possibly new) Snapshot in the **E3NN** convention."""
        return self._change_basis("e3nn")

    def to_openmx(self) -> "Snapshot":
        """Return a (possibly new) Snapshot in the **OpenMX** convention."""
        return self._change_basis("openmx")

    def to_fhiaims(self) -> "Snapshot":
        """Return a (possibly new) Snapshot in the **FHI-AIMS** convention."""
        return self._change_basis("fhi-aims")

    def to_pyscf(self) -> "Snapshot":
        """Return a (possibly new) Snapshot in the raw **PySCF** AO convention."""
        return self._change_basis("pyscf")

    # ---------------------------------------------------------------- rotation
    def rotate(self, R: torch.Tensor) -> "Snapshot":
        """
        Return a new **rotated** snapshot where all three matrices have been
        rotated by the same 3x3 matrix **R**.
        """
        ham = self.hamiltonian.rotate(R)
        ovl = self.overlap.rotate(R)
        den = self.density.rotate(R)
        return Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions @ R.T if self.positions is not None else None,
            forces=self.forces @ R.T if self.forces is not None else None,
            box=self.box @ R.T if self.box is not None else None,
            stress=(
                R @ self.stress @ R.T
                if self.stress is not None and self.stress.shape == torch.Size([3, 3])
                else None
            ),
            matrix_path=None,
            info_path=None,
            cutoff_radius=self.cutoff_radius,
            cfg=self.cfg,
            info=self.info,
        )

    # -------------------------------------------------------------------- helpers
    def _edge_displacements(
        self, mat: BlockMatrix | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return dict ``key -> (E,3)`` of minimal-image displacement vectors.
        """
        if self.positions is None or self.box is None:
            raise RuntimeError("Snapshot has no position/box information")

        if mat is None:
            mat = self.density

        # inv_box = torch.inverse(self.box)
        vecs: Dict[str, torch.Tensor] = {}

        for key, edges in mat.pair_edges.items():
            sx, sy, sz, src, dst = edges
            edge_shift = (
                torch.stack([sx, sy, sz], dim=-1)
                .to(self.positions.device)
                .to(self.positions.dtype)
            )
            delta = self.positions[dst] - self.positions[src] + edge_shift @ self.box

            # frac = delta @ inv_box
            # frac_wrapped = frac - torch.round(frac)
            # vecs[key] = frac_wrapped @ self.box
            vecs[key] = delta

        return vecs

    def _edge_distances(
        self, mat: BlockMatrix | None = None
    ) -> Dict[str, torch.Tensor]:
        """Return dict ``key -> (E,)`` with minimal-image distances."""
        disp = self._edge_displacements(mat)
        return {k: torch.linalg.norm(v, dim=-1) for k, v in disp.items()}

    # -------------------- public API -------------------------------------------
    def max_distance(self, which: str = "density") -> torch.Tensor:
        """
        Largest minimal-image distance appearing in *which* sparse matrix.
        """
        mat = self._mats[which]
        d = self._edge_distances(mat)
        return torch.stack([v.max() for v in d.values()]).max()

    def filter_by_distance(self, cutoff: float, which: str = "density") -> "Snapshot":
        """
        Return a **new** snapshot where edges whose minimal-image distance
        exceeds ``cutoff`` (Å) are removed *in **all** three matrices*.
        """
        dist = self._edge_distances(self._mats[which])
        mask_dict = {k: (v <= cutoff) for k, v in dist.items()}

        ham = self.hamiltonian._apply_edge_mask(mask_dict)
        ovl = self.overlap._apply_edge_mask(mask_dict)
        den = self.density._apply_edge_mask(mask_dict)

        return Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions,
            forces=self.forces,
            box=self.box,
            stress=self.stress,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=cutoff,
            cfg=self.cfg,
            info=self.info,
        )

    def to_k_space(
        self,
        *,
        kpoints_abs: torch.Tensor,
        kmesh: tuple[int, int, int] | None = None,
        shifts: torch.Tensor | None = None,
    ):
        from data.kspace_snapshot import KSpaceSnapshot

        return KSpaceSnapshot.from_shift_space(
            self,
            kpoints_abs=kpoints_abs,
            kmesh=kmesh,
            shifts=shifts,
        )

    def get_translation_shifts(self) -> torch.Tensor:
        shift_set: set[tuple[int, int, int]] = set()
        for mat in self._mats.values():
            for edges in mat.pair_edges.values():
                for sx, sy, sz in edges[:3].t().tolist():
                    shift_set.add((int(sx), int(sy), int(sz)))
        if not shift_set:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.tensor(sorted(shift_set), dtype=torch.long)

    def get_band_structure(
        self,
        *,
        path: str | None = None,
        special_points: (
            dict[str, list[float] | tuple[float, float, float]] | None
        ) = None,
        npoints: int = 200,
        kpoints_abs: torch.Tensor | None = None,
        fractional_kpoints: torch.Tensor | None = None,
        linear_k: torch.Tensor | None = None,
        tick_positions: torch.Tensor | None = None,
        tick_labels: list[str] | tuple[str, ...] | None = None,
        shifts: torch.Tensor | None = None,
        chunk_size: int | None = None,
        show_progress: bool = False,
        psd_cleanup: bool = False,
        allow_jitter: bool = False,
    ):
        if self.box is None:
            raise ValueError("Snapshot needs a periodic box to compute band structure.")

        from data.kspace_snapshot import (
            BandStructure,
            _linear_k_axis,
            _generalized_eigenvalues_kspace,
            block_matrix_to_shiftspace_dense,
            build_band_path,
        )

        progress = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:  # pragma: no cover - optional dependency
                tqdm = None
            if tqdm is not None:
                progress = tqdm

        fractional_kpoints_t = fractional_kpoints
        linear_k_t = linear_k
        tick_positions_t = tick_positions
        tick_labels_t = tick_labels
        if kpoints_abs is None:
            if fractional_kpoints_t is not None:
                reciprocal = 2 * torch.pi * torch.linalg.inv(self.box).T
                kpoints_abs = (
                    fractional_kpoints_t.to(
                        device=self.box.device,
                        dtype=self.box.dtype,
                    )
                    @ reciprocal
                )
            else:
                (
                    fractional_kpoints_t,
                    kpoints_abs,
                    linear_k_t,
                    tick_positions_t,
                    tick_labels_t,
                ) = build_band_path(
                    self.box,
                    path=path,
                    special_points=special_points,
                    npoints=npoints,
                )
        if kpoints_abs is None:
            raise ValueError("Could not resolve kpoints for band structure.")
        shift_t = self.get_translation_shifts() if shifts is None else shifts
        if shift_t.numel() == 0:
            raise ValueError("Snapshot does not contain any translation shifts.")
        shift_t = shift_t.to(device=self.box.device)

        kpoints_abs = kpoints_abs.to(device=self.box.device, dtype=self.box.dtype)

        ham_shift = block_matrix_to_shiftspace_dense(self.hamiltonian, shifts=shift_t)
        ovl_shift = block_matrix_to_shiftspace_dense(self.overlap, shifts=shift_t)

        nk = int(kpoints_abs.shape[0])
        effective_chunk = (
            nk if chunk_size is None or chunk_size <= 0 else int(chunk_size)
        )
        iterator = range(0, nk, effective_chunk)
        if progress is not None and nk > effective_chunk:
            iterator = progress(
                iterator,
                total=(nk + effective_chunk - 1) // effective_chunk,
                desc="Band chunks",
            )

        eigen_chunks = []
        for start in iterator:
            stop = min(start + effective_chunk, nk)
            k_chunk = kpoints_abs[start:stop]
            ham_k = shiftspace_to_kspace_dense(
                ham_shift,
                kpoints_abs=k_chunk,
                shifts=shift_t,
                box=self.box,
            )
            ovl_k = shiftspace_to_kspace_dense(
                ovl_shift,
                kpoints_abs=k_chunk,
                shifts=shift_t,
                box=self.box,
            )
            eigen_chunks.append(
                _generalized_eigenvalues_kspace(
                    ham_k,
                    ovl_k,
                    psd_cleanup=psd_cleanup,
                    allow_jitter=allow_jitter,
                )
            )

        eigenvalues = torch.cat(eigen_chunks, dim=0)
        linear_axis = (
            linear_k_t.to(device=kpoints_abs.device, dtype=kpoints_abs.dtype)
            if linear_k_t is not None
            else _linear_k_axis(kpoints_abs)
        )

        fermi_level = None
        if getattr(self.info, "fermi_level", None) is not None:
            fermi_level = self.info.fermi_level.to(
                device=eigenvalues.device,
                dtype=eigenvalues.real.dtype,
            )

        return BandStructure(
            eigenvalues=eigenvalues,
            kpoints_abs=kpoints_abs,
            linear_k=linear_axis,
            tick_positions=(
                tick_positions_t.to(device=kpoints_abs.device, dtype=kpoints_abs.dtype)
                if tick_positions_t is not None
                else torch.tensor(
                    [float(linear_axis[0].item()), float(linear_axis[-1].item())],
                    device=kpoints_abs.device,
                    dtype=kpoints_abs.dtype,
                )
            ),
            tick_labels=(
                ["X0", "X1"]
                if tick_labels_t is None
                else [str(label) for label in tick_labels_t]
            ),
            fractional_kpoints=fractional_kpoints_t,
            fermi_level=fermi_level,
        )

    @classmethod
    def _parse_orbital_selection(cls, spec: str) -> Dict[int, int]:
        """
        Parse strings like "1s1p", "2s3p", "1s+1p" into counts by l.
        """
        if not isinstance(spec, str):
            raise TypeError(f"Selection spec must be str, got {type(spec)}")

        cleaned = (
            spec.strip().lower().replace(" ", "").replace("+", "").replace(",", "")
        )
        if not cleaned:
            raise ValueError("Empty orbital selection spec")

        counts: Dict[int, int] = {}
        pos = 0
        for match in re.finditer(r"(\d+)([spdfghiklm])", cleaned):
            if match.start() != pos:
                raise ValueError(f"Invalid orbital selection spec: '{spec}'")
            n = int(match.group(1))
            l = cls._ORBITAL_TO_L[match.group(2)]
            counts[l] = counts.get(l, 0) + n
            pos = match.end()
        if pos != len(cleaned):
            raise ValueError(f"Invalid orbital selection spec: '{spec}'")
        return counts

    def reduce_orbitals(
        self,
        selection: str | Mapping[str, str] | None = None,
        *,
        strict: bool = True,
        keep_unspecified: bool = True,
    ) -> "Snapshot":
        """
        Return a new Snapshot with reduced orbital basis per element.

        Selection examples:
        - "1s1p": apply to all elements
        - {"O": "2s3p", "H": "1s"}: per-element selection

        Selection is prefix/in-order per orbital type: if an element has multiple
        sets of a given l, the first n sets are kept.
        """
        if selection is None:
            return self

        full_cfg = self.hamiltonian.orbital_cfg
        elements = full_cfg.elements()

        if isinstance(selection, str):
            selection_map = {el: selection for el in elements}
        elif isinstance(selection, Mapping):
            selection_map = dict(selection)
        else:
            raise TypeError(
                "selection must be None, a string like '1s1p', or mapping {element: spec}"
            )

        unknown_elements = set(selection_map.keys()) - set(elements)
        if unknown_elements:
            raise ValueError(
                f"Selection provided for unknown elements: {sorted(unknown_elements)}"
            )

        elem_keep_indices: Dict[str, torch.Tensor] = {}
        reduced_irreps: Dict[str, Irreps] = {}

        for el in elements:
            irreps_el = full_cfg.element_to_irreps[el]

            if el in selection_map:
                request_by_l = self._parse_orbital_selection(selection_map[el])
            else:
                request_by_l = None if keep_unspecified else {}

            remaining = None if request_by_l is None else dict(request_by_l)

            keep_indices: list[int] = []
            new_terms: list[tuple[int, object]] = []
            offset = 0

            for mul, ir in irreps_el:
                l = ir.l
                dim = ir.dim
                keep_mul = 0

                for _ in range(mul):
                    keep_this = False
                    if remaining is None:
                        keep_this = True
                    else:
                        want = remaining.get(l, 0)
                        if want > 0:
                            keep_this = True
                            remaining[l] = want - 1

                    if keep_this:
                        keep_indices.extend(range(offset, offset + dim))
                        keep_mul += 1
                    offset += dim

                if keep_mul > 0:
                    new_terms.append((keep_mul, ir))

            if remaining is not None and strict:
                missing = {l: cnt for l, cnt in remaining.items() if cnt > 0}
                if missing:
                    missing_str = ", ".join(
                        [f"l={l}:{cnt}" for l, cnt in sorted(missing.items())]
                    )
                    raise ValueError(
                        f"Element '{el}' does not have enough requested orbitals ({missing_str})"
                    )

            if not keep_indices:
                raise ValueError(
                    f"Orbital reduction removed all orbitals for element '{el}'"
                )

            elem_keep_indices[el] = torch.tensor(keep_indices, dtype=torch.long)
            reduced_irreps[el] = Irreps(new_terms)

        reduced_cfg = OrbitalIrrepConfig(reduced_irreps)

        def _reduce_matrix(mat: BlockMatrix) -> BlockMatrix:
            new_pair_blocks: Dict[str, torch.Tensor] = {}
            for key, blocks in mat.pair_blocks.items():
                el_i, el_j = key.split("-")
                idx_i = elem_keep_indices[el_i].to(device=blocks.device)
                idx_j = elem_keep_indices[el_j].to(device=blocks.device)
                reduced = blocks.index_select(1, idx_i).index_select(2, idx_j)
                new_pair_blocks[key] = reduced
            return BlockMatrix(
                atoms=mat.atoms,
                atom_counts=mat.atom_counts,
                pair_blocks=new_pair_blocks,
                pair_edges=mat.pair_edges,
                lookup=mat.lookup,
                orbital_cfg=reduced_cfg,
                basis=mat.basis,
            )

        return Snapshot(
            _reduce_matrix(self.hamiltonian),
            _reduce_matrix(self.overlap),
            _reduce_matrix(self.density),
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

    # ---------------------------------------------------------------- dunder access
    def __getitem__(self, item: str) -> BlockMatrix:
        return self._mats[item]

    def __getattr__(self, name: str) -> Any:
        if name in self._mats:
            return self._mats[name]
        raise AttributeError(name)

    # -------------------------------------------------------------------- constructors
    @staticmethod
    def from_openmx(
        matrix_path: str | os.PathLike,
        info_path: str | os.PathLike,
        *,
        convention: str = "e3nn",
        symmetrize_density: bool = True,
        cutoff_radius: float | None = None,
        dtype: torch.dtype = torch.float32,
        cfg: Config = None,
    ) -> "Snapshot":
        from data.openmx_info_parser import parse_info_out
        from data.openmx_parser import parse_openmx_scfout

        info = parse_info_out(info_path, dtype)
        matrix_path = Path(matrix_path)
        info_path = Path(info_path)
        atoms: list[str] = info.elements
        if not atoms:
            raise RuntimeError("Info-file does not contain <coordinates.forces>")

        orb_cfg = OrbitalIrrepConfig.from_dict(info.orbital_set)

        snap = parse_openmx_scfout(
            matrix_path,
            atoms,
            orb_cfg,
            convention="openmx",
            symmetrize_density=symmetrize_density,
        )

        snap.matrix_path = matrix_path
        snap.info_path = info_path
        cif_path = info_path.with_suffix(".cif")
        if cif_path.exists():
            cif = ase_read(str(cif_path))
            cif_atoms = list(cif.get_chemical_symbols())
            if cif_atoms != atoms:
                raise RuntimeError(
                    f"CIF atom order does not match OpenMX atom order for {info_path}. "
                    f"OpenMX atoms={atoms[:8]}... (n={len(atoms)}), "
                    f"CIF atoms={cif_atoms[:8]}... (n={len(cif_atoms)})"
                )
            cif_positions = torch.tensor(cif.get_positions(), dtype=dtype)
            cif_box = torch.tensor(cif.cell.array, dtype=dtype)
            atom_shifts = _compute_atom_image_shifts(
                info.positions.to(dtype=dtype),
                cif_positions,
                cif_box,
            )
            snap = snap.relabel_atom_images(atom_shifts)
            snap.positions = cif_positions
            snap.box = cif_box
        elif cfg is not None and getattr(
            cfg, "allow_openmx_positions_box_from_out", False
        ):
            if not info.positions.numel() or not info.box.numel():
                raise RuntimeError(
                    f"OpenMX fallback requested but info file does not contain positions/box: {info_path}"
                )
            snap.positions = info.positions
            snap.box = info.box
        else:
            raise FileNotFoundError(
                f"Missing CIF file for OpenMX geometry: {cif_path}. "
                "This loader now expects CIF geometry by default. "
                "Set cfg.allow_openmx_positions_box_from_out=True to fall back to the .out file geometry."
            )
        snap.forces = info.forces if info.forces.numel() else None
        snap.stress = info.stress if info.stress.numel() else None
        snap.cfg = cfg
        snap.info = info

        if cutoff_radius is not None:
            snap = snap.filter_by_distance(cutoff_radius)

        snap = snap._change_basis(convention)

        snap = snap.canonicalize_edges()
        return snap

    @staticmethod
    def from_pyscf(
        npz_path: str | os.PathLike,
        json_path: str | os.PathLike | None = None,
        *,
        convention: str = "e3nn",
        cutoff_radius: float | None = None,
        dtype: torch.dtype = torch.float32,
        cfg: Config = None,
    ) -> "Snapshot":
        from data.pyscf_baseline_parser import load_pyscf_snapshot

        snap = load_pyscf_snapshot(
            npz_path=npz_path,
            json_path=json_path,
            dtype=dtype,
            basis="pyscf",
        )
        snap.cfg = cfg

        if cutoff_radius is not None:
            snap = snap.filter_by_distance(cutoff_radius)

        snap = snap._change_basis(convention)
        snap = snap.canonicalize_edges()
        return snap

    @staticmethod
    def from_pyscf_kspace(
        npz_path: str | os.PathLike,
        json_path: str | os.PathLike | None = None,
        *,
        convention: str = "e3nn",
        dtype: torch.dtype = torch.float32,
        cfg: Config = None,
    ):
        from data.kspace_snapshot import load_pyscf_kspace_snapshot

        snap = load_pyscf_kspace_snapshot(
            npz_path=npz_path,
            json_path=json_path,
            dtype=dtype,
            basis="pyscf",
        )
        snap.cfg = cfg
        snap = snap.change_basis(convention)
        return snap

    @staticmethod
    def from_fhiaims(
        geometry_path: str | os.PathLike,
        basis_path: str | os.PathLike,
        hamiltonian_path: str | os.PathLike,
        overlap_path: str | os.PathLike,
        density_path: str | os.PathLike,
        *,
        convention: str = "e3nn",
        cutoff_radius: float | None = None,
        cfg: Config = None,
    ) -> "Snapshot":
        from data.fhiaims_parser import parse_fhiaims_output

        snap = parse_fhiaims_output(
            geometry_path,
            basis_path,
            hamiltonian_path,
            overlap_path,
            density_path,
        )

        snap.cfg = cfg

        if convention == "e3nn":
            snap = snap.to_e3nn()

        if cutoff_radius is not None:
            snap = snap.filter_by_distance(cutoff_radius)

        snap = snap.canonicalize_edges()
        return snap

    def dos(self, sigma=0.005, bin_width=0.001, E_min=-1.0, E_max=1.0):
        """
        Compute the density of states (DOS) for this snapshot.

        Parameters
        ----------
        sigma : float
            The broadening parameter for the DOS.
        bin_width : float
            The width of each energy bin.
        E_min : float
            The minimum energy to consider.
        E_max : float
            The maximum energy to consider.

        Returns a dict with keys:
        "energies" : torch.Tensor
            The energy bins.
        "dos" : torch.Tensor
            The computed density of states for each energy bin.
        """
        H = self.hamiltonian.to_dense().detach()
        S = self.overlap.to_dense().detach()

        # Generalized eigensolve for H x = lambda S x via Cholesky reduction.
        # Symmetrization improves numerical robustness.
        H = 0.5 * (H + H.T)
        S = 0.5 * (S + S.T)
        L = torch.linalg.cholesky(S)
        tmp = torch.linalg.solve(L, H)
        A = torch.linalg.solve(L, tmp.T).T  # A = L^{-1} H L^{-T}
        A = 0.5 * (A + A.T)

        eigenvalues = torch.linalg.eigvalsh(A)
        # Energy bins
        grid = torch.arange(
            E_min,
            E_max + bin_width,
            bin_width,
            dtype=eigenvalues.dtype,
            device=eigenvalues.device,
        )
        dos = torch.sum(
            torch.exp(-((grid[:, None] - eigenvalues[None, :]) ** 2) / (2 * sigma**2)),
            axis=1,
        ) / (
            torch.sqrt(
                torch.tensor(
                    2 * torch.pi, dtype=eigenvalues.dtype, device=eigenvalues.device
                )
            )
            * sigma
        )

        return grid, dos

    def export_to_deephe3(self, path: str | os.PathLike):
        """
        Export the snapshot to the DeepH-E3 format.

        Parameters
        ----------
        path : str | os.PathLike
            The directory where the files will be saved.
        """
        path = Path(path)
        os.makedirs(path, exist_ok=True)

        # save_element
        atoms = Atoms(self.hamiltonian.atoms)
        with open(path / "element.dat", "w") as f:
            for number in atoms.numbers:
                f.write(f"{number}\n")

        # save_info
        with open(path / "info.json", "w") as f:
            json.dump(
                {"fermi_level": self.info.fermi_level.item(), "isspinful": False}, f
            )

        # save_lat
        with open(path / "lat.dat", "w") as f:
            for row in self.box:
                for el in row:
                    f.write(f"{el} ")
                f.write("\n")

        # save_rlat
        rlat = 2 * torch.pi * torch.linalg.inv(self.box).T
        with open(path / "rlat.dat", "w") as f:
            for row in rlat:
                for el in row:
                    f.write(f"{el} ")
                f.write("\n")

        # save_site_positions
        with open(path / "site_positions.dat", "w") as f:
            for row in self.positions.T:
                for el in row:
                    f.write(f"{el}\t")
                f.write("\n")

        # save_hamiltonians
        with h5py.File(path / "hamiltonians.h5", "w") as f:
            for key in self.hamiltonian.keys():
                edges = self.hamiltonian.pair_edges[key]
                blocks = self.hamiltonian.pair_blocks[key]

                # edges rows: 0:sx, 1:sy, 2:sz, 3:src, 4:dst
                for i in range(edges.shape[1]):
                    sx = edges[0, i].item()
                    sy = edges[1, i].item()
                    sz = edges[2, i].item()
                    src = edges[3, i].item()
                    dst = edges[4, i].item()

                    # Key format: [sx, sy, sz, src, dst]
                    name = str([sx, sy, sz, src, dst])
                    f.create_dataset(
                        name, data=blocks[i].to(torch.float64).cpu().numpy()
                    )

        # save_orbital_types
        with open(path / "orbital_types.dat", "w") as f:
            for atom in self.hamiltonian.atoms:
                orbital_list = self.hamiltonian.orbital_cfg.element_to_irreps[atom].ls
                for l in orbital_list:
                    f.write(f"{l}\t")
                f.write("\n")
