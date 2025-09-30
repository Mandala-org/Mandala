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

    * ``get_number_of_electrons()``  →  Tr(D·S)
    * ``get_energy()``               →  Tr(D·H)

  using the highly-optimised
  :func:`core.sparse_math.trace_matmul_sparse_snap_vectorized`.
* Simple ``save(path)`` / ``load(path)`` round-trip based on the underlying
  Block snapshots' serialisation.
"""

from __future__ import annotations

import os
from typing import Dict, Any
import torch

from core.sparse_math import (
    trace_matmul_sparse_snap,
)
from data.block_matrix import BlockMatrix
from core.basis_converter import OpenMXE3NNConverter, FHIaimsE3NNConverter
from core.orbital_irrep_config import OrbitalIrrepConfig

__all__ = ["Snapshot"]


class Snapshot:
    """Bundle H, S, D for one configuration and provide physics helpers."""

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
        1. Off-diagonal edges, sorted by the L2 norm of their corresponding
           density matrix block in ascending order.
        2. Diagonal edges, sorted by their node index.
        """
        density = self._mats["density"]
        order_dict = {}

        for key, edges in density.pair_edges.items():
            is_diag_mask = edges[0] == edges[1]

            # Get the sorting permutation for the diagonal edges
            perm_diag = torch.argsort(edges[0] + edges[1] * 1e6)
            diag_mask = is_diag_mask[perm_diag]
            perm_diag = perm_diag[diag_mask]

            # Sort off-diagonal edges by density norm
            norms = density.pair_blocks[key].pow(2).sum(dim=(-2, -1)).sqrt()
            # Get the sorting permutation for the off-diagonal edges
            perm_offdiag = torch.argsort(norms)
            offdiag_mask = ~is_diag_mask[perm_offdiag]
            perm_offdiag = perm_offdiag[offdiag_mask]

            order_dict[key] = torch.cat([perm_offdiag, perm_diag])

        # Apply the SAME permutation to every matrix
        new_mats = {
            name: mat.reorder_edges(order_dict) for name, mat in self._mats.items()
        }

        return Snapshot(
            new_mats["hamiltonian"],
            new_mats["overlap"],
            new_mats["density"],
            positions=self.positions,
            forces=self.forces,
            box=self.box,
            stress=self.stress,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
        )

    # ---------------------------------------------------------------- physics helpers
    def get_number_of_electrons(self) -> torch.Tensor:
        """Return *scalar* Tr(D·S)."""
        return trace_matmul_sparse_snap(self.density, self.overlap)

    def get_energy(self) -> torch.Tensor:
        """Return *scalar* Tr(D·H)."""
        return trace_matmul_sparse_snap(self.hamiltonian, self.density)

    # ---------------------------------------------------------------- serialisation
    def _payload(self):
        return {
            "positions": self.positions.cpu() if self.positions is not None else None,
            "forces": self.forces.cpu() if self.forces is not None else None,
            "box": self.box.cpu() if self.box is not None else None,
            "stress": self.stress.cpu() if self.box is not None else None,
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
            for idx, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, idx)

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
        payload_top = torch.load(path, map_location="cpu")
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
        Return a **new** snapshot in `target` basis ("openmx" | "e3nn" | "fhi-aims").
        If already in that basis the current instance is returned unchanged.
        """
        if target not in {"openmx", "e3nn", "fhi-aims"}:
            raise ValueError("target must be 'openmx', 'e3nn', or 'fhi-aims'")

        if self.density.basis == target:
            return self

        cfg = self.density.orbital_cfg
        any_block = next(iter(self.density.pair_blocks.values()))
        device = any_block.device
        pos = self.positions
        forces = self.forces
        box = self.box

        if self.density.basis == "openmx" and target == "e3nn":
            conv = OpenMXE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_e3nn(self.hamiltonian)
            ovl = conv.matrix_to_e3nn(self.overlap)
            den = conv.matrix_to_e3nn(self.density)
            # pos = pos @ torch.eye(3, dtype=torch.float32)[[2, 0, 1]]
            # forces = (
            #     forces @ torch.eye(3, dtype=torch.float32)[[2, 0, 1]]
            #     if forces is not None
            #     else None
            # )
            # box = (
            #     box @ torch.eye(3, dtype=torch.float32)[[2, 0, 1]]
            #     if box is not None
            #     else None
            # )
            print("Changing convention with [1, 2, 0]")
            pos = pos @ torch.eye(3, dtype=torch.float32)[[1, 2, 0]]
            forces = (
                forces @ torch.eye(3, dtype=torch.float32)[[1, 2, 0]]
                if forces is not None
                else None
            )
            box = (
                box @ torch.eye(3, dtype=torch.float32)[[1, 2, 0]]
                if box is not None
                else None
            )
        elif self.density.basis == "fhi-aims" and target == "e3nn":
            conv = FHIaimsE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_e3nn(self.hamiltonian)
            ovl = conv.matrix_to_e3nn(self.overlap)
            den = conv.matrix_to_e3nn(self.density)
        elif self.density.basis == "e3nn" and target == "openmx":
            conv = OpenMXE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_openmx(self.hamiltonian)
            ovl = conv.matrix_to_openmx(self.overlap)
            den = conv.matrix_to_openmx(self.density)
            print(
                "Warning: Position/force/box conversion from e3nn to openmx to be checked!"
            )
            pos = pos @ torch.eye(3, dtype=torch.float32)[[1, 2, 0]]
            forces = (
                forces @ torch.eye(3, dtype=torch.float32)[[1, 2, 0]]
                if forces is not None
                else None
            )
            box = (
                box @ torch.eye(3, dtype=torch.float32)[[1, 2, 0]]
                if box is not None
                else None
            )
        elif self.density.basis == "e3nn" and target == "fhi-aims":
            conv = FHIaimsE3NNConverter(cfg, device=device)
            ham = conv.matrix_to_fhiaims(self.hamiltonian)
            ovl = conv.matrix_to_fhiaims(self.overlap)
            den = conv.matrix_to_fhiaims(self.density)
        else:
            raise RuntimeError("Unsupported basis conversion")

        return Snapshot(
            ham,
            ovl,
            den,
            positions=pos,
            forces=self.forces,
            box=self.box,
            stress=self.stress,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
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

    # ---------------------------------------------------------------- rotation
    def rotate(self, R: torch.Tensor) -> "Snapshot":
        """
        Return a new **rotated** snapshot where all three matrices have been
        rotated by the same 3x3 matrix **R**.
        """
        ham = self.hamiltonian.rotate(R)
        ovl = self.overlap.rotate(R)
        den = self.density.rotate(R)
        if self.stress is not None:
            print("Warning: stress rotation to be checked!")
        return Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions @ R.T if self.positions is not None else None,
            # positions=self.positions @ R if self.positions is not None else None,
            forces=self.forces @ R.T if self.forces is not None else None,
            box=self.box @ R.T if self.box is not None else None,
            stress=(
                self.stress @ R.T
                if self.stress is not None and self.stress.shape != torch.Size([0])
                else None
            ),  # ! Check that it's correct
            matrix_path=None,
            info_path=None,
            cutoff_radius=self.cutoff_radius,
        )

    # -------------------------------------------------------------------- helpers
    def _edge_displacements(
        self, mat: BlockMatrix | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return dict ``key → (E,3)`` of minimal-image displacement vectors.
        """
        if self.positions is None or self.box is None:
            raise RuntimeError("Snapshot has no position/box information")

        if mat is None:
            mat = self.density

        inv_box = torch.inverse(self.box)
        vecs: Dict[str, torch.Tensor] = {}

        for key, edges in mat.pair_edges.items():
            src, dst = edges
            delta = self.positions[dst] - self.positions[src]

            frac = delta @ inv_box
            frac_wrapped = frac - torch.round(frac)

            vecs[key] = frac_wrapped @ self.box

        return vecs

    def _edge_distances(
        self, mat: BlockMatrix | None = None
    ) -> Dict[str, torch.Tensor]:
        """Return dict ``key → (E,)`` with minimal-image distances."""
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
    ) -> "Snapshot":
        from data.openmx_info_parser import parse_info_out
        from data.openmx_parser import parse_openmx_scfout

        info = parse_info_out(info_path, dtype)
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
        snap.positions = info.positions if info.positions.numel() else None
        snap.forces = info.forces if info.forces.numel() else None
        snap.box = info.box if info.box.numel() else None
        snap.stress = info.stress if info.box.numel() else None

        if cutoff_radius is not None:
            snap = snap.filter_by_distance(cutoff_radius)

        snap = snap._change_basis(convention)

        snap = snap.canonicalize_edges()
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
    ) -> "Snapshot":
        from data.fhiaims_parser import parse_fhiaims_output

        snap = parse_fhiaims_output(
            geometry_path,
            basis_path,
            hamiltonian_path,
            overlap_path,
            density_path,
        )

        if convention == "e3nn":
            snap = snap.to_e3nn()

        if cutoff_radius is not None:
            snap = snap.filter_by_distance(cutoff_radius)

        return snap.canonicalize_edges()

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

        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        ### setting up the energy bins
        grid = torch.arange(E_min, E_max + bin_width, bin_width)
        dos = torch.sum(
            torch.exp(-((grid[:, None] - eigenvalues[None, :]) ** 2) / (2 * sigma**2)),
            axis=1,
        ) / (torch.sqrt(torch.tensor(2 * torch.pi)) * sigma)

        return grid, dos
