"""
snapshot.py
============

High-level container that bundles *all* block-sparse matrices (Hamiltonian,
Overlap, Density …) for **one structure** and exposes convenience physics
utilities.

Key features
------------
* Keeps three :class:`BlockMatrix` objects (`hamiltonian`, `overlap`,
  `density`) under a unified interface.  Access via ``snap["density"]`` **or**
  attribute ``snap.density``.
* Upon construction **re-orders edges** for every element-pair by *ascending*
  L2-norm of the density blocks - this yields deterministic ordering and is
  helpful for compression / batching.
* Implements

    * ``get_number_of_electrons()``  →  Tr(D·S)
    * ``get_energy()``                →  Tr(D·H)

  using the highly-optimised
  :func:`core.sparse_math.trace_matmul_sparse_snap_vectorized`.
* Simple ``save(path)`` / ``load(path)`` round-trip based on the underlying
  Block snapshots' serialisation.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import torch

from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.block_matrix import BlockMatrix
from core.basis_converter import OpenMXE3NNConverter

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
        box: torch.Tensor | None = None,  # (3,3)
    ) -> None:
        # quick consistency sanity checks
        self._check_compatibility(hamiltonian, overlap, density)

        self._mats: Dict[str, BlockMatrix] = {
            "hamiltonian": hamiltonian,
            "overlap": overlap,
            "density": density,
        }

        self.positions = positions  # may be None for non-periodic test cases
        self.box = box  # ditto

        # edge ordering according to |D| magnitude -------------------------
        self._order_edges_by_density()

    # ---------------------------------------------------------------- compatibility
    @staticmethod
    def _check_compatibility(*mats: BlockMatrix) -> None:
        atoms = mats[0].atoms
        mapper_cfg = mats[0].mapper.orbital_cfg.to_dict()
        basis = mats[0].basis
        for m in mats[1:]:
            if m.atoms != atoms:
                raise ValueError("All matrices must share the same atom list")
            if m.mapper.orbital_cfg.to_dict() != mapper_cfg:
                raise ValueError("All matrices must share the same orbital config")
            if m.basis != basis:
                raise ValueError("All matrices must use the same basis (openmx/e3nn)")

    # ---------------------------------------------------------------- edge ordering
    def _order_edges_by_density(self) -> None:
        density = self._mats["density"]
        order_dict = {}
        for key, blk in density.pair_blocks.items():
            # L2 magnitude of each block tensor
            mag = blk.pow(2).sum(dim=(-2, -1)).sqrt()
            order_dict[key] = torch.argsort(mag)

        # apply the SAME permutation to every matrix that has that key
        for name, mat in self._mats.items():
            self._mats[name] = mat.reorder_edges(order_dict)

        # expose as attributes after ordering so self.density etc. match
        self.hamiltonian = self._mats["hamiltonian"]
        self.overlap = self._mats["overlap"]
        self.density = self._mats["density"]

    # ---------------------------------------------------------------- physics helpers
    def get_number_of_electrons(self) -> torch.Tensor:
        """Return *scalar* Tr(D·S)."""
        return trace_matmul_sparse_snap_vectorized(self.density, self.overlap)

    def get_energy(self) -> torch.Tensor:
        """Return *scalar* Tr(D·H)."""
        return trace_matmul_sparse_snap_vectorized(self.hamiltonian, self.density)

    # ---------------------------------------------------------------- serialisation
    def _payload(self):
        return {
            "positions": self.positions.cpu() if self.positions is not None else None,
            "box": self.box.cpu() if self.box is not None else None,
            "mats": {k: v._to_payload() for k, v in self._mats.items()},
        }

    def save(self, path: str | os.PathLike) -> None:
        torch.save(self._payload(), path)

    # helper to reconstruct one BlockMatrix from saved payload ----------
    @staticmethod
    def _matrix_from_payload(payload: Dict[str, Any], device="cpu") -> BlockMatrix:
        from core.orbital_irrep_config import OrbitalIrrepConfig
        from core.block_irrep_mapper import BlockIrrepMapper

        orb_cfg = OrbitalIrrepConfig.from_dict(payload["orbital_cfg"])
        mapper = BlockIrrepMapper(orb_cfg, diagonal=payload["diagonal"], device="cpu")

        pair_blocks = {k: v.to(device) for k, v in payload["pair_blocks"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup
        lookup = {}
        for key, edges in pair_edges.items():
            for idx, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, idx)

        return BlockMatrix(
            tuple(payload["atoms"]),
            pair_blocks,
            pair_edges,
            lookup,
            mapper,
            payload.get("basis", "openmx"),
        )

    # public classmethod ----------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike, *, device="cpu") -> "Snapshot":
        payload_top = torch.load(path, map_location="cpu")
        mats = {
            name: cls._matrix_from_payload(pld, device)
            for name, pld in payload_top["mats"].items()
        }
        pos = payload_top.get("positions", None)
        box = payload_top.get("box", None)
        if pos is not None:
            pos = pos.to(device)
        if box is not None:
            box = box.to(device)
        return cls(
            mats["hamiltonian"],
            mats["overlap"],
            mats["density"],
            positions=pos,
            box=box,
        )

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return (
            "Snapshot(\n"
            f"  atoms   = {''.join(self.density.atoms)}\n"
            f"  keys    = {sorted(self.density.keys())}\n"
            f"  basis   = {self.density.basis}\n"
            ")"
        )

    def _change_basis(self, target: str) -> "Snapshot":
        """
        Return a **new** snapshot in `target` basis ("openmx" | "e3nn").
        If already in that basis the current instance is returned unchanged.
        """
        if target not in {"openmx", "e3nn"}:
            raise ValueError("target must be 'openmx' or 'e3nn'")

        if self.density.basis == target:
            return self  # nothing to do

        cfg = self.density.mapper.orbital_cfg  # shared by all mats
        conv = OpenMXE3NNConverter(cfg, device=self.density["H-H"].device)  # any device

        if target == "e3nn":
            ham = conv.matrix_to_e3nn(self.hamiltonian)
            ovl = conv.matrix_to_e3nn(self.overlap)
            den = conv.matrix_to_e3nn(self.density)
        else:  # target == "openmx"
            ham = conv.matrix_to_openmx(self.hamiltonian)
            ovl = conv.matrix_to_openmx(self.overlap)
            den = conv.matrix_to_openmx(self.density)

        # Constructor will re-order edges deterministically (norm is preserved
        # by orthogonal transforms so ordering identical).
        return Snapshot(ham, ovl, den)

    # public façade --------------------------------------------------------
    def to_e3nn(self) -> "Snapshot":
        """Return a (possibly new) Snapshot in the **E3NN** convention."""
        return self._change_basis("e3nn")

    def to_openmx(self) -> "Snapshot":
        """Return a (possibly new) Snapshot in the **OpenMX** convention."""
        return self._change_basis("openmx")

    # ---------------------------------------------------------------- rotation
    def rotate(self, R: torch.Tensor) -> "Snapshot":
        """
        Return a new **rotated** snapshot where all three matrices have been
        rotated by the same 3x3 matrix **R**.
        """
        ham = self.hamiltonian.rotate(R)
        ovl = self.overlap.rotate(R)
        den = self.density.rotate(R)
        return Snapshot(ham, ovl, den)

        # -------------------------------------------------------------------- helpers

    # -------------------- minimal-image displacements ---------------------------
    def _edge_displacements(
        self, mat: BlockMatrix | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return dict ``key → (E,3)`` of minimal-image displacement vectors.

        Requires ``self.positions`` **and** ``self.box``.
        """
        if self.positions is None or self.box is None:
            raise RuntimeError("Snapshot has no position/box information")

        if mat is None:
            mat = self.density  # default

        inv_box = torch.inverse(self.box.to(self.positions))  # (3,3)
        vecs: Dict[str, torch.Tensor] = {}

        for key, edges in mat.pair_edges.items():  # edges (2,E)
            src, dst = edges
            delta = self.positions[dst] - self.positions[src]  # (E,3)

            # fractional coordinates & wrap to (-0.5,0.5]
            frac = delta @ inv_box
            frac_wrapped = frac - torch.round(frac)

            vecs[key] = frac_wrapped @ self.box.to(self.positions)

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
        Largest minimal-image distance appearing in *which* sparse matrix
        (\"hamiltonian\" | \"overlap\" | \"density\").
        """
        mat = self._mats[which]
        d = self._edge_distances(mat)
        return torch.stack([v.max() for v in d.values()]).max()

    def filter_by_distance(self, cutoff: float, which: str = "density") -> "Snapshot":
        """
        Return a **new** snapshot where edges whose minimal-image distance
        exceeds ``cutoff`` (Å) are removed *in **all** three matrices*.

        Edge set is taken from *which* (defaults to \"density\").
        """
        dist = self._edge_distances(self._mats[which])
        mask_dict = {k: (v <= cutoff) for k, v in dist.items()}

        ham = self.hamiltonian._apply_edge_mask(mask_dict)
        ovl = self.overlap._apply_edge_mask(mask_dict)
        den = self.density._apply_edge_mask(mask_dict)

        # Constructor will re-order edges by |D| again
        return Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions,
            box=self.box,
        )

    # ---------------------------------------------------------------- dunder access
    def __getitem__(self, item: str) -> BlockMatrix:
        return self._mats[item]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401  (# type: ignore[override]
        if name in self._mats:
            return self._mats[name]
        raise AttributeError(name)
