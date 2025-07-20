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
        box: torch.Tensor | None = None,  # (3,3)
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
        self.box = box  # may be None for non-periodic test cases

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
                    f"Matrices 0 and {i+1} must use the same basis (openmx/e3nn)"
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

        # # Sanity checks to make sure the graph make sense
        # for mat in new_mats.values():
        # lookup = mat.lookup
        # pair_edges = mat.pair_edges
        # pair_blocks = mat.pair_blocks
        # atoms = mat.atoms

        # # 1. Test whether all edges are unique
        # all_edges = set()
        # for edges in pair_edges.values():
        #     for edge in edges.t().tolist():
        #         all_edges.add(tuple(edge))
        # if len(all_edges) != sum(len(edges.t()) for edges in pair_edges.values()):
        #     raise ValueError("Duplicate edges found in pair edges")
        # # 2. Test whether lookup contains all edges
        # if len(lookup) != len(all_edges):
        #     raise ValueError("Lookup size does not match edge count")
        # for (i, j) in all_edges:
        #     if (i, j) not in lookup:
        #         raise ValueError(f"Edge {(i, j)} not found in lookup")
        #     key, idx = lookup[(i, j)]
        #     if key not in pair_blocks or idx >= len(pair_blocks[key]):
        #         raise ValueError(f"Edge {(i, j)} lookup points to invalid block")
        # # 3. Test whether all self-edges are present
        # for i, atom in enumerate(atoms):
        #     key = f"{atom}-{atom}"
        #     if key not in pair_blocks:
        #         raise ValueError(f"Self-edge {key} not found in pair blocks")
        #     if (i, i) not in lookup:
        #         raise ValueError(f"Self-edge lookup for {key} missing")
        # # 4. Test whether graph is symmetric
        # for (i, j) in lookup:
        #     key, idx = lookup[(i, j)]
        #     if (j, i) not in lookup:
        #         raise ValueError(f"Edge {(i, j)} is not symmetric with {(j, i)}")

        return Snapshot(
            new_mats["hamiltonian"],
            new_mats["overlap"],
            new_mats["density"],
            positions=self.positions,
            box=self.box,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
        )

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
        Return a **new** snapshot in `target` basis ("openmx" | "e3nn").
        If already in that basis the current instance is returned unchanged.
        """
        if target not in {"openmx", "e3nn"}:
            raise ValueError("target must be 'openmx' or 'e3nn'")

        if self.density.basis == target:
            return self  # nothing to do

        cfg = self.density.orbital_cfg  # shared by all mats
        # pick an arbitrary block to determine the device
        any_block = next(iter(self.density.pair_blocks.values()))
        conv = OpenMXE3NNConverter(cfg, device=any_block.device)

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
        return Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions,
            box=self.box,
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

    def change_basis(self, d_dict: Dict[str, torch.Tensor]) -> "Snapshot":
        ham = self.hamiltonian.change_basis(d_dict)
        ovl = self.overlap.change_basis(d_dict)
        den = self.density.change_basis(d_dict)
        return Snapshot(
            ham,
            ovl,
            den,
            positions=self.positions,
            box=self.box,
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=self.cutoff_radius,
        )

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
            box=self.box @ R.T if self.box is not None else None,
            matrix_path=None,  # set to None to invalidate the cache
            info_path=None,
            cutoff_radius=self.cutoff_radius,
        )

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
            matrix_path=self.matrix_path,
            info_path=self.info_path,
            cutoff_radius=cutoff,
        )

    # ---------------------------------------------------------------- dunder access
    def __getitem__(self, item: str) -> BlockMatrix:
        return self._mats[item]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401  (# type: ignore[override]
        if name in self._mats:
            return self._mats[name]
        raise AttributeError(name)

    # ════════════════════════════════════════════════════════════════════════
    #                Convenient constructor from OpenMX files
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def from_openmx(
        matrix_path: str | os.PathLike,
        info_path: str | os.PathLike,
        *,
        convention: str = "e3nn",
        symmetrize_density: bool = True,
        cutoff_radius: float | None = None,
    ) -> "Snapshot":
        """
        Build a :class:`Snapshot` directly from an **OpenMX SCF output pair**:

        * ``matrix_path`` - the ``*.scfout`` file containing H/S/D blocks
        * ``info_path``   - the corresponding ``*.out`` / ``*.info`` file
          parsed by :func:`data.openmx_info_parser.parse_info_out`
        """

        from data.openmx_info_parser import parse_info_out
        from data.openmx_parser import parse_openmx_scfout

        # Parse auxiliary info file (atoms, positions, orbital spec…)
        info = parse_info_out(info_path)

        atoms: list[str] = info.elements
        if not atoms:
            raise RuntimeError("Info-file does not contain <coordinates.forces>")

        # orbital_set maps element -> compact string  ("3s2p2d1f")
        orb_cfg = OrbitalIrrepConfig.from_dict(info.orbital_set)

        # ── ②  Let the existing parser build the block-matrix snapshot ──────
        snap = parse_openmx_scfout(
            matrix_path,
            atoms,
            orb_cfg,
            convention=convention,
            symmetrize_density=symmetrize_density,
        )

        snap.matrix_path = matrix_path  # store source file path
        snap.info_path = info_path

        # ── ③  Attach geometry (positions, later box) and return ────────────
        snap.positions = info.xyz if info.xyz.numel() else None
        snap.box = info.box if info.box.numel() else None
        # Users can still `.rotate(...)` / `.filter_by_distance(...)`
        # without PBC if `box` stays *None*.
        if cutoff_radius is not None:
            snap = snap.filter_by_distance(cutoff_radius)

        return snap.canonicalize_edges()
