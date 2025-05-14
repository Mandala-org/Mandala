"""
snapshot_block.py
=================

In-memory containers for a *single* snapshot:

* **MatrixBlockData** - raw sparse blocks (H, D, S, …) grouped by element pair.
* **IrrepsBlockData** - same data after change-of-basis to irrep vectors.

Both dataclasses keep:
    * `atoms`          : list[str]  (global order)
    * `pair_edges`     : dict[key] → (2,E_ab)  long tensor with global indices
    * `lookup`         : dict[(i,j)] → (key, local_idx)  for O(1) access
    * reference to the shared `BlockIrrepMapper`

Conversion between the two is a one-liner via `.to_vectors()` / `.to_blocks()`.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.basis_converter import OpenMXE3NNConverter


import torch

from core.block_irrep_mapper import BlockIrrepMapper

PairKey = str  # canonical "A-B"


# --------------------------------------------------------------------------- #
@dataclass
class MatrixBlockData:
    atoms: Tuple[str, ...]
    pair_blocks: Dict[PairKey, torch.Tensor]  # (E_ab, d_A, d_B)
    pair_edges: Dict[PairKey, torch.Tensor]  # (2, E_ab)
    lookup: Dict[Tuple[int, int], Tuple[PairKey, int]]
    mapper: BlockIrrepMapper
    basis: str = "openmx"  # "openmx" | "e3nn"

    # --------------- convenience constructors ------------------------------ #
    @classmethod
    def empty(cls, atoms, mapper):
        return cls(tuple(atoms), {}, {}, {}, mapper)

    # --------------- dict-like access -------------------------------------- #
    def __getitem__(self, item):
        # item = (i, j) global indices
        if isinstance(item, tuple) and len(item) == 2:
            key, k = self.lookup[item]
            return self.pair_blocks[key][k]
        # item = "A-B"
        if isinstance(item, str):
            return self.pair_blocks[item]
        raise KeyError("use (i,j) or 'A-B'")

    def edges(self, key: PairKey) -> torch.Tensor:
        return self.pair_edges[key]

    def keys(self):
        return self.pair_blocks.keys()

    # --------------- device handling --------------------------------------- #
    def to(self, device):
        new_blocks = {k: v.to(device) for k, v in self.pair_blocks.items()}
        new_edges = {k: v.to(device) for k, v in self.pair_edges.items()}
        return MatrixBlockData(
            self.atoms, new_blocks, new_edges, self.lookup, self.mapper, self.basis
        )

    # --------------- change-of-basis --------------------------------------- #
    def to_vectors(self) -> "IrrepsBlockData":
        pair_vec: Dict[PairKey, torch.Tensor] = {}
        for key, blk in self.pair_blocks.items():
            pair_vec[key] = self.mapper.blocks_to_vectors(key, blk)
        return IrrepsBlockData(
            self.atoms, pair_vec, self.pair_edges, self.lookup, self.mapper
        )

        # ------------------------------------------------------------------ transpose

    def transpose(self) -> "MatrixBlockData":
        """
        Return a **new** snapshot representing the transposed matrix
        (conjugate-transpose is identical here, blocks are real).

        * Blocks are individually transposed.
        * Pair-key orientation is flipped (``"A-B"`` → ``"B-A"``).
        * Edge indices are swapped (i,j) → (j,i).
        """
        new_blocks, new_edges = {}, {}
        for key, blk in self.pair_blocks.items():
            el_a, el_b = key.split("-")
            new_key = f"{el_b}-{el_a}"
            new_blocks[new_key] = blk.transpose(-1, -2).clone()
            # swap edge orientation
            edge = self.pair_edges[key].flip(0).clone()  # (2,E) with rows swapped
            new_edges[new_key] = edge

        # rebuild lookup
        new_lookup = {}
        for new_key, edges in new_edges.items():
            for idx, (i, j) in enumerate(edges.t().tolist()):
                new_lookup[(i, j)] = (new_key, idx)

        return MatrixBlockData(
            atoms=self.atoms,
            pair_blocks=new_blocks,
            pair_edges=new_edges,
            lookup=new_lookup,
            mapper=self.mapper,
        )

    # ------------------------------------------------------------------ edge reordering
    def reorder_edges(self, order_dict: Dict[str, torch.Tensor]) -> "MatrixBlockData":
        """
        Re-order edge *rows* for given keys. ``order_dict`` maps
        ``key -> permutation indices`` (1-D LongTensor of length ``E_key``).

        Keys **not** in ``order_dict`` keep their original order.
        """
        pair_blocks, pair_edges, lookup = {}, {}, {}
        for key, blk in self.pair_blocks.items():
            if key in order_dict:
                idx = order_dict[key]
                blk = blk[idx]
                edges = self.pair_edges[key][:, idx]
            else:
                edges = self.pair_edges[key]
            pair_blocks[key] = blk
            pair_edges[key] = edges
            for new_k, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, new_k)

        return MatrixBlockData(
            atoms=self.atoms,
            pair_blocks=pair_blocks,
            pair_edges=pair_edges,
            lookup=lookup,
            mapper=self.mapper,
        )

    # ------------------------------------------------------------------ canonical sort
    def standardize_edges(self) -> "MatrixBlockData":
        """
        Return a snapshot where *each* pair-key’s edges are sorted by global
        `(src, dst)` (lexicographic).  Useful for deterministic equality tests.
        """
        order_dict = {}
        n_atoms = len(self.atoms)
        for key, edges in self.pair_edges.items():
            score = edges[0] * n_atoms + edges[1]  # monotonic mapping
            order = torch.argsort(score)
            order_dict[key] = order
        return self.reorder_edges(order_dict)

    # ------------------------------------------------------------------ arithmetic
    # private helper ------------------------------------------------------------
    def _align_with(
        self, other: "MatrixBlockData"
    ) -> Tuple["MatrixBlockData", "MatrixBlockData"]:
        """Return *standardised* copies whose edge order is identical pair-wise."""
        if not isinstance(other, MatrixBlockData):
            raise TypeError("Operand must be MatrixBlockData")
        if self.atoms != other.atoms:
            raise ValueError("Atoms differ; cannot add/subtract snapshots")
        if self.basis != other.basis:
            raise ValueError("Basis differs (openmx vs e3nn)")
        if self.mapper.orbital_cfg.to_dict() != other.mapper.orbital_cfg.to_dict():
            raise ValueError("OrbitalIrrepConfig differs")

        a_std = self.standardize_edges()
        b_std = other.standardize_edges()

        if a_std.keys() != b_std.keys():
            raise ValueError("Snapshots contain different element-pair keys")
        for k in a_std.keys():
            if a_std.pair_blocks[k].shape != b_std.pair_blocks[k].shape:
                raise ValueError(f"Shape mismatch for key '{k}'")
        return a_std, b_std

    # -------------- public dunder ops -----------------------------------------
    def __add__(self, other):
        if other == 0:  # allow sum() with start=0
            return self
        a, b = self._align_with(other)
        new_blocks = {k: a.pair_blocks[k] + b.pair_blocks[k] for k in a.pair_blocks}
        return a._replace_pair_blocks(new_blocks, basis=a.basis)

    __radd__ = __add__  # commutative

    def __neg__(self):
        new_blocks = {k: -v for k, v in self.pair_blocks.items()}
        return self._replace_pair_blocks(new_blocks, basis=self.basis)

    def __sub__(self, other):
        if other == 0:
            return self
        a, b = self._align_with(other)
        new_blocks = {k: a.pair_blocks[k] - b.pair_blocks[k] for k in a.pair_blocks}
        return a._replace_pair_blocks(new_blocks, basis=a.basis)

    def __rsub__(self, other):
        # allow (0 - snapshot)
        if other == 0:
            return -self
        return NotImplemented

    # --------------------- dense helpers --------------------------------------
    # global dim offsets ---------------------------------------------------------
    def _atom_offsets(self) -> Tuple[torch.Tensor, int]:
        """Return 1-D tensor of start indices per atom and total dimension."""
        dims = torch.tensor(
            [self.mapper.block_dims(f"{el}-{el}")[0] for el in self.atoms],
            dtype=torch.long,
        )
        offsets = torch.cumsum(
            torch.cat([torch.tensor([0]), dims[:-1]]), dim=0
        )  # start indices
        total_dim = int(offsets[-1] + dims[-1])
        return offsets, total_dim

    # --- public API -------------------------------------------------------------
    def to_dense(self) -> torch.Tensor:
        """
        Assemble a full dense matrix of shape ``(Σ d_i, Σ d_i)`` where
        ``d_i`` is orbital dimension of atom *i*.
        """
        offsets, total = self._atom_offsets()
        device = next(iter(self.pair_blocks.values())).device
        dtype = next(iter(self.pair_blocks.values())).dtype
        dense = torch.zeros(total, total, device=device, dtype=dtype)

        for (i, j), (key, k) in self.lookup.items():
            d_i, d_j = self.mapper.block_dims(key)
            r0 = int(offsets[i])
            c0 = int(offsets[j])
            dense[r0 : r0 + d_i, c0 : c0 + d_j] = self.pair_blocks[key][k]
        return dense

    # ------------------ serialisation ------------------------------------
    def _to_payload(self) -> dict:
        """Plain python types + **CPU** tensors → ready for torch.save."""
        blocks_cpu = {k: v.detach().cpu() for k, v in self.pair_blocks.items()}
        edges_cpu = {k: v.detach().cpu() for k, v in self.pair_edges.items()}
        return {
            "atoms": list(self.atoms),
            "orbital_cfg": self.mapper.orbital_cfg.to_dict(),
            "diagonal": self.mapper.diagonal,
            "pair_blocks": blocks_cpu,
            "pair_edges": edges_cpu,
            "type": "block",
            "basis": self.basis,
        }

    def save(self, path: str | "os.PathLike[str]") -> None:
        import torch

        torch.save(self._to_payload(), path)

    # ------------------ alternate constructors ---------------------------
    @classmethod
    def load(cls, path, device="cpu") -> "MatrixBlockData":
        import torch
        from core.orbital_irrep_config import OrbitalIrrepConfig

        payload = torch.load(path, map_location="cpu")
        if payload.get("type") != "block":
            raise ValueError("file does not contain block snapshot")

        orb_cfg = OrbitalIrrepConfig.from_dict(payload["orbital_cfg"])
        mapper = BlockIrrepMapper(orb_cfg, diagonal=payload["diagonal"], device="cpu")

        pair_blocks = {k: v.to(device) for k, v in payload["pair_blocks"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup
        lookup = {}
        for key, edges in pair_edges.items():
            for idx, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, idx)

        return cls(
            tuple(payload["atoms"]),
            pair_blocks,
            pair_edges,
            lookup,
            mapper,
            payload.get("basis", "openmx"),
        ).to(device)

    # ------------- internal helper to clone with new blocks / basis ---------
    def _replace_pair_blocks(
        self, new_blocks: Dict[PairKey, torch.Tensor], *, basis: str
    ):
        """Return a shallow copy with *pair_blocks* replaced."""
        return MatrixBlockData(
            atoms=self.atoms,
            pair_blocks=new_blocks,
            pair_edges=self.pair_edges,
            lookup=self.lookup,
            mapper=self.mapper,
            basis=basis,
        )

    # ---------------- basis conversion wrappers ----------------------------
    def to_e3nn(self, converter: "OpenMXE3NNConverter") -> "MatrixBlockData":
        return converter.snapshot_to_e3nn(self)

    def to_openmx(self, converter: "OpenMXE3NNConverter") -> "MatrixBlockData":
        return converter.snapshot_to_openmx(self)

    # ------------------------ alternate constructor -----------------------------
    @classmethod
    def from_dense(
        cls,
        matrix: torch.Tensor,
        orbital_cfg,  # OrbitalIrrepConfig
        atoms: Tuple[str, ...] | list[str],
        *,
        diagonal: bool = False,
        sparsity_threshold: float = 0.0,
    ) -> "MatrixBlockData":
        """
        Build a block snapshot from a fully dense matrix *in global atom order*.
        Primarily for tests / debugging.
        """
        from core.block_irrep_mapper import BlockIrrepMapper  # loc import

        atoms = tuple(atoms)
        mapper = BlockIrrepMapper(orbital_cfg, diagonal=diagonal, device=matrix.device)

        offsets = [0]
        for el in atoms[:-1]:
            d = mapper.block_dims(f"{el}-{el}")[0]
            offsets.append(offsets[-1] + d)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=matrix.device)

        pair_blocks, pair_edges, lookup = {}, {}, {}
        for i, el_i in enumerate(atoms):
            d_i = mapper.block_dims(f"{el_i}-{el_i}")[0]
            r0 = int(offsets_t[i])
            for j, el_j in enumerate(atoms):
                d_j = mapper.block_dims(f"{el_j}-{el_j}")[0]
                c0 = int(offsets_t[j])
                blk = matrix[r0 : r0 + d_i, c0 : c0 + d_j].clone()
                key = f"{el_i}-{el_j}"
                if key not in pair_blocks:
                    pair_blocks[key], pair_edges[key] = [], []
                local_idx = len(pair_blocks[key])
                pair_blocks[key].append(blk)
                pair_edges[key].append([i, j])
                lookup[(i, j)] = (key, local_idx)
        # stack
        pair_blocks = {k: torch.stack(v) for k, v in pair_blocks.items()}
        pair_edges = {
            k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
        }

        snapshot = cls(atoms, pair_blocks, pair_edges, lookup, mapper)
        if sparsity_threshold is not None:
            snapshot = snapshot.sparsify(sparsity_threshold)
        return snapshot

        # ---------------------------------------------------------------- sparsify

    def sparsify(self, threshold: float) -> "MatrixBlockData":
        """
        Return a **new** snapshot in which only blocks whose root-mean-square
        (RMS) magnitude is **strictly greater** than ``threshold`` are kept.

        A block's RMS is computed as
        ``rms = sqrt( mean( block**2 ) )`` across its last two dimensions
        (orbital rows & columns).

        Parameters
        ----------
        threshold
            Cut-off applied *per block*.  A Python float is accepted; it is
            internally cast to the block's dtype and device.

        Notes
        -----
        • Pruning is performed **per block** (first tensor dimension).
          Keys for which *all* blocks are removed disappear entirely.
        """
        if not self.pair_blocks:
            return self  # nothing to do

        sample_blk = next(iter(self.pair_blocks.values()))
        thr = torch.as_tensor(
            threshold, dtype=sample_blk.dtype, device=sample_blk.device
        )

        new_blocks: Dict[PairKey, torch.Tensor] = {}
        new_edges: Dict[PairKey, torch.Tensor] = {}
        new_lookup: Dict[Tuple[int, int], Tuple[PairKey, int]] = {}

        for key, blk in self.pair_blocks.items():
            # (E, d_i, d_j) → (E,)
            rms = blk.pow(2).mean(dim=(-2, -1)).sqrt()
            keep = rms > thr

            if torch.any(keep):
                blk_kept = blk[keep]
                edges_kept = self.pair_edges[key][:, keep]

                new_blocks[key] = blk_kept
                new_edges[key] = edges_kept

                # rebuild lookup for the surviving edges of this key
                for local_idx, (i, j) in enumerate(edges_kept.t().tolist()):
                    new_lookup[(i, j)] = (key, local_idx)

        # assemble the sparsified snapshot
        return MatrixBlockData(
            atoms=self.atoms,
            pair_blocks=new_blocks,
            pair_edges=new_edges,
            lookup=new_lookup,
            mapper=self.mapper,
            basis=self.basis,
        )


# --------------------------------------------------------------------------- #
@dataclass
class IrrepsBlockData:
    atoms: Tuple[str, ...]
    pair_vectors: Dict[PairKey, torch.Tensor]  # (E_ab, n_vec_AB)
    pair_edges: Dict[PairKey, torch.Tensor]
    lookup: Dict[Tuple[int, int], Tuple[PairKey, int]]
    mapper: BlockIrrepMapper

    # -------- device -------- #
    def to(self, device):
        vecs = {k: v.to(device) for k, v in self.pair_vectors.items()}
        edges = {k: v.to(device) for k, v in self.pair_edges.items()}
        return IrrepsBlockData(
            self.atoms, vecs, edges, self.lookup, self.mapper
        )  # TODO: mapper also changes device

    # -------- inverse change-of-basis ----- #
    def to_blocks(self) -> MatrixBlockData:
        pair_blk: Dict[PairKey, torch.Tensor] = {}
        for key, vec in self.pair_vectors.items():
            pair_blk[key] = self.mapper.vectors_to_blocks(key, vec)
        return MatrixBlockData(
            self.atoms, pair_blk, self.pair_edges, self.lookup, self.mapper
        )

    # -------- indexing paralleling MatrixBlockData -------- #
    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            key, k = self.lookup[item]
            return self.pair_vectors[key][k]
        if isinstance(item, str):
            return self.pair_vectors[item]
        raise KeyError

    # ------------------------------------------------------------------ serialisation
    def _to_payload(self) -> dict:
        """
        Convert to a CPU-resident, torch-savable python dict.
        """
        vec_cpu = {k: v.detach().cpu() for k, v in self.pair_vectors.items()}
        edges_cpu = {k: v.detach().cpu() for k, v in self.pair_edges.items()}
        return {
            "atoms": list(self.atoms),
            "orbital_cfg": self.mapper.orbital_cfg.to_dict(),
            "diagonal": self.mapper.diagonal,
            "pair_vectors": vec_cpu,
            "pair_edges": edges_cpu,
            "type": "irrep",
        }

    def save(self, path: str | "os.PathLike[str]") -> None:
        import torch

        torch.save(self._to_payload(), path)

    # --------------------- alternate constructor ----------------------------- #
    @classmethod
    def load(cls, path, device="cpu") -> "IrrepsBlockData":
        import torch
        from core.orbital_irrep_config import OrbitalIrrepConfig

        payload = torch.load(path, map_location="cpu")
        if payload.get("type") != "irrep":
            raise ValueError("file does not contain irrep snapshot")

        orb_cfg = OrbitalIrrepConfig.from_dict(payload["orbital_cfg"])
        mapper = BlockIrrepMapper(orb_cfg, diagonal=payload["diagonal"], device="cpu")

        pair_vec = {k: v.to(device) for k, v in payload["pair_vectors"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup
        lookup = {}
        for key, edges in pair_edges.items():
            for idx, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, idx)

        return cls(tuple(payload["atoms"]), pair_vec, pair_edges, lookup, mapper).to(
            device
        )
