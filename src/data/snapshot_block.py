"""
snapshot_block.py
=================

In-memory containers for a *single* snapshot:

* **SnapshotBlockData** – raw sparse blocks (H, D, S, …) grouped by element pair.
* **SnapshotIrrepsData** – same data after change-of-basis to irrep vectors.

Both dataclasses keep:
    * `atoms`          : list[str]  (global order)
    * `pair_edges`     : dict[key] → (2,E_ab)  long tensor with global indices
    * `lookup`         : dict[(i,j)] → (key, local_idx)  for O(1) access
    * reference to the shared `BlockIrrepMapper`

Conversion between the two is a one-liner via `.to_vectors()` / `.to_blocks()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from core.block_irrep_mapper import BlockIrrepMapper

PairKey = str  # canonical "A-B"


# --------------------------------------------------------------------------- #
@dataclass
class SnapshotBlockData:
    atoms: Tuple[str, ...]
    pair_blocks: Dict[PairKey, torch.Tensor]  # (E_ab, d_A, d_B)
    pair_edges: Dict[PairKey, torch.Tensor]  # (2, E_ab)
    lookup: Dict[Tuple[int, int], Tuple[PairKey, int]]
    mapper: BlockIrrepMapper

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
        return SnapshotBlockData(
            self.atoms, new_blocks, new_edges, self.lookup, self.mapper
        )  # TODO: mapper also changes device

    # --------------- change-of-basis --------------------------------------- #
    def to_vectors(self) -> "SnapshotIrrepsData":
        pair_vec: Dict[PairKey, torch.Tensor] = {}
        for key, blk in self.pair_blocks.items():
            pair_vec[key] = self.mapper.blocks_to_vectors(key, blk)
        return SnapshotIrrepsData(
            self.atoms, pair_vec, self.pair_edges, self.lookup, self.mapper
        )

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

    # ------------------------ alternate constructor -----------------------------
    @classmethod
    def from_dense(
        cls,
        matrix: torch.Tensor,
        orbital_cfg,  # OrbitalIrrepConfig
        atoms: Tuple[str, ...] | list[str],
        *,
        diagonal: bool = False,
    ) -> "SnapshotBlockData":
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

        return cls(atoms, pair_blocks, pair_edges, lookup, mapper)


# --------------------------------------------------------------------------- #
@dataclass
class SnapshotIrrepsData:
    atoms: Tuple[str, ...]
    pair_vectors: Dict[PairKey, torch.Tensor]  # (E_ab, n_vec_AB)
    pair_edges: Dict[PairKey, torch.Tensor]
    lookup: Dict[Tuple[int, int], Tuple[PairKey, int]]
    mapper: BlockIrrepMapper

    # -------- device -------- #
    def to(self, device):
        vecs = {k: v.to(device) for k, v in self.pair_vectors.items()}
        edges = {k: v.to(device) for k, v in self.pair_edges.items()}
        return SnapshotIrrepsData(
            self.atoms, vecs, edges, self.lookup, self.mapper
        )  # TODO: mapper also changes device

    # -------- inverse change-of-basis ----- #
    def to_blocks(self) -> SnapshotBlockData:
        pair_blk: Dict[PairKey, torch.Tensor] = {}
        for key, vec in self.pair_vectors.items():
            pair_blk[key] = self.mapper.vectors_to_blocks(key, vec)
        return SnapshotBlockData(
            self.atoms, pair_blk, self.pair_edges, self.lookup, self.mapper
        )

    # -------- indexing paralleling SnapshotBlockData -------- #
    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            key, k = self.lookup[item]
            return self.pair_vectors[key][k]
        if isinstance(item, str):
            return self.pair_vectors[item]
        raise KeyError
