"""
snapshot_block.py
=================

In-memory containers for a *single* snapshot:

* **BlockMatrix** - raw sparse blocks (H, D, S, …) grouped by element pair.
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
from typing import Sequence

if TYPE_CHECKING:
    from core.basis_converter import OpenMXE3NNConverter


import torch


from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig

PairKey = str  # canonical "A-B"


# --------------------------------------------------------------------------- #
@dataclass
class BlockMatrix:
    atoms: Tuple[str, ...]
    atom_counts: Dict[str, int]
    pair_blocks: Dict[PairKey, torch.Tensor]  # (E_ab, d_A, d_B)
    pair_edges: Dict[PairKey, torch.Tensor]  # (5, E_ab)
    lookup: Dict[Tuple[int, int], Tuple[PairKey, int]]
    orbital_cfg: OrbitalIrrepConfig
    basis: str  # "openmx" | "e3nn"

    # --------------- convenience constructors ------------------------------ #
    @classmethod
    def empty(cls, atoms, orbital_cfg, basis="e3nn"):
        from collections import Counter

        return cls(tuple(atoms), Counter(atoms), {}, {}, {}, orbital_cfg, basis)

    # --------------- dict-like access -------------------------------------- #
    def __getitem__(self, item):
        # item = (i, j) global indices, sum periodic images
        if isinstance(item, tuple) and len(item) == 2:
            print("Warning: BlockMatrix.__getitem__((i,j)) sums over periodic images.")
            ret = None
            for key in self.lookup:
                # key is (sx, sy, sz, i, j)
                if key[3] == item[0] and key[4] == item[1]:
                    k_key, k_idx = self.lookup[key]
                    block = self.pair_blocks[k_key][k_idx]
                    if ret is None:
                        ret = block.clone()
                    else:
                        ret += block
            if ret is None:
                raise KeyError(f"No blocks found for indices {item}")
            return ret
        # item = (sx, sy, sz, i, j) global indices
        if isinstance(item, tuple) and len(item) == 5:
            key, k = self.lookup[item]
            return self.pair_blocks[key][k]
        # item = "A-B"
        if isinstance(item, str):
            return self.pair_blocks[item]
        raise KeyError("use (i,j), (sx,sy,sz,i,j) or 'A-B'")

    def edges(self, key: PairKey) -> torch.Tensor:
        return self.pair_edges[key]

    def keys(self):
        return self.pair_blocks.keys()

    # ─────────────────────────────────────────────────────────────────────────
    #   helpers to access diagonal / off-diagonal blocks
    # ─────────────────────────────────────────────────────────────────────────
    def diag(self) -> Dict[PairKey, torch.Tensor]:
        """
        Return a **dict** ``key → tensor`` that contains *only the blocks whose
        global source/target atom are identical* (self-edges).

        Off-diagonal blocks are omitted; keys that would become empty disappear.
        """
        diag_dict: Dict[PairKey, torch.Tensor] = {}
        for key, blk in self.pair_blocks.items():
            el_a, el_b = key.split("-")
            if el_a != el_b:
                continue
            n_atoms = self.atom_counts[el_a]
            diag_dict[key] = blk[:n_atoms]

        return diag_dict

    def offdiag(self) -> Dict[PairKey, torch.Tensor]:
        """
        Same as :meth:`diag` but returns the **off-diagonal** blocks
        ( *i* ≠ *j* ).
        """
        off_dict: Dict[PairKey, torch.Tensor] = {}
        for key, blk in self.pair_blocks.items():
            el_a, el_b = key.split("-")
            if el_a == el_b:
                n_atoms = self.atom_counts[el_a]
                off_dict[key] = blk[n_atoms:]
            else:
                off_dict[key] = blk
        return off_dict

    # --------------- device handling --------------------------------------- #
    def to(self, device):
        new_blocks = {k: v.to(device) for k, v in self.pair_blocks.items()}
        new_edges = {k: v.to(device) for k, v in self.pair_edges.items()}
        return BlockMatrix(
            self.atoms,
            self.atom_counts,
            new_blocks,
            new_edges,
            self.lookup,
            self.orbital_cfg,
            self.basis,
        )

    # --------------- change-of-basis --------------------------------------- #
    def to_vectors(self, mapper: BlockIrrepMapper) -> IrrepsBlockData:
        """Convert blocks to irreducible representation vectors using a mapper."""
        pair_vec: Dict[PairKey, torch.Tensor] = {}
        for key, blk in self.pair_blocks.items():
            pair_vec[key] = mapper.blocks_to_vectors(key, blk)
        return IrrepsBlockData(
            self.atoms,
            self.atom_counts,
            pair_vec,
            self.pair_edges,
            self.lookup,
            self.orbital_cfg,
            self.basis,
        )

        # ------------------------------------------------------------------ transpose

    def transpose(self) -> "BlockMatrix":
        """
        Return a **new** snapshot representing the transposed matrix.
        Edge order is preserved from the original matrix where possible.
        """
        # 1. Perform a simple transpose, flipping keys and edges.
        transposed_blocks = {}
        transposed_edges = {}
        for key, blk in self.pair_blocks.items():
            el_a, el_b = key.split("-")
            new_key = f"{el_b}-{el_a}"
            transposed_blocks[new_key] = blk.transpose(-1, -2).clone()
            edges = self.pair_edges[key]
            transposed_edges[new_key] = torch.stack(
                [-edges[0], -edges[1], -edges[2], edges[4], edges[3]]
            )

        final_blocks = {}
        final_edges = {}

        # 2. For each key in the transposed matrix, restore original order if the key
        #    existed in the original matrix. Otherwise, create a canonical order.
        for key, t_edges in transposed_edges.items():
            if key in self.pair_edges:
                # This key existed in the original matrix. We should match its edge order.
                original_edges = self.pair_edges[key]

                # Build a map from a transposed edge to its current index.
                map_edge_to_idx = {
                    tuple(t_edges[:, i].tolist()): i for i in range(t_edges.shape[1])
                }

                # Create permutation by looking up original edges in the map.
                try:
                    perm = torch.tensor(
                        [
                            map_edge_to_idx[tuple(original_edges[:, i].tolist())]
                            for i in range(original_edges.shape[1])
                        ]
                    )
                    final_edges[key] = t_edges[:, perm]
                    final_blocks[key] = transposed_blocks[key][perm]
                except KeyError:
                    raise Exception(
                        f"Key '{key}' not found in the transposed matrix; matrix not symmetric."
                    )
            else:
                raise Exception(
                    f"Key '{key}' not found in the original matrix; matrix not symmetric."
                )

        # Rebuild lookup
        new_lookup = {}
        for key, edges in final_edges.items():
            for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                new_lookup[(sx, sy, sz, i, j)] = (key, idx)

        return BlockMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_blocks=final_blocks,
            pair_edges=final_edges,
            lookup=new_lookup,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
        )

    def __mul__(self, scalar: float) -> "BlockMatrix":
        """
        Scalar multiplication of all blocks by a float or int.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        new_blocks = {k: v * scalar for k, v in self.pair_blocks.items()}
        return self._replace_pair_blocks(new_blocks, basis=self.basis)

    # ------------------------------------------------------------------ edge reordering
    def reorder_edges(self, order_dict: Dict[str, torch.Tensor]) -> "BlockMatrix":
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
            for new_k, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, new_k)

        return BlockMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_blocks=pair_blocks,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
        )

    # ------------------------------------------------------------------ arithmetic
    # private helper ------------------------------------------------------------
    def _align_with(self, other: "BlockMatrix") -> Tuple["BlockMatrix", "BlockMatrix"]:
        """
        Make sure the matrices have the same atoms, basis, orbital_cfg and edge order.
        """
        if not isinstance(other, BlockMatrix):
            raise TypeError("Operand must be BlockMatrix")
        if self.atoms != other.atoms:
            raise ValueError("Atoms differ; cannot add/subtract snapshots")
        if self.basis != other.basis:
            raise ValueError("Basis differs (openmx vs e3nn)")
        if self.orbital_cfg.to_dict() != other.orbital_cfg.to_dict():
            raise ValueError("OrbitalIrrepConfig differs")

        if self.keys() != other.keys():
            raise ValueError("Snapshots contain different element-pair keys")

        reorder_dict = {}
        for k in self.keys():
            if not torch.equal(self.pair_edges[k], other.pair_edges[k]):
                # If edge order differs, reorder self to match other.
                map_edge_to_idx = {
                    tuple(self.pair_edges[k][:, i].tolist()): i
                    for i in range(self.pair_edges[k].shape[1])
                }
                try:
                    perm = torch.tensor(
                        [
                            map_edge_to_idx[tuple(other.pair_edges[k][:, i].tolist())]
                            for i in range(other.pair_edges[k].shape[1])
                        ]
                    )
                    reorder_dict[k] = perm
                except KeyError:
                    raise ValueError(f"Edge sets for key '{k}' differ between matrices")

            if self.pair_blocks[k].shape != other.pair_blocks[k].shape:
                raise ValueError(f"Shape mismatch for key '{k}'")

        if reorder_dict:
            reordered_self = self.reorder_edges(reorder_dict)
            return reordered_self, other
        else:
            return self, other

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
            [self.orbital_cfg.block_dims(f"{el}-{el}")[0] for el in self.atoms],
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
        Summing all periodic images into the central cell.
        """
        offsets, total = self._atom_offsets()
        device = next(iter(self.pair_blocks.values())).device
        dtype = next(iter(self.pair_blocks.values())).dtype
        dense = torch.zeros(total, total, device=device, dtype=dtype)

        for (_sx, _sy, _sz, i, j), (key, k) in self.lookup.items():
            d_i, d_j = self.orbital_cfg.block_dims(key)
            r0 = int(offsets[i])
            c0 = int(offsets[j])
            dense[r0 : r0 + d_i, c0 : c0 + d_j] += self.pair_blocks[key][k]
        return dense

    def to_dense_k(self, lattice, wave,
                   wave_frac=True) -> torch.Tensor:
        """
        Assemble a full dense matrix of shape ``(Σ d_i, Σ d_i)`` where
        ``d_i`` is orbital dimension of atom *i*.
        Summing all periodic images into the central cell. Additionally,
        computes the phase factors for the fourier transform to
        reciprocal space and multiplies each (d_i,d_j) by the
        appropriate factor.
        Inputs:
            lattice: Ndim x Ndim matrix, torch.tensor
                the lattice vectors of the Hamiltonian stored row-wise.
                [[x],[y],[z]]
            k: Ndim vector, torch.tensor
                The reciprocal lattice vector that determines the
                reciprocal Hamiltonian to compute.
                [1/x,1/y,1/z]
        """

        offsets, total = self._atom_offsets()
        device = next(iter(self.pair_blocks.values())).device
        dtype = next(iter(self.pair_blocks.values())).dtype

        # decide on the complex type to store the k hamiltonian
        if dtype == torch.float32:
            ctype = torch.complex64
        elif dtype == torch.float64:
            ctype = torch.complex128
        else:
            ctype = torch.complex

        #initalize dense and wave on device
        dense = torch.zeros(total, total, device=device, dtype=ctype)
        wave = wave.to(device=device, dtype=dtype)

        # transform the fractional k_space to cartesian k-space
        if wave_frac:

            # compute the reciprocal lattice vectors
            inv_lat = 2*torch.pi*torch.linalg.inv(lattice.T).T

            # bring the wave vector into the inverse cartesian coordinates
            wave = torch.matmul(inv_lat, wave)

        for (_sx, _sy, _sz, i, j), (key, k) in self.lookup.items():
            # construct the translation vector
            trans = torch.matmul(lattice.T, torch.tensor([_sx,_sy,_sz],
                                         device=device, dtype=dtype))

            # compute the fourier component
            fij = torch.exp(1j*torch.matmul(wave, trans))

            # perform sum
            d_i, d_j = self.orbital_cfg.block_dims(key)
            r0 = int(offsets[i])
            c0 = int(offsets[j])
            dense[r0 : r0 + d_i, c0 : c0 + d_j] += fij*self.pair_blocks[key][k]
        return dense

    # ------------------ serialisation ------------------------------------
    def _to_payload(self) -> dict:
        """Plain python types + **CPU** tensors → ready for torch.save."""
        blocks_cpu = {k: v.detach().cpu() for k, v in self.pair_blocks.items()}
        edges_cpu = {k: v.detach().cpu() for k, v in self.pair_edges.items()}
        return {
            "atoms": list(self.atoms),
            "atom_counts": self.atom_counts,
            "orbital_cfg": self.orbital_cfg.to_dict(),
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
    def load(cls, path, device="cpu") -> "BlockMatrix":
        import torch
        from core.orbital_irrep_config import OrbitalIrrepConfig

        payload = torch.load(path, map_location="cpu")
        if payload.get("type") != "block":
            raise ValueError("file does not contain block snapshot")

        orb_cfg = OrbitalIrrepConfig.from_dict(payload["orbital_cfg"])

        pair_blocks = {k: v.to(device) for k, v in payload["pair_blocks"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup
        lookup = {}
        for key, edges in pair_edges.items():
            for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, idx)

        atoms = tuple(payload["atoms"])
        from collections import Counter

        atom_counts = payload.get("atom_counts", Counter(atoms))

        return cls(
            atoms,
            atom_counts,
            pair_blocks,
            pair_edges,
            lookup,
            orb_cfg,
            payload.get("basis", "openmx"),
        ).to(device)

    # ------------- internal helper to clone with new blocks / basis ---------
    def _replace_pair_blocks(
        self, new_blocks: Dict[PairKey, torch.Tensor], *, basis: str
    ):
        """Return a shallow copy with *pair_blocks* replaced."""
        return BlockMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_blocks=new_blocks,
            pair_edges=self.pair_edges,
            lookup=self.lookup,
            orbital_cfg=self.orbital_cfg,
            basis=basis,
        )

    def _apply_edge_mask(
        self,
        mask_dict: Dict[PairKey, torch.Tensor | Sequence[bool]],
        *,
        drop_empty: bool = True,
    ) -> "BlockMatrix":
        """
        Internal utility - return a **new** BlockMatrix where, for every
        ``key`` contained in ``mask_dict``, only the edges with a *True*
        entry in the 1-D boolean mask are kept.

        Keys **not** present in the dict are copied unchanged.
        """
        pair_blocks, pair_edges, lookup = {}, {}, {}

        for key, blk in self.pair_blocks.items():
            if key in mask_dict:
                mask = torch.as_tensor(
                    mask_dict[key], dtype=torch.bool, device=blk.device
                )
                if mask.ndim != 1 or mask.numel() != blk.shape[0]:
                    raise ValueError(f"Mask for '{key}' has wrong shape")
                blk_kept = blk[mask]
                edges_kept = self.pair_edges[key][:, mask]
            else:  # untouched
                blk_kept = blk
                edges_kept = self.pair_edges[key]

            if blk_kept.shape[0] == 0 and drop_empty:
                continue  # drop key entirely

            pair_blocks[key] = blk_kept
            pair_edges[key] = edges_kept
            for idx, (sx, sy, sz, i, j) in enumerate(edges_kept.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, idx)

        return BlockMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_blocks=pair_blocks,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
        )

    # ---------------- basis conversion wrappers ----------------------------
    def to_e3nn(self, converter: "OpenMXE3NNConverter") -> "BlockMatrix":
        matrix = converter.matrix_to_e3nn(self)
        matrix.basis = "e3nn"
        return matrix

    def to_openmx(self, converter: "OpenMXE3NNConverter") -> "BlockMatrix":
        matrix = converter.matrix_to_openmx(self)
        matrix.basis = "openmx"
        return matrix

    def change_basis(self, d_dict: Dict[str, torch.Tensor]) -> "BlockMatrix":
        """
        Change the basis of the snapshot using a dictionary of transformation matrices.
        Each key in `d_dict` corresponds to an element symbol, and the value is a
        transformation matrix that will be applied to the blocks associated with that element.
        The transformation is applied as follows:
        For a block corresponding to the pair (A-B):
        .. math::
            \\text{new\_block}_{AB} = d_{A} \\cdot \\text{block}_{AB} \\cdot d_{B}^T
        where :math:`d_{A}` and :math:`d_{B}` are the transformation matrices for elements A and B,
        respectively.
        Parameters
        ----------
        d_dict : Dict[str, torch.Tensor]
            A dictionary mapping element symbols to transformation matrices.
            Each matrix should have the shape (d_A, d_A) for element A.
        Returns
        -------
        BlockMatrix
            A new BlockMatrix instance with the transformed blocks.
        Notes
        -----
        This method assumes that the transformation matrices in `d_dict` are square matrices
        with dimensions matching the orbital dimensions of the respective elements.
        """
        new_pair_blocks = {}
        for key, blk in self.pair_blocks.items():
            el_i, el_j = key.split("-")

            new_pair_blocks[key] = d_dict[el_i] @ blk @ d_dict[el_j].T
        # Return a new BlockMatrix with the updated blocks and unchanged edges
        return BlockMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_blocks=new_pair_blocks,
            pair_edges=self.pair_edges,
            lookup=self.lookup,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
        )

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
        basis: str,
    ) -> "BlockMatrix":
        """
        Build a block snapshot from a fully dense matrix *in global atom order*.
        Primarily for tests / debugging.
        """
        atoms = tuple(atoms)
        from collections import Counter

        atom_counts = Counter(atoms)

        offsets = [0]
        for el in atoms[:-1]:
            d = orbital_cfg.block_dims(f"{el}-{el}")[0]
            offsets.append(offsets[-1] + d)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=matrix.device)

        pair_blocks, pair_edges, lookup = {}, {}, {}
        for i, el_i in enumerate(atoms):
            d_i = orbital_cfg.block_dims(f"{el_i}-{el_i}")[0]
            r0 = int(offsets_t[i])
            for j, el_j in enumerate(atoms):
                d_j = orbital_cfg.block_dims(f"{el_j}-{el_j}")[0]
                c0 = int(offsets_t[j])
                blk = matrix[r0 : r0 + d_i, c0 : c0 + d_j].clone()
                key = f"{el_i}-{el_j}"
                if key not in pair_blocks:
                    pair_blocks[key], pair_edges[key] = [], []
                local_idx = len(pair_blocks[key])
                pair_blocks[key].append(blk)
                pair_edges[key].append([0, 0, 0, i, j])
                lookup[(0, 0, 0, i, j)] = (key, local_idx)
        print("Warning: from_dense currently assumes no periodic images!")
        # stack
        pair_blocks = {k: torch.stack(v) for k, v in pair_blocks.items()}
        pair_edges = {
            k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
        }

        snapshot = cls(
            atoms, atom_counts, pair_blocks, pair_edges, lookup, orbital_cfg, basis
        )
        if sparsity_threshold is not None:
            snapshot = snapshot.sparsify(sparsity_threshold)
        return snapshot

        # ---------------------------------------------------------------- sparsify

    def sparsify(self, threshold: float) -> "BlockMatrix":
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
                for local_idx, (sx, sy, sz, i, j) in enumerate(edges_kept.t().tolist()):
                    new_lookup[(sx, sy, sz, i, j)] = (key, local_idx)

        # assemble the sparsified snapshot
        return BlockMatrix(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_blocks=new_blocks,
            pair_edges=new_edges,
            lookup=new_lookup,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
        )

    # ----------------------------------------------------------------- reload from payload
    @classmethod
    def from_payload(
        cls, payload: dict, device: str | torch.device = "cpu"
    ) -> "BlockMatrix":
        """
        Build :class:`BlockMatrix` from a dict previously produced by
        :meth:`_to_payload`.  Used internally by Snapshot.load().
        """
        from core.orbital_irrep_config import OrbitalIrrepConfig  # local import

        orb_cfg = OrbitalIrrepConfig.from_dict(payload["orbital_cfg"])

        pair_blocks = {k: v.to(device) for k, v in payload["pair_blocks"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup table
        lookup: Dict[Tuple[int, int], Tuple[str, int]] = {}
        for key, edges in pair_edges.items():
            for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, idx)

        atoms = tuple(payload["atoms"])
        from collections import Counter

        atom_counts = payload.get("atom_counts", Counter(atoms))

        return cls(
            atoms=atoms,
            atom_counts=atom_counts,
            pair_blocks=pair_blocks,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=orb_cfg,
            basis=payload.get("basis", "openmx"),
        )

    # ════════════════════════════════════════════════════════════════════════════
    #                                ROTATION
    # ═══════════════════════════════════════════════════════════════════════════
    def rotate(self, R: torch.Tensor) -> "BlockMatrix":
        """
        Return a **new** :class:`BlockMatrix` whose *orbital reference frame*
        has been rotated by the (3 x 3) matrix **R** (active rotation).

        Notes
        -----
        * The operation is block-wise:

              M'_(A,B)  =  U_A · M_(A,B) · U_Bᵀ

          where ``U_A = irreps_A.D_from_matrix(R)`` is block-diagonal with one
          Wigner-D copy per orbital.
        """

        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 rotation matrix")

        device = next(iter(self.pair_blocks.values())).device
        R = R.to(device=device, dtype=torch.float32)

        # cache one U per element
        U_cache: Dict[str, torch.Tensor] = {}
        for el in self.orbital_cfg.elements():
            irr = self.orbital_cfg.element_to_irreps[el]
            U_cache[el] = irr.D_from_matrix(R)  # (dim_el, dim_el)

        # rotate every block ------------------------------------------------
        new_blocks: Dict[PairKey, torch.Tensor] = {}
        for key, blk in self.pair_blocks.items():
            el_i, el_j = key.split("-")
            U_i = U_cache[el_i]
            U_j = U_cache[el_j]
            # tensor contraction:  (E,d_i,d_j)
            blk_rot = U_i @ blk @ U_j.T
            new_blocks[key] = blk_rot

        # edge indices / lookup are unchanged
        return self._replace_pair_blocks(new_blocks, basis=self.basis)


# --------------------------------------------------------------------------- #
@dataclass
class IrrepsBlockData:
    atoms: Tuple[str, ...]
    atom_counts: Dict[str, int]
    pair_vectors: Dict[PairKey, torch.Tensor]  # (E_ab, n_vec_AB)
    pair_edges: Dict[PairKey, torch.Tensor]
    lookup: Dict[Tuple[int, int], Tuple[PairKey, int]]
    orbital_cfg: OrbitalIrrepConfig
    basis: str = "e3nn"  # always "e3nn"

    # -------- device -------- #
    def to(self, device):
        vecs = {k: v.to(device) for k, v in self.pair_vectors.items()}
        edges = {k: v.to(device) for k, v in self.pair_edges.items()}
        return IrrepsBlockData(
            self.atoms,
            self.atom_counts,
            vecs,
            edges,
            self.lookup,
            self.orbital_cfg,
            self.basis,
        )

    # -------- inverse change-of-basis ----- #
    def to_blocks(self, mapper: BlockIrrepMapper) -> BlockMatrix:
        pair_blk: Dict[PairKey, torch.Tensor] = {}
        for key, vec in self.pair_vectors.items():
            pair_blk[key] = mapper.vectors_to_blocks(key, vec)
        return BlockMatrix(
            self.atoms,
            self.atom_counts,
            pair_blk,
            self.pair_edges,
            self.lookup,
            self.orbital_cfg,
            self.basis,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #   diag / offdiag access for vector form
    # ─────────────────────────────────────────────────────────────────────────
    def diag(self) -> Dict[PairKey, torch.Tensor]:
        diag_dict: Dict[PairKey, torch.Tensor] = {}
        for key, vec in self.pair_vectors.items():
            el_a, el_b = key.split("-")
            if el_a != el_b:
                continue
            n_atoms = self.atom_counts[el_a]
            diag_dict[key] = vec[:n_atoms]
        return diag_dict

    def offdiag(self) -> Dict[PairKey, torch.Tensor]:
        off_dict: Dict[PairKey, torch.Tensor] = {}
        for key, vec in self.pair_vectors.items():
            el_a, el_b = key.split("-")
            if el_a == el_b:
                n_atoms = self.atom_counts[el_a]
                off_dict[key] = vec[n_atoms:]
            else:
                off_dict[key] = vec
        return off_dict

    # -------- indexing paralleling BlockMatrix -------- #
    def __getitem__(self, item):
        # item = (i, j) global indices, sum periodic images
        if isinstance(item, tuple) and len(item) == 2:
            print(
                "Warning: IrrepsBlockData.__getitem__((i,j)) sums over periodic images."
            )
            item_i, item_j = item
            ret = None
            for key in self.lookup:
                sx, sy, sz, i, j = key
                if i == item_i and j == item_j:
                    k_key, k_idx = self.lookup[key]
                    vec = self.pair_vectors[k_key][k_idx]
                    if ret is None:
                        ret = vec.clone()
                    else:
                        ret += vec
            if ret is None:
                raise KeyError(f"No vectors found for indices {item}")
            return ret
        # item = (sx, sy, sz, i, j) global indices
        if isinstance(item, tuple) and len(item) == 5:
            key, k = self.lookup[item]
            return self.pair_vectors[key][k]
        # item = "A-B"
        if isinstance(item, str):
            return self.pair_vectors[item]
        raise KeyError("use (i,j), (sx,sy,sz,i,j) or 'A-B'")

    # ════════════════════════════════════════════════════════════════════════════
    #                                ROTATION
    # ═══════════════════════════════════════════════════════════════════════════
    def rotate(self, R: torch.Tensor, mapper: BlockIrrepMapper) -> "IrrepsBlockData":
        """
        Return a **new** :class:`IrrepsBlockData` whose *orbital reference frame*
        has been rotated by the (3 x 3) matrix **R** (active rotation).

        Parameters
        ----------
        R
            The (3 x 3) rotation matrix.
        mapper
            The BlockIrrepMapper that provides the pair irreps for computing
            Wigner-D matrices.

        Notes
        -----
        * For irrep vectors representing the tensor product of orbitals from
          atoms A and B, we apply the Wigner-D matrix for the output irreps:

              v'_(A,B)  =  D_out(R) · v_(A,B)

          where ``D_out = pair_irreps.D_from_matrix(R)`` acts on the tensor
          product irreps.
        """
        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 rotation matrix")

        device = next(iter(self.pair_vectors.values())).device
        R = R.to(device=device, dtype=torch.float32)

        # cache one Wigner-D matrix per pair type
        D_cache: Dict[PairKey, torch.Tensor] = {}
        for key in self.pair_vectors.keys():
            pair_irreps = mapper.get_pair_irreps(key)
            D_cache[key] = pair_irreps.D_from_matrix(R)  # (n_vec, n_vec)

        # rotate every vector ------------------------------------------------
        new_vectors: Dict[PairKey, torch.Tensor] = {}
        for key, vec in self.pair_vectors.items():
            D = D_cache[key]
            # tensor contraction:  (E, n_vec)
            vec_rot = vec @ D.T
            new_vectors[key] = vec_rot

        # edge indices / lookup are unchanged
        return IrrepsBlockData(
            atoms=self.atoms,
            atom_counts=self.atom_counts,
            pair_vectors=new_vectors,
            pair_edges=self.pair_edges,
            lookup=self.lookup,
            orbital_cfg=self.orbital_cfg,
            basis=self.basis,
        )

    # ------------------------------------------------------------------ serialisation
    def _to_payload(self) -> dict:
        """
        Convert to a CPU-resident, torch-savable python dict.
        """
        vec_cpu = {k: v.detach().cpu() for k, v in self.pair_vectors.items()}
        edges_cpu = {k: v.detach().cpu() for k, v in self.pair_edges.items()}
        return {
            "atoms": list(self.atoms),
            "atom_counts": self.atom_counts,
            "orbital_cfg": self.orbital_cfg.to_dict(),
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

        pair_vec = {k: v.to(device) for k, v in payload["pair_vectors"].items()}
        pair_edges = {k: v.to(device) for k, v in payload["pair_edges"].items()}

        # rebuild lookup
        lookup = {}
        for key, edges in pair_edges.items():
            for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, idx)

        atoms = tuple(payload["atoms"])
        from collections import Counter

        atom_counts = payload.get("atom_counts", Counter(atoms))

        return cls(
            atoms,
            atom_counts,
            pair_vec,
            pair_edges,
            lookup,
            orb_cfg,
            payload.get("basis", "e3nn"),
        ).to(device)
