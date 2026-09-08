"""
sparse_math.py
==============

Sparse-matrix utilities.
"""

from __future__ import annotations

from typing import Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass

import torch
from data.block_matrix import BlockMatrix

TraceAlignment = Dict[str, Tuple[str, torch.Tensor]]


@dataclass(frozen=True)
class BlockProductGroup:
    """One batch of compatible (d_i,d_k) @ (d_k,d_j) contractions."""

    output_key: str
    left_key: str
    right_key: str
    left_indices: torch.Tensor
    right_indices: torch.Tensor
    output_indices: torch.Tensor


@dataclass(frozen=True)
class BlockProductAlignment:
    """Reusable integer topology for a product on a specified output support."""

    groups: tuple[BlockProductGroup, ...]
    left_edges: dict[str, torch.Tensor]
    right_edges: dict[str, torch.Tensor]
    output_edges: dict[str, torch.Tensor]

    @property
    def num_paths(self) -> int:
        return sum(group.left_indices.numel() for group in self.groups)


def _validate_product_matrices(*matrices: BlockMatrix) -> None:
    reference = matrices[0]
    if not reference.pair_blocks:
        raise ValueError("Sparse products require at least one block tensor.")
    sample = next(iter(reference.pair_blocks.values()))
    for matrix in matrices:
        if (
            matrix.atoms != reference.atoms
            or matrix.basis != reference.basis
            or matrix.orbital_cfg.to_dict() != reference.orbital_cfg.to_dict()
        ):
            raise ValueError(
                "Sparse products require matching atoms, basis and orbitals."
            )
        if matrix.pair_blocks.keys() != matrix.pair_edges.keys():
            raise ValueError("Block keys and edge keys must match.")
        for key, block in matrix.pair_blocks.items():
            edges = matrix.pair_edges[key]
            if block.ndim != 3 or block.shape[1:] != matrix.orbital_cfg.block_dims(key):
                raise ValueError(f"Invalid orbital block shape for {key}.")
            if edges.dtype != torch.long or edges.shape != (5, block.shape[0]):
                raise ValueError(f"Expected int64 (5,E) edges for {key}.")
            if block.dtype != sample.dtype or block.device != sample.device:
                raise ValueError("All product blocks must share dtype and device.")


def build_matmul_alignment(
    A: BlockMatrix, B: BlockMatrix, output: BlockMatrix
) -> BlockProductAlignment:
    """Plan P_output(A B), including periodic convolution of lattice shifts.

    (AB)[i,j,L] = sum_(k,L1) A[i,k,L1] B[k,j,L-L1]. Missing
    input blocks contribute zero. Only the requested output edges are retained.
    CPU integer indexing is performed once; numerical evaluation uses grouped
    batched torch products, like the pre-aligned sparse trace implementation.
    """
    _validate_product_matrices(A, B, output)

    def entries(matrix):
        seen = set()
        for key, edges in matrix.pair_edges.items():
            for index, row in enumerate(edges.T.cpu().tolist()):
                edge = tuple(row)
                if edge in seen:
                    raise ValueError(f"Duplicate periodic edge {edge}.")
                seen.add(edge)
                if not (
                    0 <= edge[3] < len(matrix.atoms)
                    and 0 <= edge[4] < len(matrix.atoms)
                ):
                    raise ValueError(f"Invalid atom indices in edge {edge}.")
                if key != f"{matrix.atoms[edge[3]]}-{matrix.atoms[edge[4]]}":
                    raise ValueError(
                        f"Element-pair key {key} disagrees with edge {edge}."
                    )
                yield edge, key, index

    left_by_source = defaultdict(list)
    for edge, key, index in entries(A):
        left_by_source[edge[3]].append((edge, key, index))
    right_lookup = {edge: (key, index) for edge, key, index in entries(B)}
    paths = defaultdict(lambda: ([], [], []))
    for (sx, sy, sz, i, j), out_key, out_index in entries(output):
        for (tx, ty, tz, _, k), left_key, left_index in left_by_source[i]:
            match = right_lookup.get((sx - tx, sy - ty, sz - tz, k, j))
            if match is None:
                continue
            right_key, right_index = match
            indices = paths[(out_key, left_key, right_key)]
            indices[0].append(left_index)
            indices[1].append(right_index)
            indices[2].append(out_index)
    device = next(iter(A.pair_blocks.values())).device
    groups = tuple(
        BlockProductGroup(
            *keys,
            *(
                torch.tensor(index, dtype=torch.long, device=device)
                for index in indices
            ),
        )
        for keys, indices in sorted(paths.items())
    )
    return BlockProductAlignment(
        groups,
        *(
            {key: edges.clone() for key, edges in matrix.pair_edges.items()}
            for matrix in (A, B, output)
        ),
    )


def matmul_sparse_block_matrix_aligned(
    A: BlockMatrix,
    B: BlockMatrix,
    output: BlockMatrix,
    alignment: BlockProductAlignment,
    *,
    chunk_size: int = 4096,
) -> BlockMatrix:
    """Evaluate a fixed-support block product, with bounded temporary memory.

    Keeps output keys, edge order and zero blocks exactly. Autograd is retained
    through gathers, bmm and index_add; only topology is precomputed on CPU.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")
    _validate_product_matrices(A, B, output)
    for matrix, planned in zip(
        (A, B, output),
        (alignment.left_edges, alignment.right_edges, alignment.output_edges),
    ):
        if matrix.pair_edges.keys() != planned.keys() or any(
            not torch.equal(
                matrix.pair_edges[key], edges.to(matrix.pair_edges[key].device)
            )
            for key, edges in planned.items()
        ):
            raise ValueError("Sparse product topology changed; rebuild the alignment.")
    blocks = {key: torch.zeros_like(value) for key, value in output.pair_blocks.items()}
    # Retain a zero derivative even for supports with no matching product paths.
    zero = sum(
        value.reshape(-1)[:1].sum() * 0
        for matrix in (A, B)
        for value in matrix.pair_blocks.values()
    )
    blocks = {key: value + zero for key, value in blocks.items()}
    for group in alignment.groups:
        left = A.pair_blocks[group.left_key]
        right = B.pair_blocks[group.right_key]
        indices = [
            index.to(left.device)
            for index in (group.left_indices, group.right_indices, group.output_indices)
        ]
        for start in range(0, indices[0].numel(), chunk_size):
            a, b, out = (index[start : start + chunk_size] for index in indices)
            values = torch.bmm(left.index_select(0, a), right.index_select(0, b))
            blocks[group.output_key].index_add_(0, out, values)
    return output._replace_pair_blocks(blocks, basis=output.basis)


def trace_matmul_sparse(
    blocks_a: torch.Tensor,  # (E, d, d)
    blocks_b: torch.Tensor,  # (E, d, d)
    edge_index: torch.Tensor,  # (2, E), long
) -> torch.Tensor:
    """
    Compute ``Tr( A · B )`` without ever materialising the dense matrix.
    """
    print("WARNING! Using inefficient trace matmul implementation")
    if blocks_a.shape != blocks_b.shape:
        raise ValueError("blocks_a and blocks_b must have same shape")

    if edge_index.shape[0] != blocks_a.shape[0]:
        raise ValueError("edge_index cols must match number of blocks")

    pairs = edge_index.tolist()
    lookup = {tuple(p): k for k, p in enumerate(pairs)}

    out = torch.zeros((), dtype=blocks_a.dtype, device=blocks_a.device)
    for k, (sx, sy, sz, i, j) in enumerate(pairs):
        rev = (-sx, -sy, -sz, j, i)
        rev_k = lookup.get(rev, None)
        if rev_k is None:
            continue
        out = out + torch.trace(blocks_a[k] @ blocks_b[rev_k])
    return out


# ---------------------------------------------------------------------------
#   BlockMatrix sparse traces
# ---------------------------------------------------------------------------
def trace_matmul_sparse_block_matrix(A: BlockMatrix, B: BlockMatrix) -> torch.Tensor:
    """
    Scalar trace **via per-edge loop** using lookup table.
    Works for any BlockMatrix / IrrepsBlockData combination.
    """
    out = torch.zeros(
        (1),
        dtype=list(A.pair_blocks.values())[0].dtype,
        device=list(A.pair_blocks.values())[0].device,
        requires_grad=True,
    )
    for (sx, sy, sz, i, j), (key, k) in A.lookup.items():
        if (-sx, -sy, -sz, j, i) not in B.lookup:
            raise ValueError(
                f"Edge {(-sx, -sy, -sz, j, i)} (reverse of {(sx, sy, sz, i, j)}) missing in second matrix."
            )
        key_rev, k_rev = B.lookup[(-sx, -sy, -sz, j, i)]
        out = out + torch.trace(A.pair_blocks[key][k] @ B.pair_blocks[key_rev][k_rev])
    return out


def build_trace_alignment_from_pair_edges(
    pair_edges_a: Dict[str, torch.Tensor],
    pair_edges_b: Dict[str, torch.Tensor] | None = None,
    *,
    device: torch.device | str | None = None,
) -> TraceAlignment:
    """
    Precompute reverse-edge alignment for vectorized sparse traces.

    For each key ``A-B`` in ``pair_edges_a``, the returned mapping stores:
    - reverse key ``B-A`` in ``pair_edges_b``
    - a 1-D LongTensor of indices aligning reverse edges in ``pair_edges_b`` to
      the order of edges in ``pair_edges_a``
    """
    if pair_edges_b is None:
        pair_edges_b = pair_edges_a

    alignment: TraceAlignment = {}
    for key, edges_a in pair_edges_a.items():
        el_a, el_b = key.split("-")
        rev_key = f"{el_b}-{el_a}"
        if rev_key not in pair_edges_b:
            raise ValueError(
                f"Key {rev_key} missing in second edge set (reverse of {key})."
            )

        edges_b_rev = pair_edges_b[rev_key]
        mapping = {
            tuple(map(int, edge.tolist())): idx
            for idx, edge in enumerate(edges_b_rev.t())
        }

        indices_b = []
        for edge in edges_a.t():
            sx, sy, sz, i, j = map(int, edge.tolist())
            rev_edge = (-sx, -sy, -sz, j, i)
            if rev_edge not in mapping:
                raise ValueError(
                    f"Edge {rev_edge} (reverse of {(sx, sy, sz, i, j)}) missing in second edge set."
                )
            indices_b.append(mapping[rev_edge])

        idx_device = device if device is not None else edges_a.device
        alignment[key] = (
            rev_key,
            torch.tensor(indices_b, device=idx_device, dtype=torch.long),
        )

    return alignment


def build_trace_alignment(A: BlockMatrix, B: BlockMatrix) -> TraceAlignment:
    """Convenience wrapper building alignment from BlockMatrix edge dictionaries."""
    return build_trace_alignment_from_pair_edges(A.pair_edges, B.pair_edges)


def trace_matmul_sparse_block_matrix_aligned(
    A: BlockMatrix,
    B: BlockMatrix,
    alignment: TraceAlignment,
) -> torch.Tensor:
    """
    Vectorized sparse trace using a precomputed reverse-edge alignment.
    """
    total = torch.zeros(
        (),
        dtype=list(A.pair_blocks.values())[0].dtype,
        device=list(A.pair_blocks.values())[0].device,
    )
    for key, blk_a in A.pair_blocks.items():
        if key not in alignment:
            raise ValueError(f"Missing trace alignment for key {key}.")
        rev_key, idx_b = alignment[key]
        if rev_key not in B.pair_blocks:
            raise ValueError(
                f"Key {rev_key} missing in second matrix (reverse of {key})."
            )
        blk_b_rev = B.pair_blocks[rev_key]
        if idx_b.device != blk_b_rev.device:
            idx_b = idx_b.to(blk_b_rev.device)
        blk_b_aligned = blk_b_rev.index_select(0, idx_b)
        total = total + torch.einsum("bij,bji->", blk_a, blk_b_aligned)
    return total


def trace_matmul_sparse_snap_vectorized(A: BlockMatrix, B: BlockMatrix) -> torch.Tensor:
    """
    Vectorized per *directed* key.

    For each key ``A-B`` we fetch the reverse key ``B-A`` from ``B``.
    This matches the mathematical trace:  Σ_{i,j} Tr( A_{ij} · B_{ji} ).
    """
    print("Warning! [opt] Building trace alignment on the fly")
    alignment = build_trace_alignment(A, B)
    return trace_matmul_sparse_block_matrix_aligned(A, B, alignment)


__all__: Tuple[str, ...] = (
    "BlockProductAlignment",
    "build_matmul_alignment",
    "matmul_sparse_block_matrix_aligned",
    "trace_matmul_sparse",
    "build_trace_alignment_from_pair_edges",
    "build_trace_alignment",
    "trace_matmul_sparse_block_matrix",
    "trace_matmul_sparse_block_matrix_aligned",
    "trace_matmul_sparse_snap_vectorized",
)
