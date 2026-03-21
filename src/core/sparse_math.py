"""
sparse_math.py
==============

Sparse-matrix utilities **independent** of message-passing.

Only `trace_matmul_sparse` remains a free function; vector/block mapping
is now routed through :class:`core.block_irrep_mapper.BlockIrrepMapper`.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix

TraceAlignment = Dict[str, Tuple[str, torch.Tensor]]


def trace_matmul_sparse(
    blocks_a: torch.Tensor,  # (E, d, d)
    blocks_b: torch.Tensor,  # (E, d, d)
    edge_index: torch.Tensor,  # (2, E), long
) -> torch.Tensor:
    """
    Compute ``Tr( A · B )`` without ever materialising the dense matrix.
    """
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


# thin wrappers kept for convenience --------------------------------------- #
def blocks_to_vectors(
    mapper: BlockIrrepMapper, pair: Union[Tuple[str, str], str], blocks: torch.Tensor
) -> torch.Tensor:
    """
    Convenience shim -> delegates to :class:`BlockIrrepMapper`.
    """
    return mapper.blocks_to_vectors(pair, blocks)


def vectors_to_blocks(
    mapper: BlockIrrepMapper, pair: Union[Tuple[str, str], str], vectors: torch.Tensor
) -> torch.Tensor:
    """
    Inverse shim.
    """
    return mapper.vectors_to_blocks(pair, vectors)


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
    alignment = build_trace_alignment(A, B)
    return trace_matmul_sparse_block_matrix_aligned(A, B, alignment)


__all__: Tuple[str, ...] = (
    "trace_matmul_sparse",
    "blocks_to_vectors",
    "vectors_to_blocks",
    "build_trace_alignment_from_pair_edges",
    "build_trace_alignment",
    "trace_matmul_sparse_block_matrix",
    "trace_matmul_sparse_block_matrix_aligned",
    "trace_matmul_sparse_snap_vectorized",
)
