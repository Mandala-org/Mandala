"""
sparse_math.py
==============

Sparse-matrix utilities **independent** of message-passing.

Only `trace_matmul_sparse` remains a free function; vector/block mapping
is now routed through :class:`core.block_irrep_mapper.BlockIrrepMapper`.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix


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

    #! Edge manipulation
    # <assumptions>
    # - `edge_index` is (E, 5) or (5, E) depending on usage, here assumed list of 5-tuples.
    # - Builds lookup for O(1) access to block indices.
    # </assumptions>
    # <implementation>
    # - Converts `edge_index` to list of tuples.
    # - Creates dictionary mapping tuple to index `k`.
    # </implementation>
    pairs = edge_index.tolist()
    lookup = {tuple(p): k for k, p in enumerate(pairs)}

    out = torch.zeros((), dtype=blocks_a.dtype, device=blocks_a.device)
    #! Edge manipulation
    # <assumptions>
    # - Computes trace of A * B.
    # - Requires finding the symmetric block B_{ji} for each A_{ij}.
    # </assumptions>
    # <implementation>
    # - Iterates over edges of A.
    # - Finds index `rev_k` of symmetric edge `(-sx, -sy, -sz, j, i)` in B.
    # - Accumulates trace of product.
    # </implementation>
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
    Convenience shim → delegates to :class:`BlockIrrepMapper`.
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
    #! Edge manipulation
    # <assumptions>
    # - Computes trace of A * B using `BlockMatrix` lookups.
    # - Iterates over all blocks in A.
    # </assumptions>
    # <implementation>
    # - For each block A_{ij} at `(sx, sy, sz, i, j)`:
    # - Checks if symmetric block B_{ji} at `(-sx, -sy, -sz, j, i)` exists.
    # - Accumulates trace of product.
    # </implementation>
    for (sx, sy, sz, i, j), (key, k) in A.lookup.items():
        if (-sx, -sy, -sz, j, i) not in B.lookup:
            raise ValueError(
                f"Edge {(-sx, -sy, -sz, j, i)} (reverse of {(sx, sy, sz, i, j)}) missing in second matrix."
            )
        key_rev, k_rev = B.lookup[(-sx, -sy, -sz, j, i)]
        out = out + torch.trace(A.pair_blocks[key][k] @ B.pair_blocks[key_rev][k_rev])
    return out


def trace_matmul_sparse_snap_vectorized(A: BlockMatrix, B: BlockMatrix) -> torch.Tensor:
    """
    Vectorized per *directed* key.

    For each key ``A-B`` we fetch the reverse key ``B-A`` from ``B``.
    This matches the mathematical trace:  Σ_{i,j} Tr( A_{ij} · B_{ji} ).
    """
    total = torch.tensor(
        0.0,
        dtype=list(A.pair_blocks.values())[0].dtype,
        device=list(A.pair_blocks.values())[0].device,
    )
    for key in A.keys():
        el_A, el_B = key.split("-")
        rev_key = f"{el_B}-{el_A}"
        if rev_key not in B.keys():
            raise ValueError(
                f"Key {rev_key} missing in second matrix (reverse of {key} in first matrix)."
            )

        blk_a = A.pair_blocks[key]  # (E, d_A, d_B)
        blk_b_rev = B.pair_blocks[rev_key]  # (E_rev, d_B, d_A)

        edges_a = A.pair_edges[key]  # (2, E)   (i, j)
        #! Edge manipulation
        # <assumptions>
        # - Retrieves edges for the reverse key block in B.
        # </assumptions>
        # <implementation>
        # - Accesses `pair_edges` for `rev_key`.
        # </implementation>
        edges_b_rev = B.pair_edges[rev_key]  # (2, E')  (j, i)

        # map (j,i) tuple -> index in B
        #! Edge manipulation
        # <assumptions>
        # - Builds mapping for fast alignment of B's blocks to A's edges.
        # </assumptions>
        # <implementation>
        # - Maps `(sx, sy, sz, i, j)` to index `idx` for `edges_b_rev`.
        # </implementation>
        mapping = {
            tuple(map(int, (sx, sy, sz, i, j))): idx
            for idx, (sx, sy, sz, i, j) in enumerate(edges_b_rev.t())
        }

        # build index list such that order matches edges_a
        #! Edge manipulation
        # <assumptions>
        # - Aligns B's blocks to match the order of A's blocks for vectorized operation.
        # - Uses symmetric edge property.
        # - Raises error if edges are missing (incompatible sparsity patterns).
        # </assumptions>
        # <implementation>
        # - For each edge in A `(sx, sy, sz, i, j)`:
        # - Checks if symmetric edge `(-sx, -sy, -sz, j, i)` exists in B.
        # - If not, raises ValueError.
        # - Collects indices for B.
        # - Performs vectorized einsum.
        # </implementation>
        indices_b = []

        for sx, sy, sz, i, j in edges_a.t():
            rev_edge = tuple(map(int, (-sx, -sy, -sz, j, i)))
            if rev_edge not in mapping:
                raise ValueError(
                    f"Edge {rev_edge} (reverse of {(sx, sy, sz, i, j)}) missing in second matrix."
                )
            indices_b.append(mapping[rev_edge])

        idx_b_tensor = torch.tensor(indices_b, device=blk_a.device, dtype=torch.long)
        blk_b_aligned = blk_b_rev[idx_b_tensor]

        # Trace of A_ij · B_ji
        total = total + torch.einsum("bij,bji->", blk_a, blk_b_aligned)

    return total


__all__: Tuple[str, ...] = (
    "trace_matmul_sparse",
    "blocks_to_vectors",
    "vectors_to_blocks",
    "trace_matmul_sparse_block_matrix",
    "trace_matmul_sparse_snap_vectorized",
)
