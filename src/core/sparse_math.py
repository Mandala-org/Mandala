"""
sparse_math.py
==============

Sparse-matrix utilities **independent** of message-passing.

Only `trace_matmul_sparse` remains a free function; vector/block mapping
is now routed through :class:`core.block_irrep_mapper.BlockIrrepMapper`.
"""

from __future__ import annotations

from typing import Tuple

import torch


# --------------------------------------------------------------------------- #
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

    if edge_index.shape[1] != blocks_a.shape[0]:
        raise ValueError("edge_index cols must match number of blocks")

    pairs = edge_index.t().tolist()
    lookup = {tuple(p): k for k, p in enumerate(pairs)}

    out = torch.zeros(
        (), dtype=blocks_a.dtype, device=blocks_a.device, requires_grad=False
    )
    for k, (i, j) in enumerate(pairs):
        rev = (j, i)
        rev_k = lookup.get(rev, None)
        if rev_k is None:
            continue
        out = out + torch.trace(blocks_a[k] @ blocks_b[rev_k])
    return out


# thin wrappers kept for convenience --------------------------------------- #
def blocks_to_vectors(mapper, pair, blocks):
    """
    Convenience shim → delegates to :class:`BlockIrrepMapper`.
    """
    return mapper.blocks_to_vectors(pair, blocks)


def vectors_to_blocks(mapper, pair, vectors):
    """
    Inverse shim.
    """
    return mapper.vectors_to_blocks(pair, vectors)


# ---------------------------------------------------------------------------
#   Snapshot-level sparse traces
# ---------------------------------------------------------------------------
def trace_matmul_sparse_snap(A, B):
    """
    Scalar trace **via per-edge loop** using lookup table.
    Works for any SnapshotBlockData / SnapshotIrrepsData combination.
    """
    out = torch.zeros(
        (),
        dtype=list(A.pair_blocks.values())[0].dtype,
        device=list(A.pair_blocks.values())[0].device,
    )
    for (i, j), (key, k) in A.lookup.items():
        if (j, i) not in B.lookup:
            continue
        key_rev, k_rev = B.lookup[(j, i)]
        out = out + torch.trace(A.pair_blocks[key][k] @ B.pair_blocks[key_rev][k_rev])
    return out


def trace_matmul_sparse_snap_vectorized(A, B):
    """
    Vectorised per *directed* key.

    For each key ``A-B`` we fetch the reverse key ``B-A`` from ``B``.
    This matches the mathematical trace:  Σ_{i,j} Tr( A_{ij} · B_{ji} ).
    """
    total = 0.0
    for key in A.keys():
        el_A, el_B = key.split("-")
        rev_key = f"{el_B}-{el_A}"
        if rev_key not in B.keys():
            continue

        blk_a = A.pair_blocks[key]  # (E, d_A, d_B)
        blk_b_rev = B.pair_blocks[rev_key]  # (E_rev, d_B, d_A)

        edges_a = A.pair_edges[key]  # (2, E)   (i, j)
        edges_b_rev = B.pair_edges[rev_key]  # (2, E')  (j, i)

        # map (j,i) tuple -> index in B
        mapping = {(int(e[0]), int(e[1])): idx for idx, e in enumerate(edges_b_rev.t())}

        # build index list such that order matches edges_a
        idx_rev = torch.tensor(
            [mapping[(int(j), int(i))] for i, j in edges_a.t()],
            device=blk_a.device,
        )

        blk_b_aligned = blk_b_rev[idx_rev]  # (E, d_B, d_A)

        # Trace of A_ij · B_ji
        total = total + torch.einsum("bij,bji->", blk_a, blk_b_aligned)

    return total


__all__: Tuple[str, ...] = (
    "trace_matmul_sparse",
    "blocks_to_vectors",
    "vectors_to_blocks",
    "trace_matmul_sparse_snap",
    "trace_matmul_sparse_snap_vectorized",
)
