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


__all__: Tuple[str, ...] = (
    "trace_matmul_sparse",
    "blocks_to_vectors",
    "vectors_to_blocks",
)
