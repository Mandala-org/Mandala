import itertools

import numpy as np
import torch

from core.sparse_math import trace_matmul_sparse


def make_toy_sparse(n_atoms=3, d=2):
    pairs = list(itertools.product(range(n_atoms), repeat=2))
    edge_index = torch.tensor(pairs, dtype=torch.long).t()
    E = edge_index.shape[1]
    rng = torch.Generator().manual_seed(11)
    a = torch.randn(E, d, d, generator=rng)
    b = torch.randn(E, d, d, generator=rng)
    return edge_index, a, b


def dense_from_blocks(blocks, edge_index, d, n_atoms):
    big = np.zeros((n_atoms * d, n_atoms * d))
    for k, (src, dst) in enumerate(edge_index.t().tolist()):
        r0, r1 = src * d, (src + 1) * d
        c0, c1 = dst * d, (dst + 1) * d
        big[r0:r1, c0:c1] = blocks[k].cpu().numpy()
    return big


def test_trace_matches_dense():
    n_atoms, d = 3, 2
    edge_index, A, B = make_toy_sparse(n_atoms, d)

    sparse_tr = trace_matmul_sparse(A, B, edge_index)

    dense_A = dense_from_blocks(A, edge_index, d, n_atoms)
    dense_B = dense_from_blocks(B, edge_index, d, n_atoms)
    dense_tr = np.trace(dense_A @ dense_B)

    assert np.isclose(sparse_tr.cpu().item(), dense_tr, atol=1e-6)
