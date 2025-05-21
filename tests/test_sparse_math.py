import itertools
import torch
import numpy as np

from core.sparse_math import trace_matmul_sparse
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot_block import MatrixBlockData
from core.sparse_math import (
    trace_matmul_sparse_snap,
    trace_matmul_sparse_snap_vectorized,
)


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


def make_mock_snapshot():
    atoms = ("H", "H", "O", "H", "H", "O")
    cfg = OrbitalIrrepConfig.from_dict({"H": ["2x0e"], "O": ["1x0e", "1x1o"]})
    mapper = BlockIrrepMapper(cfg)
    pair_blocks, pair_edges, lookup = {}, {}, {}
    for i, el_i in enumerate(atoms):
        for j, el_j in enumerate(atoms):
            key = f"{el_i}-{el_j}"
            d_i, d_j = mapper.block_dims(key)
            blk = torch.randn(d_i, d_j)
            pair_blocks.setdefault(key, []).append(blk)
            pair_edges.setdefault(key, []).append([i, j])
            lookup[(i, j)] = (key, len(pair_blocks[key]) - 1)
    pair_blocks = {k: torch.stack(v) for k, v in pair_blocks.items()}
    pair_edges = {
        k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
    }
    return MatrixBlockData(atoms, pair_blocks, pair_edges, lookup, mapper, "openmx")


def test_trace_sparse_vs_vectorized():
    A = make_mock_snapshot()
    B = make_mock_snapshot()
    t_sparse = trace_matmul_sparse_snap(A, B)
    t_vec = trace_matmul_sparse_snap_vectorized(A, B)
    assert torch.allclose(t_sparse, t_vec, atol=1e-6)


def test_trace_sparse_vs_dense():
    A = make_mock_snapshot()
    B = make_mock_snapshot()
    t_sparse = trace_matmul_sparse_snap(A, B)
    bigA = A.to_dense()
    bigB = B.to_dense()
    dense_trace = torch.trace(bigA @ bigB).item()
    assert np.isclose(t_sparse.item(), dense_trace, atol=1e-4)


def test_trace_matches_dense():
    n_atoms, d = 3, 2
    edge_index, A, B = make_toy_sparse(n_atoms, d)

    sparse_tr = trace_matmul_sparse(A, B, edge_index)

    dense_A = dense_from_blocks(A, edge_index, d, n_atoms)
    dense_B = dense_from_blocks(B, edge_index, d, n_atoms)
    dense_tr = np.trace(dense_A @ dense_B)

    assert np.isclose(sparse_tr.cpu().item(), dense_tr, atol=1e-6)
