import pytest
import torch
import numpy as np

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix
from core.sparse_math import (
    trace_matmul_sparse_block_matrix,
    trace_matmul_sparse_snap_vectorized,
)


def mock_block_matrix():
    atoms = ("H", "H", "O", "H", "H", "O")
    from collections import Counter

    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": ["2x0e"], "O": ["1x0e", "1x1o"]})
    mapper = BlockIrrepMapper(orbital_cfg)
    pair_blocks, pair_edges, lookup = {}, {}, {}
    for i, el_i in enumerate(atoms):
        for j, el_j in enumerate(atoms):
            key = f"{el_i}-{el_j}"
            d_i, d_j = mapper.block_dims(key)
            blk = torch.randn(d_i, d_j)
            pair_blocks.setdefault(key, []).append(blk)
            pair_edges.setdefault(key, []).append([0, 0, 0, i, j])
            lookup[(0, 0, 0, i, j)] = (key, len(pair_blocks[key]) - 1)
    pair_blocks = {k: torch.stack(v) for k, v in pair_blocks.items()}
    pair_edges = {
        k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
    }
    return BlockMatrix(
        atoms,
        atom_counts,
        pair_blocks,
        pair_edges,
        lookup,
        mapper.orbital_cfg,
        "openmx",
    )


@pytest.mark.unit
def test_trace_sparse_vs_vectorized():
    A = mock_block_matrix()
    B = mock_block_matrix()
    t_sparse = trace_matmul_sparse_block_matrix(A, B)
    t_vec = trace_matmul_sparse_snap_vectorized(A, B)
    assert torch.allclose(t_sparse, t_vec, atol=1e-5)


@pytest.mark.unit
def test_trace_sparse_vs_dense():
    A = mock_block_matrix()
    B = mock_block_matrix()
    t_sparse = trace_matmul_sparse_block_matrix(A, B)
    bigA = A.to_dense()
    bigB = B.to_dense()
    dense_trace = torch.trace(bigA @ bigB).item()
    assert np.isclose(t_sparse.item(), dense_trace, atol=1e-4)


@pytest.mark.unit
def test_trace_matches_dense():
    A = mock_block_matrix()
    B = mock_block_matrix()
    sparse_tr = trace_matmul_sparse_block_matrix(A, B)
    dense_A = A.to_dense()
    dense_B = B.to_dense()
    dense_tr = np.trace(dense_A @ dense_B)
    assert np.isclose(sparse_tr.cpu().item(), dense_tr, atol=1e-6)
