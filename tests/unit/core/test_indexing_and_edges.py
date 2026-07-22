"""
Additional low-level checks for (i) global-index access and (ii) edge mappings
inside BlockMatrix / IrrepsBlockData.
"""

import pytest
import random
import torch

from core.block_irrep_mapper import BlockIrrepMapper


@pytest.fixture(scope="module")
def data(small_angular_snapshot_e3nn):
    snapshot = small_angular_snapshot_e3nn
    return snapshot, BlockIrrepMapper(snapshot.hamiltonian.orbital_cfg)


# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_blockmatrix_lookup(data):
    """Global indexing ((i,j)) must hit the *same* tensor as pair-local lookup."""
    snapshot, mapper = data
    B = snapshot.hamiltonian  # BlockMatrix
    rng = random.Random(2024)

    for _ in range(10):  # random edges
        i = rng.randrange(len(B.atoms))
        j = rng.randrange(len(B.atoms))
        key, local = B.lookup[(0, 0, 0, i, j)]  # local index *inside* key bucket
        blk_global = B[(0, 0, 0, i, j)]  # global accessor
        blk_local = B.pair_blocks[key][local]  # local accessor
        assert torch.allclose(blk_global, blk_local, atol=1e-7)


@pytest.mark.unit
def test_irrepsblockdata_lookup(data):
    """Global indexing ((i,j)) must hit the *same* vector as pair-local lookup."""
    snapshot, mapper = data
    V = snapshot.density.to_vectors(mapper)  # IrrepsBlockData
    rng = random.Random(17)

    for _ in range(10):
        i = rng.randrange(len(V.atoms))
        j = rng.randrange(len(V.atoms))
        key, local = V.lookup[(0, 0, 0, i, j)]  # local index *inside* key bucket
        vec_g = V[(0, 0, 0, i, j)]  # global accessor
        vec_l = V.pair_vectors[key][local]  # local accessor
        assert torch.allclose(vec_g, vec_l, atol=1e-7)
