"""
Additional low-level checks for (i) global-index access and (ii) edge mappings
inside BlockMatrix / IrrepsBlockData.
"""

import pytest
import random
from pathlib import Path
import torch

from data.openmx_parser import parse_openmx_scfout
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper


@pytest.fixture(scope="module")
def data():
    atoms = list("HHHHOO")  # global indices 0…5
    cfg = OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})
    file = Path("./data/small/H2O/original/H2O.matrix")
    return parse_openmx_scfout(file, atoms, cfg, convention="openmx"), BlockIrrepMapper(
        cfg
    )


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
        key, local = B.lookup[(i, j)]  # local index *inside* key bucket
        blk_global = B[(i, j)]  # global accessor
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
        key, local = V.lookup[(i, j)]
        vec_g = V[(i, j)]
        vec_l = V.pair_vectors[key][local]
        assert torch.allclose(vec_g, vec_l, atol=1e-7)
