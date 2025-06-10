"""
Additional low-level checks for (i) global-index access and (ii) edge mappings
inside BlockMatrix / IrrepsBlockData.
"""

import random
from pathlib import Path
import torch
import pytest

from data.openmx_parser import parse_openmx_scfout
from core.orbital_irrep_config import OrbitalIrrepConfig


@pytest.fixture(scope="module")
def snapshot():
    atoms = list("HHHHOO")  # global indices 0…5
    cfg = OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})
    file = Path("./data/small/H2O/original/H2O.matrix")
    return parse_openmx_scfout(file, atoms, cfg, convention="openmx")


# --------------------------------------------------------------------------- #
def test_blockmatrix_lookup(snapshot):
    """Global indexing ((i,j)) must hit the *same* tensor as pair-local lookup."""
    B = snapshot.hamiltonian  # BlockMatrix
    rng = random.Random(2024)

    for _ in range(10):  # random edges
        i = rng.randrange(len(B.atoms))
        j = rng.randrange(len(B.atoms))
        key, local = B.lookup[(i, j)]  # local index *inside* key bucket
        blk_global = B[(i, j)]  # global accessor
        blk_local = B.pair_blocks[key][local]  # local accessor
        assert torch.allclose(blk_global, blk_local, atol=1e-7)


def test_irrepsblockdata_lookup(snapshot):
    V = snapshot.density.to_vectors()  # IrrepsBlockData
    rng = random.Random(17)

    for _ in range(10):
        i = rng.randrange(len(V.atoms))
        j = rng.randrange(len(V.atoms))
        key, local = V.lookup[(i, j)]
        vec_g = V[(i, j)]
        vec_l = V.pair_vectors[key][local]
        assert torch.allclose(vec_g, vec_l, atol=1e-7)
