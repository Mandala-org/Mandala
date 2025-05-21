"""
Unit tests that target *BlockMatrix.rotate* in isolation.
"""

import math
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix


def _make_small_matrix() -> BlockMatrix:
    """
    Construct a **tiny** 2-atom block matrix (one block per orientation)
    directly in *e3nn* basis for fast tests.
    """
    cfg = OrbitalIrrepConfig.from_dict({"H": ["1x0e"], "O": ["1x0e"]})
    mapper = BlockIrrepMapper(cfg)
    atoms = ("H", "O")

    # single 1×1 scalar blocks → easy numerics
    pair_blocks = {
        "H-H": torch.randn(1, 1, 1),
        "H-O": torch.randn(1, 1, 1),
        "O-H": torch.randn(1, 1, 1),
        "O-O": torch.randn(1, 1, 1),
    }
    pair_edges = {
        "H-H": torch.tensor([[0], [0]]),
        "H-O": torch.tensor([[0], [1]]),
        "O-H": torch.tensor([[1], [0]]),
        "O-O": torch.tensor([[1], [1]]),
    }
    lookup = {
        (0, 0): ("H-H", 0),
        (0, 1): ("H-O", 0),
        (1, 0): ("O-H", 0),
        (1, 1): ("O-O", 0),
    }

    return BlockMatrix(atoms, pair_blocks, pair_edges, lookup, mapper, basis="e3nn")


def _rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32
    )


def test_blockmatrix_rotation_invariance():
    mat = _make_small_matrix()
    R = _rot_y(math.pi / 3.0)
    mat_rot = mat.rotate(R)

    # All blocks are scalars → rotation acts as identity;
    # check exact equality as a sanity test.
    for key in mat.keys():
        assert torch.allclose(mat[key], mat_rot[key], atol=1e-7)
