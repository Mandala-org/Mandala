"""
Unit tests that target *BlockMatrix.rotate* in isolation.
"""

import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix


def _make_periodic_angular_matrix() -> BlockMatrix:
    """One s+p site with distinct onsite and reverse-image blocks."""
    from collections import Counter

    cfg = OrbitalIrrepConfig.from_dict({"H": ["1x0e", "1x1o"]})
    atoms = ("H",)
    edges = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    ).T
    blocks = torch.arange(1, 49, dtype=torch.float32).reshape(3, 4, 4)
    lookup = {tuple(edge.tolist()): ("H-H", i) for i, edge in enumerate(edges.T)}
    return BlockMatrix(
        atoms,
        Counter(atoms),
        {"H-H": blocks},
        {"H-H": edges},
        lookup,
        cfg,
        basis="e3nn",
    )


@pytest.mark.physics
def test_blockmatrix_rotation_uses_angular_action_and_preserves_periodic_edges():
    mat = _make_periodic_angular_matrix()
    # A +90 degree active z rotation has this known real-vector action in the
    # x,y,z basis used by 1o: x -> y, y -> -x, z -> z.  This literal matrix is
    # independent of BlockMatrix.rotate and checks the p-orbital convention.
    orbital_rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    R = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rotated = mat.rotate(R)

    expected = orbital_rotation @ mat.pair_blocks["H-H"] @ orbital_rotation.T
    torch.testing.assert_close(rotated.pair_blocks["H-H"], expected, atol=3e-4, rtol=0)
    assert torch.equal(rotated.pair_edges["H-H"], mat.pair_edges["H-H"])
    assert rotated.lookup == mat.lookup
    # The +x and -x image records remain distinct and retain their own values.
    assert (1, 0, 0, 0, 0) in rotated.lookup
    assert (-1, 0, 0, 0, 0) in rotated.lookup
