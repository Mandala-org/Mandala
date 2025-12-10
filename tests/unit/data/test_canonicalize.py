import pytest
import torch
from collections import Counter
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig


@pytest.mark.unit
def test_canonicalize_edges():
    # Setup
    atoms = ("A", "A")
    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"A": "1s"})

    # Create dummy blocks (1x1 for simplicity)
    # Key "A-A"
    # Edges:
    # 1. (0,0,0, 0, 0) - Diag, src=0
    # 2. (0,0,0, 1, 1) - Diag, src=1
    # 3. (1,0,0, 0, 1) - Off-diag, dist=10 (dummy)
    # 4. (0,1,0, 0, 1) - Off-diag, dist=5 (dummy)
    # 5. (0,0,1, 0, 1) - Off-diag, dist=5 (dummy), different shift

    # We need positions and box to compute distances.
    # Let's set positions such that distances are predictable.
    # Pos: A0 at (0,0,0), A1 at (2,0,0)
    # Box: 10x10x10 identity

    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    box = torch.eye(3) * 10.0

    # Edges definition (sx, sy, sz, src, dst)
    # 1. Diag A0: (0,0,0, 0, 0) -> dist 0
    # 2. Diag A1: (0,0,0, 1, 1) -> dist 0
    # 3. Off-diag A0->A1 shift(0,0,0): (0,0,0, 0, 1) -> delta = (2,0,0) -> dist 2.0
    # 4. Off-diag A0->A1 shift(1,0,0): (1,0,0, 0, 1) -> delta = (2,0,0) + (10,0,0) = (12,0,0) -> dist 12.0
    # 5. Off-diag A0->A1 shift(-1,0,0): (-1,0,0, 0, 1) -> delta = (2,0,0) - (10,0,0) = (-8,0,0) -> dist 8.0

    # Let's mix them up in input
    edges_list = [
        [1, 0, 0, 0, 1],  # dist 12
        [0, 0, 0, 1, 1],  # diag 1
        [-1, 0, 0, 0, 1],  # dist 8
        [0, 0, 0, 0, 0],  # diag 0
        [0, 0, 0, 0, 1],  # dist 2
    ]

    # Expected order:
    # Diagonals first, sorted by src:
    # 1. (0,0,0, 0, 0) [Index 3 in input]
    # 2. (0,0,0, 1, 1) [Index 1 in input]
    # Off-diagonals, sorted by distance:
    # 3. (0,0,0, 0, 1) [Index 4, dist 2]
    # 4. (-1,0,0, 0, 1) [Index 2, dist 8]
    # 5. (1,0,0, 0, 1) [Index 0, dist 12]

    edges_tensor = torch.tensor(edges_list).t()  # (5, E)

    # Dummy blocks
    blocks = torch.randn(5, 1, 1)

    pair_blocks = {"A-A": blocks}
    pair_edges = {"A-A": edges_tensor}

    # Lookup needs to be valid
    lookup = {}
    for idx, row in enumerate(edges_list):
        lookup[tuple(row)] = ("A-A", idx)

    bm = BlockMatrix(
        atoms, atom_counts, pair_blocks, pair_edges, lookup, orbital_cfg, "e3nn"
    )

    snap = Snapshot(bm, bm, bm, positions=positions, box=box)

    # Run canonicalize
    new_snap = snap.canonicalize_edges()

    new_edges = new_snap.density.pair_edges["A-A"].t().tolist()

    expected_order = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
    ]

    assert new_edges == expected_order

    # Check tie-breaking
    # Add two edges with same distance but different shift
    # Pos A0=(0,0,0), A1=(2,0,0)
    # Edge 1: (0, 1, 0, 0, 0) -> shift (0,10,0) -> delta (0,10,0) -> dist 10
    # Edge 2: (0, 0, 1, 0, 0) -> shift (0,0,10) -> delta (0,0,10) -> dist 10
    # Tie-breaker: (sx, sy, sz, src, dst)
    # (0,0,1,0,0) < (0,1,0,0,0)

    edges_list_2 = [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    edges_tensor_2 = torch.tensor(edges_list_2).t()
    blocks_2 = torch.randn(2, 1, 1)
    pair_blocks_2 = {"A-A": blocks_2}
    pair_edges_2 = {"A-A": edges_tensor_2}
    lookup_2 = {tuple(r): ("A-A", i) for i, r in enumerate(edges_list_2)}

    bm2 = BlockMatrix(
        atoms, atom_counts, pair_blocks_2, pair_edges_2, lookup_2, orbital_cfg, "e3nn"
    )
    snap2 = Snapshot(bm2, bm2, bm2, positions=positions, box=box)

    new_snap2 = snap2.canonicalize_edges()
    new_edges2 = new_snap2.density.pair_edges["A-A"].t().tolist()

    expected_order_2 = [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]

    assert new_edges2 == expected_order_2
