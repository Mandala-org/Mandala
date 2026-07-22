import pytest
import torch
from collections import Counter
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.sparse_math import build_trace_alignment_from_pair_edges


@pytest.fixture
def mock_block_matrix():
    # Atoms: A, B
    atoms = ("A", "B")
    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"A": "1s", "B": "1s"})

    # Key "A-A":
    # 0. (0,0,0, 0, 0) - Diag
    # 1. (1,0,0, 0, 0) - Off-diag (self-image)
    edges_AA = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]).t()
    blocks_AA = torch.randn(2, 1, 1)

    # Key "A-B":
    # 0. (0,0,0, 0, 1)
    # 1. (0,1,0, 0, 1) - Repeated pair, different shift
    edges_AB = torch.tensor([[0, 0, 0, 0, 1], [0, 1, 0, 0, 1]]).t()
    blocks_AB = torch.randn(2, 1, 1)

    pair_blocks = {"A-A": blocks_AA, "A-B": blocks_AB}
    pair_edges = {"A-A": edges_AA, "A-B": edges_AB}

    lookup = {}
    for i, edge in enumerate(edges_AA.t().tolist()):
        lookup[tuple(edge)] = ("A-A", i)
    for i, edge in enumerate(edges_AB.t().tolist()):
        lookup[tuple(edge)] = ("A-B", i)

    return BlockMatrix(
        atoms, atom_counts, pair_blocks, pair_edges, lookup, orbital_cfg, "e3nn"
    )


def test_getitem_tuple_5(mock_block_matrix):
    # Test (sx, sy, sz, i, j) access
    bm = mock_block_matrix

    # A-A diag
    edge = (0, 0, 0, 0, 0)
    val = bm[edge]
    assert torch.allclose(val, bm.pair_blocks["A-A"][0])

    # A-B repeated pair 2
    edge2 = (0, 1, 0, 0, 1)
    val2 = bm[edge2]
    assert torch.allclose(val2, bm.pair_blocks["A-B"][1])


def test_getitem_tuple_2(mock_block_matrix):
    # Test (i, j) access - should sum over images
    bm = mock_block_matrix

    # (0, 0) has two images: (0,0,0) and (1,0,0)
    val = bm[(0, 0)]
    expected = bm.pair_blocks["A-A"][0] + bm.pair_blocks["A-A"][1]
    assert torch.allclose(val, expected)

    # (0, 1) has two images
    val2 = bm[(0, 1)]
    expected2 = bm.pair_blocks["A-B"][0] + bm.pair_blocks["A-B"][1]
    assert torch.allclose(val2, expected2)


def test_diag(mock_block_matrix):
    bm = mock_block_matrix
    diag = bm.diag()

    # A-A should have 1 block (the first one, since atom count for A is 1)
    assert "A-A" in diag
    assert diag["A-A"].shape[0] == 1
    assert torch.allclose(diag["A-A"][0], bm.pair_blocks["A-A"][0])

    # A-B is hetero, should not be in diag
    assert "A-B" not in diag


def test_offdiag(mock_block_matrix):
    bm = mock_block_matrix
    off = bm.offdiag()

    # A-A should have the second block
    assert "A-A" in off
    assert off["A-A"].shape[0] == 1
    assert torch.allclose(off["A-A"][0], bm.pair_blocks["A-A"][1])

    # A-B is hetero, should have all blocks
    assert "A-B" in off
    assert off["A-B"].shape[0] == 2
    assert torch.allclose(off["A-B"], bm.pair_blocks["A-B"])


def test_transpose_symmetric():
    atoms = ("A",)
    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"A": "1s"})

    # Symmetric edges: (0,0,0,0,0) and pair (1,0,0,0,0) & (-1,0,0,0,0)
    edges = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]]).t()
    blocks = torch.randn(3, 1, 1)

    pair_blocks = {"A-A": blocks}
    pair_edges = {"A-A": edges}
    lookup = {tuple(e): ("A-A", i) for i, e in enumerate(edges.t().tolist())}

    bm = BlockMatrix(
        atoms, atom_counts, pair_blocks, pair_edges, lookup, orbital_cfg, "e3nn"
    )

    bm_t = bm.transpose()

    # Check A-A
    # Transpose of (1,0,0,0,0) is (-1,0,0,0,0).
    # Transpose of (-1,0,0,0,0) is (1,0,0,0,0).
    # The `transpose` method reorders the transposed blocks to match the original edge order.
    # Original order: 0:diag, 1:pos_shift, 2:neg_shift
    # Transposed raw: 0:diag, 1:neg_shift, 2:pos_shift
    # Reordered: 0:diag, 1:pos_shift (was index 2), 2:neg_shift (was index 1)

    # Block 1 in result should be Block 2 of input (transposed)
    assert torch.allclose(bm_t.pair_blocks["A-A"][1], blocks[2].transpose(-1, -2))
    # Block 2 in result should be Block 1 of input (transposed)
    assert torch.allclose(bm_t.pair_blocks["A-A"][2], blocks[1].transpose(-1, -2))

    alignment = build_trace_alignment_from_pair_edges(bm.pair_edges)
    bm_t_aligned = bm.transpose_aligned(alignment)
    assert torch.equal(bm_t_aligned.pair_edges["A-A"], bm_t.pair_edges["A-A"])
    assert torch.allclose(bm_t_aligned.pair_blocks["A-A"], bm_t.pair_blocks["A-A"])


def test_reorder_edges(mock_block_matrix):
    bm = mock_block_matrix
    # Swap the two edges of A-B
    perm = torch.tensor([1, 0])
    order_dict = {"A-B": perm}

    bm_new = bm.reorder_edges(order_dict)

    # Check edges
    expected_edges = bm.pair_edges["A-B"][:, perm]
    assert torch.equal(bm_new.pair_edges["A-B"], expected_edges)

    # Check blocks
    expected_blocks = bm.pair_blocks["A-B"][perm]
    assert torch.equal(bm_new.pair_blocks["A-B"], expected_blocks)

    # Check lookup update
    # Old: (0,0,0,0,1)->0, (0,1,0,0,1)->1
    # New: (0,1,0,0,1)->0, (0,0,0,0,1)->1
    assert bm_new.lookup[(0, 1, 0, 0, 1)] == ("A-B", 0)
    assert bm_new.lookup[(0, 0, 0, 0, 1)] == ("A-B", 1)


def test_align_with(mock_block_matrix):
    bm1 = mock_block_matrix
    # Create bm2 with swapped A-B edges
    perm = torch.tensor([1, 0])
    order_dict = {"A-B": perm}
    bm2 = bm1.reorder_edges(order_dict)

    # Align bm2 to bm1
    # bm2._align_with(bm1) -> returns (bm2_reordered, bm1)
    # bm2_reordered should match bm1 order

    bm2_aligned, bm1_out = bm2._align_with(bm1)

    assert torch.equal(bm2_aligned.pair_edges["A-B"], bm1.pair_edges["A-B"])
    # Blocks should be back to original order (values)
    # Note: bm2 blocks were permuted. bm2_aligned un-permutes them.
    # bm2 blocks: [b1, b0]. perm to match bm1 (which is [b0, b1]) is [1, 0].
    # bm2_aligned blocks: [b1, b0][1,0] = [b0, b1].
    assert torch.allclose(bm2_aligned.pair_blocks["A-B"], bm1.pair_blocks["A-B"])
