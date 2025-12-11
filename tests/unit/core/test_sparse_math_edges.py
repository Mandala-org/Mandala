import pytest
import torch
from collections import Counter

from core.sparse_math import (
    trace_matmul_sparse,
    trace_matmul_sparse_block_matrix,
    trace_matmul_sparse_snap_vectorized,
)
from data.block_matrix import BlockMatrix

# -----------------------------------------------------------------------------
# Fixtures & Mocks
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_orbital_config():
    # Simple config: H -> 1s (dim 1), C -> 1s1p (dim 1+3=4)
    # But for simplicity, let's just use dims.
    # We'll mock the config object to return specific dims.
    class MockConfig:
        def block_dims(self, key):
            # key is "A-B"
            el_a, el_b = key.split("-")
            d = {"H": 1, "C": 2}  # simple dims
            return d[el_a], d[el_b]

        def to_dict(self):
            return {}

    return MockConfig()


def create_mock_block_matrix(atoms, edges_data, orbital_cfg):
    """
    atoms: list of element strings
    edges_data: list of dicts with keys:
        'src': int
        'dst': int
        'shift': tuple (sx, sy, sz)
        'block': torch.Tensor
    """
    atom_counts = Counter(atoms)

    pair_blocks = {}
    pair_edges = {}
    lookup = {}

    # Group by pair key
    grouped = {}
    for item in edges_data:
        src, dst = item["src"], item["dst"]
        el_src, el_dst = atoms[src], atoms[dst]
        key = f"{el_src}-{el_dst}"
        grouped.setdefault(key, []).append(item)

    for key, items in grouped.items():
        # Stack blocks
        blocks = torch.stack([item["block"] for item in items])
        pair_blocks[key] = blocks

        # Stack edges (sx, sy, sz, src, dst)
        edge_list = []
        for idx, item in enumerate(items):
            sx, sy, sz = item["shift"]
            src, dst = item["src"], item["dst"]
            edge_list.append([sx, sy, sz, src, dst])

            # Update lookup
            # lookup key is (sx, sy, sz, src, dst)
            lookup[(sx, sy, sz, src, dst)] = (key, idx)

        pair_edges[key] = torch.tensor(edge_list, dtype=torch.long).t()  # (5, E)

    return BlockMatrix(
        atoms=tuple(atoms),
        atom_counts=atom_counts,
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis="e3nn",
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_matmul_sparse_raw():
    """
    Test the raw tensor version: trace_matmul_sparse(blocks_a, blocks_b, edge_index)
    """
    # Setup: 2 edges
    # Edge 0: (0, 0, 0, 0, 1) -> A_01
    # Edge 1: (0, 0, 0, 1, 0) -> A_10 (symmetric to Edge 0)

    # A matrices
    A0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # dim 2x2
    A1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # dim 2x2
    blocks_a = torch.stack([A0, A1])

    # B matrices
    B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity
    B1 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # 2 * Identity
    blocks_b = torch.stack([B0, B1])

    # Edge indices (sx, sy, sz, src, dst)
    # Let's say edge 0 is 0->1 (shift 0)
    # Let's say edge 1 is 1->0 (shift 0)
    # Note: trace_matmul_sparse iterates over A edges.
    # For A_k at edge (sx, sy, sz, i, j), it looks for B at (-sx, -sy, -sz, j, i).

    edge_index = torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], dtype=torch.long)

    # Expected calculation:
    # k=0: edge (0,0,0,0,1). Rev is (0,0,0,1,0). This is index 1 in edge_index.
    #      Trace(A[0] @ B[1]) = Trace(A0 @ B1)
    #      A0 @ B1 = [[1,2],[3,4]] @ [[2,0],[0,2]] = [[2,4],[6,8]]. Trace = 10.

    # k=1: edge (0,0,0,1,0). Rev is (0,0,0,0,1). This is index 0 in edge_index.
    #      Trace(A[1] @ B[0]) = Trace(A1 @ B0)
    #      A1 @ B0 = [[5,6],[7,8]] @ [[1,0],[0,1]] = [[5,6],[7,8]]. Trace = 13.

    # Total = 10 + 13 = 23.

    res = trace_matmul_sparse(blocks_a, blocks_b, edge_index)
    assert torch.isclose(res, torch.tensor(23.0))


@pytest.mark.unit
def test_trace_matmul_sparse_pbc():
    """
    Test raw tensor version with PBC shifts.
    """
    # Edge 0: 0->1 with shift (1, 0, 0)
    # Edge 1: 1->0 with shift (-1, 0, 0)

    A0 = torch.eye(2)
    A1 = torch.eye(2) * 2
    blocks_a = torch.stack([A0, A1])
    blocks_b = torch.stack([A0, A1])  # Same blocks

    edge_index = torch.tensor([[1, 0, 0, 0, 1], [-1, 0, 0, 1, 0]], dtype=torch.long)

    # k=0: edge (1,0,0,0,1). Rev (-1,0,0,1,0) -> index 1.
    #      Trace(A[0] @ B[1]) = Trace(I @ 2I) = Trace(2I) = 4.

    # k=1: edge (-1,0,0,1,0). Rev (1,0,0,0,1) -> index 0.
    #      Trace(A[1] @ B[0]) = Trace(2I @ I) = Trace(2I) = 4.

    # Total = 8.

    res = trace_matmul_sparse(blocks_a, blocks_b, edge_index)
    assert torch.isclose(res, torch.tensor(8.0))


@pytest.mark.unit
def test_trace_matmul_sparse_block_matrix(mock_orbital_config):
    """
    Test trace_matmul_sparse_block_matrix with BlockMatrix objects.
    """
    atoms = ["H", "H"]  # 0, 1

    # A: 0->1 (shift 0) = A_01
    #    1->0 (shift 0) = A_10
    A_01 = torch.tensor([[1.0]])
    A_10 = torch.tensor([[2.0]])

    edges_A = [
        {"src": 0, "dst": 1, "shift": (0, 0, 0), "block": A_01},
        {"src": 1, "dst": 0, "shift": (0, 0, 0), "block": A_10},
    ]

    # B: 0->1 (shift 0) = B_01
    #    1->0 (shift 0) = B_10
    B_01 = torch.tensor([[3.0]])
    B_10 = torch.tensor([[4.0]])

    edges_B = [
        {"src": 0, "dst": 1, "shift": (0, 0, 0), "block": B_01},
        {"src": 1, "dst": 0, "shift": (0, 0, 0), "block": B_10},
    ]

    BM_A = create_mock_block_matrix(atoms, edges_A, mock_orbital_config)
    BM_B = create_mock_block_matrix(atoms, edges_B, mock_orbital_config)

    # Calculation:
    # Loop over A edges:
    # 1. (0,0,0,0,1) -> A_01. Rev in B is (0,0,0,1,0) -> B_10.
    #    Trace(A_01 @ B_10) = 1.0 * 4.0 = 4.0
    # 2. (0,0,0,1,0) -> A_10. Rev in B is (0,0,0,0,1) -> B_01.
    #    Trace(A_10 @ B_01) = 2.0 * 3.0 = 6.0
    # Total = 10.0

    res = trace_matmul_sparse_block_matrix(BM_A, BM_B)
    assert torch.isclose(res, torch.tensor(10.0))


@pytest.mark.unit
def test_trace_matmul_sparse_snap_vectorized(mock_orbital_config):
    """
    Test trace_matmul_sparse_snap_vectorized with BlockMatrix objects.
    Should give same result as loop version.
    """
    atoms = ["H", "H"]

    # A: 0->1 (shift 0) = A_01
    #    1->0 (shift 0) = A_10
    A_01 = torch.tensor([[1.0]])
    A_10 = torch.tensor([[2.0]])

    edges_A = [
        {"src": 0, "dst": 1, "shift": (0, 0, 0), "block": A_01},
        {"src": 1, "dst": 0, "shift": (0, 0, 0), "block": A_10},
    ]

    # B: 0->1 (shift 0) = B_01
    #    1->0 (shift 0) = B_10
    B_01 = torch.tensor([[3.0]])
    B_10 = torch.tensor([[4.0]])

    edges_B = [
        {"src": 0, "dst": 1, "shift": (0, 0, 0), "block": B_01},
        {"src": 1, "dst": 0, "shift": (0, 0, 0), "block": B_10},
    ]

    BM_A = create_mock_block_matrix(atoms, edges_A, mock_orbital_config)
    BM_B = create_mock_block_matrix(atoms, edges_B, mock_orbital_config)

    res_vec = trace_matmul_sparse_snap_vectorized(BM_A, BM_B)
    res_loop = trace_matmul_sparse_block_matrix(BM_A, BM_B)

    assert torch.isclose(res_vec, res_loop)
    assert torch.isclose(res_vec, torch.tensor(10.0))


@pytest.mark.unit
def test_trace_matmul_missing_edges(mock_orbital_config):
    """
    Test behavior when edges are missing in B (should raise ValueError).
    """
    atoms = ["H", "H"]

    # A has 0->1
    edges_A = [{"src": 0, "dst": 1, "shift": (0, 0, 0), "block": torch.tensor([[1.0]])}]

    # B has nothing
    edges_B = []

    BM_A = create_mock_block_matrix(atoms, edges_A, mock_orbital_config)
    BM_B = create_mock_block_matrix(atoms, edges_B, mock_orbital_config)

    with pytest.raises(ValueError, match="missing in second matrix"):
        trace_matmul_sparse_block_matrix(BM_A, BM_B)

    with pytest.raises(ValueError, match="missing in second matrix"):
        trace_matmul_sparse_snap_vectorized(BM_A, BM_B)


@pytest.mark.unit
def test_trace_matmul_pbc_mismatch(mock_orbital_config):
    """
    Test that edges with different shifts are not matched and raise ValueError.
    """
    atoms = ["H", "H"]

    # A: 0->1 shift (0,0,0)
    edges_A = [{"src": 0, "dst": 1, "shift": (0, 0, 0), "block": torch.tensor([[1.0]])}]

    # B: 1->0 shift (1,0,0)  <-- This is NOT the reverse of A's edge.
    # Reverse of A's edge would be 1->0 shift (0,0,0).
    edges_B = [{"src": 1, "dst": 0, "shift": (1, 0, 0), "block": torch.tensor([[1.0]])}]

    BM_A = create_mock_block_matrix(atoms, edges_A, mock_orbital_config)
    BM_B = create_mock_block_matrix(atoms, edges_B, mock_orbital_config)

    with pytest.raises(ValueError, match="missing in second matrix"):
        trace_matmul_sparse_block_matrix(BM_A, BM_B)

    with pytest.raises(ValueError, match="missing in second matrix"):
        trace_matmul_sparse_snap_vectorized(BM_A, BM_B)
