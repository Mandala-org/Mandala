import pytest
import torch
from collections import Counter
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig


@pytest.fixture
def mock_snapshot():
    # Create a simple snapshot with one BlockMatrix (density)
    atoms = ("H", "H")
    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

    # Edges:
    # 1. (0,0,0, 0, 1) - Off-diag
    # 2. (1,0,0, 0, 1) - Off-diag, shifted
    edges = torch.tensor([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1]]).t()
    blocks = torch.randn(2, 1, 1)

    pair_blocks = {"H-H": blocks}
    pair_edges = {"H-H": edges}
    lookup = {tuple(e): ("H-H", i) for i, e in enumerate(edges.t().tolist())}

    bm = BlockMatrix(
        atoms, atom_counts, pair_blocks, pair_edges, lookup, orbital_cfg, "e3nn"
    )

    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    box = torch.eye(3) * 10.0

    return Snapshot(bm, bm, bm, positions=positions, box=box)


def test_matrix_from_payload(mock_snapshot):
    snap = mock_snapshot
    # Use BlockMatrix._to_payload to generate payload
    payload = snap.density._to_payload()

    # Reconstruct
    bm_new = Snapshot._matrix_from_payload(payload)

    # Check lookup reconstruction
    assert (0, 0, 0, 0, 1) in bm_new.lookup
    assert (1, 0, 0, 0, 1) in bm_new.lookup
    assert bm_new.lookup[(0, 0, 0, 0, 1)] == ("H-H", 0)
    assert bm_new.lookup[(1, 0, 0, 0, 1)] == ("H-H", 1)

    assert torch.allclose(bm_new.pair_blocks["H-H"], snap.density.pair_blocks["H-H"])


def test_edge_displacements(mock_snapshot):
    snap = mock_snapshot
    # Edges:
    # 1. (0,0,0, 0, 1) -> delta = pos[1] - pos[0] + shift@box = (2,0,0) - (0,0,0) + 0 = (2,0,0)
    # 2. (1,0,0, 0, 1) -> delta = (2,0,0) + (10,0,0) = (12,0,0)

    disps = snap._edge_displacements(snap.density)

    assert "H-H" in disps
    d = disps["H-H"]
    assert d.shape == (2, 3)

    assert torch.allclose(d[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(d[1], torch.tensor([12.0, 0.0, 0.0]))


def test_change_basis_geometry(mock_snapshot):
    snap = mock_snapshot
    # Current basis is e3nn (default in mock)

    # Mock the converter to avoid needing real orbital configs logic or just ignore matrix conversion
    # But _change_basis calls OpenMXE3NNConverter.
    # We can just check the geometry part if we can bypass matrix conversion or if matrix conversion works with 1s.
    # 1s is scalar, so conversion is trivial (identity or scalar factor).

    snap_openmx = snap._change_basis("openmx")

    # Check positions transformation
    # e3nn -> openmx: pos = pos @ [[1,2,0]]
    # pos was [[0,0,0], [2,0,0]]
    # [2,0,0] @ [[0,0,1],[1,0,0],[0,1,0]] (permutation matrix for [1,2,0] indices?)
    # Wait, `pos @ torch.eye(3)[[1, 2, 0]]`
    # eye:
    # 1 0 0
    # 0 1 0
    # 0 0 1
    # [[1, 2, 0]]:
    # 0 1 0 (row 1)
    # 0 0 1 (row 2)
    # 1 0 0 (row 0)
    # So P = [[0,1,0], [0,0,1], [1,0,0]]
    # [x, y, z] @ P = [z, x, y]

    # Original pos[1] = [2, 0, 0] (x=2)
    # New pos[1] should be [0, 2, 0] (y=2)

    assert torch.allclose(snap_openmx.positions[1], torch.tensor([0.0, 2.0, 0.0]))


def test_export_to_deephe3(mock_snapshot, tmp_path):
    snap = mock_snapshot

    # Mock info object which is needed for export
    class MockInfo:
        fermi_level = torch.tensor(0.0)

    snap.info = MockInfo()

    out_dir = tmp_path / "deephe3"
    snap.export_to_deephe3(out_dir)

    assert (out_dir / "hamiltonians.h5").exists()

    import h5py

    with h5py.File(out_dir / "hamiltonians.h5", "r") as f:
        # Keys should be string of list [sx, sy, sz, src, dst]
        key1 = str([0, 0, 0, 0, 1])
        key2 = str([1, 0, 0, 0, 1])

        assert key1 in f
        assert key2 in f

        assert f[key1].shape == (1, 1)
