import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot_block import SnapshotBlockData, SnapshotIrrepsData


def make_mock_snapshot():
    atoms = ("H", "H", "O", "H", "H", "O")
    cfg = OrbitalIrrepConfig.from_dict(
        {
            "H": ["2x0e"],  # dim 2
            "O": ["1x0e", "1x1o"],  # dim 4
        }
    )
    mapper = BlockIrrepMapper(cfg)

    pair_blocks = {}
    pair_edges = {}
    lookup = {}

    # iterate all ordered pairs
    for i, el_i in enumerate(atoms):
        for j, el_j in enumerate(atoms):
            key = f"{el_i}-{el_j}"
            d_i, d_j = mapper.block_dims(key)
            blk = torch.randn(d_i, d_j)  # single block
            if key not in pair_blocks:
                pair_blocks[key] = []
                pair_edges[key] = []
            local_idx = len(pair_blocks[key])
            pair_blocks[key].append(blk)
            pair_edges[key].append([i, j])
            lookup[(i, j)] = (key, local_idx)

    # stack per key
    pair_blocks = {k: torch.stack(v) for k, v in pair_blocks.items()}
    pair_edges = {
        k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
    }

    return SnapshotBlockData(atoms, pair_blocks, pair_edges, lookup, mapper)


def test_roundtrip_blocks_vectors():
    snap_blk = make_mock_snapshot()
    snap_vec = snap_blk.to_vectors()
    snap_reco = snap_vec.to_blocks()

    # test one random global pair
    assert torch.allclose(snap_blk[(2, 5)], snap_reco[(2, 5)], atol=1e-6)

    # test full tensors key "O-O"
    assert torch.allclose(snap_blk["O-O"], snap_reco["O-O"], atol=1e-6)


def test_dense_roundtrip():
    snap = make_mock_snapshot()
    dense = snap.to_dense()
    re_snap = SnapshotBlockData.from_dense(
        dense,
        snap.mapper.orbital_cfg,
        snap.atoms,
    )
    # check one arbitrary oriented pair
    assert torch.allclose(snap[(0, 3)], re_snap[(0, 3)], atol=1e-6)


def test_dense_to_sparse_roundtrip_full_system():
    atoms = ("H", "H", "O", "H", "H", "O")
    cfg = OrbitalIrrepConfig.from_dict(
        {
            "H": ["2x0e"],  # dim 2
            "O": ["1x0e", "1x1o"],  # dim 4
        }
    )
    # total dim = 2+2+4+2+2+4 = 16
    dense = torch.randn(16, 16)
    snap = SnapshotBlockData.from_dense(dense, cfg, atoms)
    dense_back = snap.to_dense()
    assert torch.allclose(dense, dense_back, atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    snap = make_mock_snapshot()
    file = tmp_path / "snap.pt"
    snap.save(file)

    snap_loaded = SnapshotBlockData.load(file)
    # compare two random edges
    assert torch.allclose(snap[(0, 2)], snap_loaded[(0, 2)], atol=1e-6)
    assert torch.allclose(snap["H-O"], snap_loaded["H-O"])


def test_irreps_save_load(tmp_path):
    snap_blk = make_mock_snapshot()
    snap_vec = snap_blk.to_vectors()

    file = tmp_path / "snap_vec.pt"
    snap_vec.save(file)

    snap_vec_loaded = SnapshotIrrepsData.load(file)

    # compare a random O‑H oriented edge
    assert torch.allclose(snap_vec[(2, 0)], snap_vec_loaded[(2, 0)], atol=1e-6)

    # convert back to blocks and verify against original blocks
    snap_blk_reco = snap_vec_loaded.to_blocks()
    assert torch.allclose(snap_blk["O-H"], snap_blk_reco["O-H"], atol=1e-6)
