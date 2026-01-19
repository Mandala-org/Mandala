import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix, IrrepsBlockData
from core.basis_converter import OpenMXE3NNConverter


def make_mock_matrix():
    atoms = ("H", "H", "O", "H", "H", "O")
    from collections import Counter

    atom_counts = Counter(atoms)
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
            pair_edges[key].append([0, 0, 0, i, j])
            lookup[(0, 0, 0, i, j)] = (key, local_idx)

    # stack per key
    pair_blocks = {k: torch.stack(v) for k, v in pair_blocks.items()}
    pair_edges = {
        k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
    }

    return (
        BlockMatrix(atoms, atom_counts, pair_blocks, pair_edges, lookup, cfg, "openmx"),
        mapper,
    )


@pytest.mark.integration
def test_roundtrip_blocks_vectors():
    snap_blk, mapper = make_mock_matrix()
    snap_vec = snap_blk.to_vectors(mapper)
    snap_reco = snap_vec.to_blocks(mapper)

    # test one random global pair
    assert torch.allclose(snap_blk[(2, 5)], snap_reco[(2, 5)], atol=1e-6)

    # test full tensors key "O-O"
    assert torch.allclose(snap_blk["O-O"], snap_reco["O-O"], atol=1e-6)


@pytest.mark.integration
def test_denseify_roundtrip():
    matrix, mapper = make_mock_matrix()
    dense = matrix.to_dense()
    re_matrix = BlockMatrix.from_dense(
        dense,
        mapper.orbital_cfg,
        matrix.atoms,
        basis=matrix.basis,
    )
    # check one arbitrary oriented pair
    assert torch.allclose(matrix[(0, 3)], re_matrix[(0, 3)], atol=1e-6)


@pytest.mark.integration
def test_sparsify_roundtrip():
    atoms = ("H", "H", "O", "H", "H", "O")
    cfg = OrbitalIrrepConfig.from_dict(
        {
            "H": ["2x0e"],  # dim 2
            "O": ["1x0e", "1x1o"],  # dim 4
        }
    )
    # total dim = 2+2+4+2+2+4 = 16
    dense = torch.randn(16, 16)
    matrix = BlockMatrix.from_dense(dense, cfg, atoms, basis="openmx")
    dense_back = matrix.to_dense()
    assert torch.allclose(dense, dense_back, atol=1e-6)


@pytest.mark.integration
def test_save_load_roundtrip(tmp_path):
    matrix, mapper = make_mock_matrix()
    file = tmp_path / "matrix.pt"
    matrix.save(file)

    matrix_loaded = BlockMatrix.load(file)
    assert matrix_loaded.basis == "openmx"

    # compare two random edges
    assert torch.allclose(matrix[(0, 2)], matrix_loaded[(0, 2)], atol=1e-6)
    assert torch.allclose(matrix["H-O"], matrix_loaded["H-O"])


@pytest.mark.integration
def test_irreps_save_load(tmp_path):
    matrix_blk, mapper = make_mock_matrix()
    matrix_vec = matrix_blk.to_vectors(mapper)

    file = tmp_path / "matrix_vec.pt"
    matrix_vec.save(file)

    matrix_vec_loaded = IrrepsBlockData.load(file)

    # compare a random O‑H oriented edge
    assert torch.allclose(matrix_vec[(2, 0)], matrix_vec_loaded[(2, 0)], atol=1e-6)

    # convert back to blocks and verify against original blocks
    matrix_blk_reco = matrix_vec_loaded.to_blocks(mapper)
    assert torch.allclose(matrix_blk["O-H"], matrix_blk_reco["O-H"], atol=1e-6)


@pytest.mark.integration
def test_basis_converter_roundtrip():
    matrix_open, mapper = make_mock_matrix()  # default basis="openmx"
    conv = OpenMXE3NNConverter(mapper.orbital_cfg)

    matrix_e3 = conv.matrix_to_e3nn(matrix_open)
    assert matrix_e3.basis == "e3nn"

    # shape sanity check on one key
    key = "H-O"
    assert matrix_e3[key].shape == matrix_open[key].shape

    matrix_back = conv.matrix_to_openmx(matrix_e3)
    assert matrix_back.basis == "openmx"

    # element‑wise equality for all blocks
    for idx in matrix_open.lookup:
        assert torch.allclose(matrix_open[idx], matrix_back[idx], atol=1e-6)


@pytest.mark.integration
def test_block_converter_direct():
    matrix, mapper = make_mock_matrix()
    conv = OpenMXE3NNConverter(mapper.orbital_cfg)

    key = "O-H"
    blk = matrix[key][0]  # (d_O, d_H)
    blk_e3 = conv.block_openmx_to_e3nn(key, blk)
    blk_open = conv.block_e3nn_to_openmx(key, blk_e3)
    assert torch.allclose(blk, blk_open, atol=1e-6)


# --------------------------------------------------------------------------- #
#  arithmetic tests
# --------------------------------------------------------------------------- #


def _shuffled_matrix():
    """Return a matrix whose edges are randomly permuted inside every key."""
    matrix, mapper = make_mock_matrix()
    import torch

    order = {}
    for k, edges in matrix.pair_edges.items():
        idx = torch.randperm(edges.shape[1])
        order[k] = idx
    return matrix.reorder_edges(order)


@pytest.mark.integration
def test_addition_commutes():
    A, mapper = make_mock_matrix()
    B = _shuffled_matrix()

    C1 = A + B
    C2 = B + A
    assert torch.allclose(C1.to_dense(), C2.to_dense(), atol=1e-6)


@pytest.mark.integration
def test_subtraction_vs_dense():
    A, mapper = make_mock_matrix()
    B = _shuffled_matrix()
    C = A - B

    dense_C = A.to_dense() - B.to_dense()
    assert torch.allclose(C.to_dense(), dense_C, atol=1e-6)


@pytest.mark.integration
def test_sum_builtin():
    A, mapper = make_mock_matrix()
    B = _shuffled_matrix()
    total = sum([A, B])  # relies on __radd__ with 0
    assert torch.allclose(total.to_dense(), A.to_dense() + B.to_dense(), atol=1e-6)
