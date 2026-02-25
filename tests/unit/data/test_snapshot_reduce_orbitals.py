import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot


def _make_snapshot(atoms, orbital_dict):
    orbital_cfg = OrbitalIrrepConfig.from_dict(orbital_dict)
    total_dim = sum(orbital_cfg.block_dims(f"{el}-{el}")[0] for el in atoms)
    dense = torch.arange(total_dim * total_dim, dtype=torch.float32).reshape(
        total_dim, total_dim
    )
    bm = BlockMatrix.from_dense(dense, orbital_cfg, atoms, basis="e3nn")
    snap = Snapshot(bm, bm, bm)
    return snap, dense


@pytest.mark.unit
def test_reduce_orbitals_prefix_order_selection():
    snap, dense = _make_snapshot(("X",), {"X": "3s2p"})

    reduced = snap.reduce_orbitals("1s1p")

    # 3s2p layout is [s0, s1, s2, p0(3), p1(3)]; keep first s and first p set.
    keep = torch.tensor([0, 3, 4, 5], dtype=torch.long)
    expected = dense.index_select(0, keep).index_select(1, keep)

    assert reduced.hamiltonian.to_dense().shape == (4, 4)
    assert torch.allclose(reduced.hamiltonian.to_dense(), expected)
    assert reduced.hamiltonian.orbital_cfg.block_dims("X-X") == (4, 4)


@pytest.mark.unit
def test_reduce_orbitals_per_element_default_keeps_unspecified():
    snap, _ = _make_snapshot(("H", "O"), {"H": "2s2p", "O": "2s1p"})

    reduced = snap.reduce_orbitals({"H": "1s1p"})

    # H reduced: 2s2p -> 1s1p => dim 4. O unspecified and kept => dim 5.
    assert reduced.hamiltonian.orbital_cfg.block_dims("H-H") == (4, 4)
    assert reduced.hamiltonian.orbital_cfg.block_dims("O-O") == (5, 5)
    assert reduced.hamiltonian.pair_blocks["H-O"].shape[1:] == (4, 5)
    assert reduced.hamiltonian.pair_blocks["O-H"].shape[1:] == (5, 4)


@pytest.mark.unit
def test_reduce_orbitals_strict_raises_when_request_exceeds_available():
    snap, _ = _make_snapshot(("X",), {"X": "3s2p"})

    with pytest.raises(ValueError, match="does not have enough requested orbitals"):
        snap.reduce_orbitals("4s")
