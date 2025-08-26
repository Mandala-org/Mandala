import pytest
from pathlib import Path
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout
from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix


def _load_snapshot():
    cfg = OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})
    sample = Path("./data/small/H2O/original/H2O.matrix")
    atoms = list("HHHHOO")
    snap = parse_openmx_scfout(sample, atoms, cfg, convention="openmx")
    return snap.canonicalize_edges()


@pytest.mark.integration
def test_energy_and_electron_count():
    snap = _load_snapshot()
    E = snap.get_energy()
    Ne = snap.get_number_of_electrons()

    # reference using sparse helper directly
    E_ref = trace_matmul_sparse_snap_vectorized(snap.hamiltonian, snap.density)
    Ne_ref = trace_matmul_sparse_snap_vectorized(snap.density, snap.overlap)

    assert torch.allclose(E, E_ref, atol=1e-6)
    assert torch.allclose(Ne, Ne_ref, atol=1e-6)


@pytest.mark.integration
def test_canonical_edge_ordering():
    snap = _load_snapshot()
    D = snap.density

    for key in D.keys():
        edges = D.pair_edges[key]
        is_diag = edges[0] == edges[1]

        # Find the first diagonal edge, if any
        diag_indices = torch.where(is_diag)[0]
        if len(diag_indices) > 0:
            first_diag_idx = diag_indices[0]
            # All edges before the first diagonal edge must be off-diagonal
            assert not torch.any(is_diag[:first_diag_idx])
            # All edges from the first diagonal edge onwards must be diagonal
            assert torch.all(is_diag[first_diag_idx:])


@pytest.mark.integration
def test_save_load_roundtrip(tmp_path):
    snap = _load_snapshot()
    file = tmp_path / "snapshot.pt"
    snap.save(file)

    snap2 = Snapshot.load(file)
    assert torch.allclose(snap2.get_energy(), snap.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap2.get_number_of_electrons(), snap.get_number_of_electrons(), atol=1e-6
    )


# ---------------------------------------------------------------- basis conversion


@pytest.mark.integration
def test_basis_conversion_roundtrip():
    snap_open = _load_snapshot()  # native OpenMX basis
    snap_e3 = snap_open.to_e3nn()

    # sanity checks
    assert snap_e3.density.basis == "e3nn"
    assert snap_e3.to_e3nn() is snap_e3  # idempotent

    # back-conversion
    snap_back = snap_e3.to_openmx()
    assert snap_back.density.basis == "openmx"

    # numerical invariants -------------------------------------------------
    # choose one representative block (H-O first edge)
    key = "H-O"
    assert torch.allclose(
        snap_back.density[key][0], snap_open.density[key][0], atol=1e-6
    )

    # physics helpers unchanged
    assert torch.allclose(snap_back.get_energy(), snap_open.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap_back.get_number_of_electrons(),
        snap_open.get_number_of_electrons(),
        atol=1e-6,
    )
    assert torch.allclose(snap_e3.get_energy(), snap_open.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap_e3.get_number_of_electrons(),
        snap_open.get_number_of_electrons(),
        atol=1e-6,
    )

# tests for selecting snapshot targets
def _create_dummy_block_matrix():
    """Helper to create a minimal BlockMatrix for testing."""
    # The contents don't matter, only that it's a BlockMatrix instance
    return BlockMatrix(
        atoms=("H",),
        atom_counts={"H": 1},
        pair_blocks={"H-H": torch.randn(1, 1, 1)},
        pair_edges={"H-H": torch.tensor([[0], [0]])},
        lookup={(0, 0): ("H-H", 0)},
        orbital_cfg=OrbitalIrrepConfig.from_dict({"H": "1x0e"}),
        basis="e3nn",
    )


@pytest.mark.unit
def test_partial_snapshot_creation():
    """Tests that a Snapshot can be created with only a Hamiltonian."""
    dummy_hamiltonian = _create_dummy_block_matrix()

    # Create a snapshot with only the hamiltonian
    snap = Snapshot(hamiltonian=dummy_hamiltonian)

    assert snap.hamiltonian is not None
    assert snap.overlap is None
    assert snap.density is None


@pytest.mark.unit
def test_get_energy_raises_error_on_partial_snapshot():
    """
    Tests that get_energy() raises a RuntimeError if density is missing.
    """
    dummy_hamiltonian = _create_dummy_block_matrix()
    snap = Snapshot(hamiltonian=dummy_hamiltonian) # Missing density matrix

    with pytest.raises(RuntimeError, match="Cannot compute energy"):
        snap.get_energy()

@pytest.mark.unit
def test_get_electrons_raises_error_on_partial_snapshot():
    """
    Tests that get_number_of_electrons() raises a RuntimeError if overlap is missing.
    """
    dummy_density = _create_dummy_block_matrix()
    snap = Snapshot(density=dummy_density) # Missing overlap matrix

    with pytest.raises(RuntimeError, match="Cannot compute number of electrons"):
        snap.get_number_of_electrons()


