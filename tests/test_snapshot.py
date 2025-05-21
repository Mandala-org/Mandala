from pathlib import Path
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout
from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.snapshot import Snapshot


def _load_snapshot():
    cfg = OrbitalIrrepConfig.from_dict({"H": "2s1p", "O": "3s2p"})
    sample = Path("./data/small/H2O/H2O_original.out")
    atoms = list("HHHHOO")
    return parse_openmx_scfout(sample, atoms, cfg, convention="openmx")


def test_energy_and_electron_count():
    snap = _load_snapshot()
    E = snap.get_energy()
    Ne = snap.get_number_of_electrons()

    # reference using sparse helper directly
    E_ref = trace_matmul_sparse_snap_vectorized(snap.hamiltonian, snap.density)
    Ne_ref = trace_matmul_sparse_snap_vectorized(snap.density, snap.overlap)

    assert torch.allclose(E, E_ref, atol=1e-6)
    assert torch.allclose(Ne, Ne_ref, atol=1e-6)


def test_density_edge_ordering():
    snap = _load_snapshot()
    D = snap.density

    for key in D.keys():
        norms = D[key].pow(2).sum((-2, -1)).sqrt()  # (E,)
        assert torch.all(
            norms[1:] >= norms[:-1]
        ), f"edge blocks for {key} not ascending by L2 norm"


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
