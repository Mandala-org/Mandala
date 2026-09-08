import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.snapshot import Snapshot
from core.block_irrep_mapper import BlockIrrepMapper
from data.openmx_parser import parse_openmx_scfout


def test_dataset_factory_fixture_restores_the_real_openmx_loader(factory_results):
    assert Snapshot.from_openmx.__module__ == "data.snapshot"


@pytest.mark.parametrize("convention", ["openmx", "e3nn"])
def test_density_symmetrization_averages_periodic_reverse_pairs(tmp_path, convention):
    # Include unequal reverse-image blocks and an asymmetric onsite block.
    path = tmp_path / "HS.out"
    blocks = {
        0: [[1.0, 2.0], [4.0, 3.0]],
        1: [[5.0, 6.0], [7.0, 8.0]],
        2: [[9.0, 10.0], [11.0, 12.0]],
    }
    lines = []
    for section in (
        "Kohn-Sham Hamiltonian spin=0",
        "Overlap matrix",
        "Density matrix spin=0",
    ):
        lines.append(section)
        for rn, block in blocks.items():
            lines.append(f"global index=1 local index=0 (global=1, Rn={rn})")
            lines.extend(" ".join(map(str, row)) for row in block)
    lines.append("Overlap matrix with position operator x")
    for rn, shift in [(0, 0), (1, 1), (2, -1)]:
        lines.append(f"global index=1 local index=0 (global=1, Rn={rn} {shift} 0 0)")
    path.write_text("\n".join(lines) + "\n")
    cfg = OrbitalIrrepConfig.from_dict({"H": "2s"})
    raw = parse_openmx_scfout(
        path, ["H"], cfg, convention=convention, symmetrize_density=False
    )
    sym = parse_openmx_scfout(path, ["H"], cfg, convention=convention)
    expected = 0.5 * (raw.density + raw.density.transpose())
    torch.testing.assert_close(sym.density["H-H"], expected["H-H"])
    assert sym.density.lookup == raw.density.lookup
    torch.testing.assert_close(sym.hamiltonian["H-H"], raw.hamiltonian["H-H"])
    torch.testing.assert_close(sym.overlap["H-H"], raw.overlap["H-H"])
    twice = 0.5 * (sym.density + sym.density.transpose())
    torch.testing.assert_close(twice["H-H"], sym.density["H-H"])


@pytest.mark.unit
def test_parse_returns_snapshot(h2o_snapshot):
    snap = h2o_snapshot
    assert isinstance(snap, Snapshot)

    ham = snap.hamiltonian
    den = snap.density
    assert ham["H-O"].shape[-2:] == (9, 22)
    den_OH = den["O-H"]
    den_T = den.transpose()
    den_OH_T = den_T["O-H"]
    assert torch.allclose(
        den_OH, den_OH_T, atol=1e-5
    ), "D[O-H] should be the same as D[H-O].T"


@pytest.mark.unit
def test_parse(h2o_orbital_cfg: OrbitalIrrepConfig, h2o_snapshot):
    mats = h2o_snapshot
    assert mats["hamiltonian"] is not None
    assert mats["overlap"] is not None
    assert mats["density"] is not None
    assert mats.hamiltonian is not None
    assert mats.overlap is not None
    assert mats.density is not None

    hamiltonian = mats["hamiltonian"]
    overlap = mats["overlap"]
    density = mats["density"]
    # there should be at least one H‑O block
    assert "H-O" in hamiltonian.keys()
    E_HO, d_H, d_O = hamiltonian["H-O"].shape
    assert d_H == 9 and d_O == 22

    mapper = BlockIrrepMapper(h2o_orbital_cfg)

    # round‑trip vector check
    snap_vec = hamiltonian.to_vectors(mapper)
    assert snap_vec["H-O"].shape[-1] == mapper.vector_dim("H-O")

    assert hamiltonian[0, 0].shape == (9, 9)
    assert hamiltonian[1, 4].shape == (9, 22)
    assert hamiltonian[5, 3].shape == (22, 9)
    assert hamiltonian[4, 5].shape == (22, 22)

    assert overlap[1, 3].shape == (9, 9)
    assert overlap[2, 5].shape == (9, 22)
    assert overlap[4, 0].shape == (22, 9)
    assert overlap[5, 4].shape == (22, 22)

    assert density[2, 3].shape == (9, 9)
    assert density[3, 4].shape == (9, 22)
    assert density[4, 3].shape == (22, 9)
    assert density[5, 5].shape == (22, 22)


@pytest.mark.unit
def test_parse_pbc_shapes(h2o_snapshot):
    mats = h2o_snapshot
    density = mats["density"]
    # ensure that duplicate Rn blocks were not summed: count of H‑H edges is 16 (fully connected dir graph)
    assert density["H-H"].shape == (712, 9, 9)
    assert density["O-O"].shape == (194, 22, 22)
    assert density["H-O"].shape == (384, 9, 22)
    assert density["O-H"].shape == (384, 22, 9)
    # assert O-H is the same as H-O.T
    density = density
    density_T = density.transpose()
    assert torch.allclose(
        density["O-H"], density_T["O-H"], atol=1e-5
    ), "D[O-H] should be the same as D[H-O].T"

    hamiltonian = mats["hamiltonian"]
    assert hamiltonian["H-H"].shape == (712, 9, 9)
    assert hamiltonian["O-O"].shape == (194, 22, 22)
    assert hamiltonian["H-O"].shape == (384, 9, 22)
    assert hamiltonian["O-H"].shape == (384, 22, 9)
    # assert O-H is the same as H-O.T
    hamiltonian = hamiltonian
    hamiltonian_T = hamiltonian.transpose()
    assert torch.allclose(
        hamiltonian["O-H"], hamiltonian_T["O-H"], atol=1e-5
    ), "Ham[O-H] should be the same as Ham[H-O].T"

    overlap = mats["overlap"]
    assert overlap["H-H"].shape == (712, 9, 9)
    assert overlap["O-O"].shape == (194, 22, 22)
    assert overlap["H-O"].shape == (384, 9, 22)
    assert overlap["O-H"].shape == (384, 22, 9)
    # assert O-H is the same as H-O.T
    overlap = overlap
    overlap_T = overlap.transpose()
    assert torch.allclose(
        overlap["O-H"], overlap_T["O-H"], atol=1e-5
    ), "Overlap[O-H] should be the same as Overlap[H-O].T"
