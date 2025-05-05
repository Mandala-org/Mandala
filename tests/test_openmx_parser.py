import pytest
from pathlib import Path

import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout


@pytest.fixture(scope="module")
def orbital_cfg():
    # H: 2s + 1p (dim 5);  O: 3s + 2p (dim 7)
    return OrbitalIrrepConfig.from_dict(
        {
            "H": ["2x0e", "1x1o"],
            "O": ["3x0e", "2x1o"],
        }
    )


def test_parse_h2o_small(tmp_path, orbital_cfg):
    sample = Path("./data/small/H2O/H2O_original.out")
    atoms = list("HHHHOO")  # global order

    mats = parse_openmx_scfout(sample, atoms, orbital_cfg)
    assert "hamiltonian" in mats and "overlap" in mats and "density" in mats

    snap_H = mats["hamiltonian"]
    # there should be at least one H‑O block
    assert "H-O" in snap_H.keys()
    E_HO, d_H, d_O = snap_H["H-O"].shape
    assert d_H == 5 and d_O == 9

    # round‑trip vector check
    snap_vec = snap_H.to_vectors()
    assert snap_vec["H-O"].shape[-1] == snap_H.mapper.vector_dim("H-O")


def test_parse_h2o_small_pbc(orbital_cfg):
    sample = Path("./data/small/H2O/H2O_pbc_original.out")
    atoms = list("HHHHOO")

    mats = parse_openmx_scfout(sample, atoms, orbital_cfg, pbc_sum=True)
    snap_D = mats["density"]
    # ensure that duplicate Rn blocks were summed: count of H‑H edges is 16 (fully connected dir graph)
    assert snap_D["H-H"].shape == (16, 5, 5)
    assert snap_D["O-O"].shape == (4, 9, 9)
    assert snap_D["H-O"].shape == (8, 5, 9)
    assert snap_D["O-H"].shape == (8, 9, 5)
    # assert O-H is the same as H-O.T
    snap_D = snap_D.standardize_edges()
    snap_D_T = snap_D.transpose().standardize_edges()
    assert torch.allclose(
        snap_D["O-H"], snap_D_T["O-H"], atol=1e-5
    ), "D[O-H] should be the same as D[H-O].T"

    snap_H = mats["hamiltonian"]
    assert snap_H["H-H"].shape == (16, 5, 5)
    assert snap_H["O-O"].shape == (4, 9, 9)
    assert snap_H["H-O"].shape == (8, 5, 9)
    assert snap_H["O-H"].shape == (8, 9, 5)
    # assert O-H is the same as H-O.T
    snap_H = snap_H.standardize_edges()
    snap_H_T = snap_H.transpose().standardize_edges()
    assert torch.allclose(
        snap_H["O-H"], snap_H_T["O-H"], atol=1e-5
    ), "Ham[O-H] should be the same as Ham[H-O].T"

    snap_O = mats["overlap"]
    assert snap_O["H-H"].shape == (16, 5, 5)
    assert snap_O["O-O"].shape == (4, 9, 9)
    assert snap_O["H-O"].shape == (8, 5, 9)
    assert snap_O["O-H"].shape == (8, 9, 5)
    # assert O-H is the same as H-O.T
    snap_O = snap_O.standardize_edges()
    snap_O_T = snap_O.transpose().standardize_edges()
    assert torch.allclose(
        snap_O["O-H"], snap_O_T["O-H"], atol=1e-5
    ), "Overlap[O-H] should be the same as Overlap[H-O].T"
