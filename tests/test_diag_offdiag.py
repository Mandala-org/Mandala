"""
Unit-tests for the new ``diag()`` / ``offdiag()`` helpers of
:class:`BlockMatrix` *and* :class:`IrrepsBlockData`.
"""

from pathlib import Path
import torch
import pytest

from data.openmx_parser import parse_openmx_scfout
from core.orbital_irrep_config import OrbitalIrrepConfig


@pytest.fixture(scope="module")
def snapshot():
    atoms = list("HHHHOO")  # global indexing: 0‥5
    cfg = OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})
    file = Path("./data/small/H2O/original/H2O.matrix")
    return parse_openmx_scfout(file, atoms, cfg, convention="openmx")


# --------------------------------------------------------------------------- #
def test_blockmatrix_diag_offdiag(snapshot):
    D = snapshot.density  # BlockMatrix

    diag = D.diag()
    off = D.offdiag()

    # partition check - every edge must end up **either** in diag or offdiag
    total_diag = sum(t.shape[0] for t in diag.values())
    total_off = sum(t.shape[0] for t in off.values())
    total_all = sum(t.shape[0] for t in D.pair_blocks.values())
    assert total_diag + total_off == total_all

    # per-key correctness against boolean masks
    for key, blk in D.pair_blocks.items():
        edges = D.pair_edges[key]  # (2,E_key)
        mask_diag = edges[0] == edges[1]
        mask_off = ~mask_diag

        if torch.any(mask_diag):
            assert torch.allclose(blk[mask_diag], diag[key], atol=1e-6)
        if torch.any(mask_off):
            assert torch.allclose(blk[mask_off], off[key], atol=1e-6)


def test_irrepsblockdata_diag_offdiag(snapshot):
    Ovec = snapshot.overlap.to_vectors()  # IrrepsBlockData

    diag = Ovec.diag()
    off = Ovec.offdiag()

    total_diag = sum(t.shape[0] for t in diag.values())
    total_off = sum(t.shape[0] for t in off.values())
    total_all = sum(t.shape[0] for t in Ovec.pair_vectors.values())
    assert total_diag + total_off == total_all

    # vector-level equality with masks
    for key, vec in Ovec.pair_vectors.items():
        edges = Ovec.pair_edges[key]
        mask_diag = edges[0] == edges[1]
        mask_off = ~mask_diag
        if torch.any(mask_diag):
            assert torch.allclose(vec[mask_diag], diag[key], atol=1e-6)
        if torch.any(mask_off):
            assert torch.allclose(vec[mask_off], off[key], atol=1e-6)
