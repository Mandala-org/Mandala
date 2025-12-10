"""
Unit-tests for the new ``diag()`` / ``offdiag()`` helpers of
:class:`BlockMatrix` *and* :class:`IrrepsBlockData`.
"""

import pytest
from pathlib import Path
import torch

from data.snapshot import Snapshot
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config


@pytest.fixture(scope="module")
def data():
    cfg = Config(cutoff_matrix=15)
    return Snapshot.from_openmx(
        Path("./data/small/H2O/original/H2O.matrix"),
        Path("./data/small/H2O/original/H2O.info.out"),
        cfg=cfg,
    )


# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_blockmatrix_diag_offdiag(data):
    snapshot = data
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
        mask_diag = (edges[3] == edges[4]) & edges[:3].eq(0).all(dim=0)
        mask_off = ~mask_diag

        if torch.any(mask_diag):
            assert torch.allclose(blk[mask_diag], diag[key], atol=1e-6)
        if torch.any(mask_off):
            assert torch.allclose(blk[mask_off], off[key], atol=1e-6)


@pytest.mark.unit
def test_irrepsblockdata_diag_offdiag(data):
    snapshot = data
    mapper = BlockIrrepMapper(snapshot.density.orbital_cfg)
    dens_vecs = snapshot.density.to_vectors(mapper)

    diag = dens_vecs.diag()
    off = dens_vecs.offdiag()

    total_diag = sum(t.shape[0] for t in diag.values())
    total_off = sum(t.shape[0] for t in off.values())
    total_all = sum(t.shape[0] for t in dens_vecs.pair_vectors.values())
    assert total_diag + total_off == total_all

    # vector-level equality with masks
    for key, vec in dens_vecs.pair_vectors.items():
        edges = dens_vecs.pair_edges[key]
        mask_diag = (edges[3] == edges[4]) & edges[:3].eq(0).all(dim=0)
        mask_off = ~mask_diag
        if torch.any(mask_diag):
            assert torch.allclose(vec[mask_diag], diag[key], atol=1e-6)
        if torch.any(mask_off):
            assert torch.allclose(vec[mask_off], off[key], atol=1e-6)
