"""
Spatial helpers on large periodic snapshot
==========================================

Uses the 216-atom diamond-Si cell included under ``data/big/silicon/300K``.
"""

from pathlib import Path

import numpy as np
import torch
import pytest

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout
from data.snapshot import Snapshot


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def si_snapshot():
    """
    Load *matrices* from the OpenMX output, add positions / box information
    and return a complete :class:`Snapshot`.
    """
    base = Path("./data/big/silicon/300K")
    scf_file = base / "Si_0.out"
    coords_npy = base / "coords_0.npy"

    # ----- atomic meta-data ------------------------------------------------
    coords = torch.tensor(np.load(coords_npy), dtype=torch.float32)  # (216,3)
    box = torch.diag(torch.tensor([16.293, 16.293, 16.293], dtype=torch.float32))

    atoms = ["Si"] * coords.shape[0]
    cfg = OrbitalIrrepConfig.from_dict(
        {"Si": ["2x0e", "2x1o", "1x2e"]}  #  13-dim orbital basis
    )

    # parse matrices (BlockMatrix objects, basis="openmx")
    snap_raw = parse_openmx_scfout(scf_file, atoms, cfg, convention="openmx")

    # wrap into full Snapshot with geometry -------------------------------
    return Snapshot(
        snap_raw.hamiltonian,
        snap_raw.overlap,
        snap_raw.density,
        positions=coords,
        box=box,
    )


# ---------------------------------------------------------------------------
def test_max_distance_periodic(si_snapshot):
    """
    Largest *minimal-image* separation must not exceed half the smallest
    box vector (≈ 8.15 Å for the 16.293 Å cube).
    """
    # get a Python float from the 0‐dim tensor
    max_d = si_snapshot.max_distance().item()
    assert max_d <= 8.15
    # trivial lower-bound sanity (nearest Si-Si is ~2.35 Å)
    assert max_d >= 2.0


def test_filter_by_distance(si_snapshot):
    """
    After applying a 5.5 Å cut-off no edge in **any** matrix should be longer
    than that and the total edge count must drop.
    """
    cutoff = 5.5
    snap_cut = si_snapshot.filter_by_distance(cutoff)

    # ------------- 1. all surviving edges respect the cut-off
    assert snap_cut.max_distance().item() <= cutoff + 1e-5

    # ------------- 2. edge count strictly decreases in the density matrix
    def _edge_count(mat):
        return sum(blk.shape[0] for blk in mat.pair_blocks.values())

    n_before = _edge_count(si_snapshot.density)
    n_after = _edge_count(snap_cut.density)
    assert n_after < n_before
