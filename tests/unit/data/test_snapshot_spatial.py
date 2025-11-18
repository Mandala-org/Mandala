"""
Spatial helpers on large periodic snapshot
==========================================

Uses the 216-atom diamond-Si cell included under ``data/big/silicon/300K``.
"""

import pytest


import torch


# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_canonical_edge_ordering(si_snapshot):
    D = si_snapshot.density

    for key in D.keys():
        edges = D.pair_edges[key]
        is_diag = edges[0] == edges[1]

        # Find the first off-diagonal edge, if any
        off_diag_indices = torch.where(~is_diag)[0]
        if len(off_diag_indices) > 0:
            first_off_diag_idx = off_diag_indices[0]
            # All edges before the first off-diagonal edge must be diagonal
            assert torch.all(is_diag[:first_off_diag_idx])
            # All edges from the first off-diagonal edge onwards must be diagonal
            assert torch.all(~is_diag[first_off_diag_idx:])


@pytest.mark.unit
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


@pytest.mark.unit
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
