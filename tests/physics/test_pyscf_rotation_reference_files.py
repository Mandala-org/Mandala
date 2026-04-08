from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.snapshot import Snapshot


def _nearest_rotation_from_boxes(
    box_orig: torch.Tensor, box_rot: torch.Tensor
) -> torch.Tensor:
    transform = torch.linalg.solve(box_orig, box_rot)
    u, _, vh = torch.linalg.svd(transform)
    rot_t = u @ vh
    if torch.det(rot_t) < 0:
        u[:, -1] *= -1.0
        rot_t = u @ vh
    return rot_t.T


def _max_block_diff(a, b) -> float:
    diff = a - b
    return max(float(block.abs().max().item()) for block in diff.pair_blocks.values())


@pytest.mark.physics
def test_pyscf_rotated_files_match_snapshot_rotation():
    base = Path("data/pyscf_baseline/results")
    snap_orig = Snapshot.from_pyscf(
        base / "h2o_original_rks_openmx_like.npz",
        base / "h2o_original_rks_openmx_like.json",
        convention="e3nn",
    )
    snap_rot = Snapshot.from_pyscf(
        base / "h2o_rotated_rks_openmx_like.npz",
        base / "h2o_rotated_rks_openmx_like.json",
        convention="e3nn",
    )

    assert snap_orig.box is not None and snap_rot.box is not None
    assert snap_orig.forces is not None and snap_rot.forces is not None
    assert snap_orig.stress is not None and snap_rot.stress is not None

    R = _nearest_rotation_from_boxes(snap_orig.box, snap_rot.box)
    snap_rot_calc = snap_orig.rotate(R)

    e_orig = float(snap_orig.info["pyscf"]["total_energy_hartree"])
    e_rot = float(snap_rot.info["pyscf"]["total_energy_hartree"])
    assert abs(e_orig - e_rot) < 5e-5

    assert torch.allclose(snap_rot_calc.positions, snap_rot.positions, atol=5e-4)
    assert torch.allclose(snap_rot_calc.box, snap_rot.box, atol=5e-4)
    assert torch.allclose(snap_rot_calc.forces, snap_rot.forces, atol=5e-4)
    assert torch.allclose(snap_rot_calc.stress, snap_rot.stress, atol=5e-4)

    assert _max_block_diff(snap_rot_calc.hamiltonian, snap_rot.hamiltonian) < 2e-3
    assert _max_block_diff(snap_rot_calc.overlap, snap_rot.overlap) < 2e-3
    assert _max_block_diff(snap_rot_calc.density, snap_rot.density) < 2e-3
