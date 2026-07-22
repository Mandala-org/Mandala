"""
Compare OpenMX reference files that were written out in two coordinate frames.

We load

* `H2O_original.out`
* `H2O_rotated.out`
* (periodic variants idem)

— parse them in **e3nn** basis, rotate the *original* snapshot inside
Python, and assert element-wise equality with the “rotated” reference.
"""

import pytest
import math

import torch


def _rotation_matrix() -> torch.Tensor:
    """Return the 3x3 tensor representing the (-π/6) y-rotation."""
    theta = -math.pi / 6.0
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1, 0.0], [-s, 0.0, c]], dtype=torch.float32)


def _assert_snapshot_equal(a, b, *, atol=1e-5):
    """Dense equality helper (covers all three matrices)."""
    for name in ("hamiltonian", "overlap", "density"):
        mat_a = getattr(a, name).to_dense()
        mat_b = getattr(b, name).to_dense()
        assert torch.allclose(mat_a, mat_b, atol=atol), f"{name} mismatch beyond {atol}"
    assert torch.allclose(a.positions, b.positions, atol=atol), "positions mismatch"
    # if a.forces is not None and b.forces is not None:
    #     assert torch.allclose(a.forces, b.forces, atol=atol), "forces mismatch"
    if a.box is not None and b.box is not None:
        assert torch.allclose(a.box, b.box, atol=atol), "box mismatch"


@pytest.mark.physics
def test_pre_rotated_files_match_in_code_rotation(h2o_rotation_pair):
    """
    Rotate the *original* snapshot by the known matrix and compare to the
    reference “rotated” file published in the dataset.
    """
    snap_orig, snap_rot_ref = h2o_rotation_pair

    R = _rotation_matrix()
    snap_rot_calc = snap_orig.rotate(R)

    # B' = R @ B
    # R  = B'@B^-1
    _assert_snapshot_equal(snap_rot_calc, snap_rot_ref, atol=0.005)

    # physics invariants (redundant but nice to have)
    assert torch.allclose(
        snap_rot_calc.get_energy(), snap_rot_ref.get_energy(), atol=0.0001
    )
    assert torch.allclose(
        snap_rot_calc.get_number_of_electrons(),
        snap_rot_ref.get_number_of_electrons(),
        atol=0.0001,
    )
    assert torch.allclose(
        snap_rot_calc.get_energy(), snap_orig.get_energy(), atol=0.0001
    )
    assert torch.allclose(
        snap_rot_calc.get_number_of_electrons(),
        snap_orig.get_number_of_electrons(),
        atol=0.0001,
    )
