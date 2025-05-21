"""
Compare OpenMX reference files that were written out in two coordinate frames.

We load

* `H2O_original.out`
* `H2O_rotated.out`
* (periodic variants idem)

— parse them in **e3nn** basis, rotate the *original* snapshot inside
Python, and assert element-wise equality with the “rotated” reference.
"""

from pathlib import Path
import math

import torch
import pytest

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout


# ------------------------------------------------------------------ helpers
_ATOMS = list("HHHHOO")  # global index order used in the test files
_CFG = OrbitalIrrepConfig.from_dict({"H": "2s1p", "O": "3s2p"})


def _rotation_matrix() -> torch.Tensor:
    """Return the 3x3 tensor representing the (-π/6) z-rotation."""
    theta = -math.pi / 6.0
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _load_pair(orig_name: str, rot_name: str):
    base = Path("./data/small/H2O")
    snap_orig = parse_openmx_scfout(base / orig_name, _ATOMS, _CFG, convention="openmx")
    snap_rot_ref = parse_openmx_scfout(
        base / rot_name, _ATOMS, _CFG, convention="openmx"
    )
    # canonicalise edge order → lexicographic (avoids ties w/ equal norms)
    snap_orig = snap_orig
    snap_rot_ref = snap_rot_ref
    return snap_orig, snap_rot_ref


def _assert_snapshot_equal(a, b, *, atol=1e-5):
    """Dense equality helper (covers all three matrices)."""
    for name in ("hamiltonian", "overlap", "density"):
        mat_a = getattr(a, name).to_dense()
        mat_b = getattr(b, name).to_dense()
        assert torch.allclose(mat_a, mat_b, atol=atol), f"{name} mismatch beyond {atol}"


# ------------------------------------------------------------------ parametrised tests
@pytest.mark.parametrize(
    "orig_file, rot_file",
    [
        ("H2O_original.out", "H2O_rotated.out"),
        ("H2O_pbc_original.out", "H2O_pbc_rotated.out"),
    ],
)
def test_pre_rotated_files_match_in_code_rotation(orig_file, rot_file):
    """
    Rotate the *original* snapshot by the known matrix and compare to the
    reference “rotated” file published in the dataset.
    """
    snap_orig, snap_rot_ref = _load_pair(orig_file, rot_file)

    R = _rotation_matrix()
    snap_rot_calc = snap_orig.rotate(R)

    _assert_snapshot_equal(snap_rot_calc, snap_rot_ref, atol=0.002)

    # physics invariants (redundant but nice to have)
    assert torch.allclose(
        snap_rot_calc.get_energy(), snap_rot_ref.get_energy(), atol=0.01
    )
    assert torch.allclose(
        snap_rot_calc.get_number_of_electrons(),
        snap_rot_ref.get_number_of_electrons(),
        atol=0.01,
    )
    assert torch.allclose(snap_rot_calc.get_energy(), snap_orig.get_energy(), atol=0.01)
    assert torch.allclose(
        snap_rot_calc.get_number_of_electrons(),
        snap_orig.get_number_of_electrons(),
        atol=0.01,
    )
