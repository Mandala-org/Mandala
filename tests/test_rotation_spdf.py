# tests/test_rotation_spdf.py

"""
Compare the “spdf” reference files—which include d- and f-orbitals—
against an in-code rotation of the original snapshot.
"""

from pathlib import Path
import math

import torch
import pytest

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout

# ---------------------------------------------
# test parameters
ATOMS = list("HHHHOO")
# we now include d (l=2) and f (l=3) channels on O
IRREPS_CFG = OrbitalIrrepConfig.from_dict(
    {
        "H": "3x0e+2x1o",  # H: 3 s + 2 p
        "O": "3x0e+3x1o+2x2e+1x3o",  # O: 3 s + 3 p + 2 d + 1 f
    }
)


def _rotation_matrix() -> torch.Tensor:
    """y-axis rotation by –π/6."""
    θ = -math.pi / 6.0
    c, s = math.cos(θ), math.sin(θ)
    return torch.tensor([[c, 0.0, s], [0.0, 1, 0.0], [-s, 0.0, c]], dtype=torch.float32)


def _load_pair(orig: str, rot: str):
    base = Path("data/small/H2O")
    snap_o = parse_openmx_scfout(base / orig, ATOMS, IRREPS_CFG, convention="e3nn")
    snap_r = parse_openmx_scfout(base / rot, ATOMS, IRREPS_CFG, convention="e3nn")
    return snap_o, snap_r


def _assert_snapshot_equal(a, b, *, atol=1e-5):
    for name in ("hamiltonian", "overlap", "density"):
        da = getattr(a, name).to_dense()
        db = getattr(b, name).to_dense()
        assert torch.allclose(da, db, atol=atol), f"{name} differs"


@pytest.mark.parametrize(
    "orig_file,rot_file",
    [
        ("H2O_spdf_original.out", "H2O_spdf_rotated.out"),
    ],
)
def test_rotation_spdf_equivariance(orig_file, rot_file):
    """rotating the original spdf-snapshot must match the rotated reference."""
    snap_orig, snap_ref = _load_pair(orig_file, rot_file)
    R = _rotation_matrix()

    # apply block-by-block rotation in-code
    snap_calc = snap_orig.rotate(R)

    # dense-matrix comparison
    _assert_snapshot_equal(snap_calc, snap_ref, atol=2e-3)

    # invariants: energy & electron count unchanged
    assert torch.allclose(snap_calc.get_energy(), snap_ref.get_energy(), atol=0.01)
    assert torch.allclose(
        snap_calc.get_number_of_electrons(),
        snap_ref.get_number_of_electrons(),
        atol=0.01,
    )
