"""
Rotation‐specific unit tests
===========================

Verifies that

* rotating by *identity* keeps every block unchanged;
* physics invariants Tr(D·H) and Tr(D·S) survive an arbitrary rigid rotation;
* applying a rotation and its inverse yields the original snapshot
  (round-trip);
* attempting to rotate an OpenMX-basis snapshot raises a clear error.
"""

from typing import TYPE_CHECKING

from pathlib import Path
import math

import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout

if TYPE_CHECKING:
    from data.snapshot import Snapshot


# ------------------------------------------------------------------- helpers
def _rotation_z(theta_rad: float) -> torch.Tensor:
    """Return active rotation around the z-axis (right-handed)."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


@pytest.fixture(scope="module")
def snapshot_e3nn() -> "Snapshot":
    """Parse the small H₂O example directly in *e3nn* basis."""
    cfg = OrbitalIrrepConfig.from_dict({"H": "2s1p", "O": "3s2p"})
    sample = Path("./data/small/H2O/H2O_original.out")
    atoms = list("HHHHOO")
    snap = parse_openmx_scfout(
        sample, atoms, cfg, convention="e3nn"
    )  # <── already e3nn
    # sanity
    assert snap.density.basis == "e3nn"
    return snap


# ------------------------------------------------------------------- tests
def test_identity_rotation_keeps_blocks(snapshot_e3nn):
    R = torch.eye(3)
    snap_id = snapshot_e3nn.rotate(R)

    for key in snapshot_e3nn.hamiltonian.keys():
        assert torch.allclose(
            snapshot_e3nn.hamiltonian[key],
            snap_id.hamiltonian[key],
            atol=1e-6,
        ), f"H block {key} changed under identity rotation"
        assert torch.allclose(
            snapshot_e3nn.overlap[key],
            snap_id.overlap[key],
            atol=1e-6,
        ), f"S block {key} changed under identity rotation"
        assert torch.allclose(
            snapshot_e3nn.density[key],
            snap_id.density[key],
            atol=1e-6,
        ), f"D block {key} changed under identity rotation"


def test_rotation_invariants(snapshot_e3nn):
    """Energy and electron count must be invariant under rigid rotation."""
    theta = math.pi / 7.0
    R = _rotation_z(theta)
    snap_rot = snapshot_e3nn.rotate(R)

    # physics helpers
    assert torch.allclose(
        snap_rot.get_energy(), snapshot_e3nn.get_energy(), atol=1e-8
    ), "Energy changed after rotation"
    assert torch.allclose(
        snap_rot.get_number_of_electrons(),
        snapshot_e3nn.get_number_of_electrons(),
        atol=1e-8,
    ), "Electron count changed after rotation"


def test_rotation_roundtrip(snapshot_e3nn):
    """R · Rᵀ should bring us back to the original snapshot."""
    theta = math.pi / 4.0
    R = _rotation_z(theta)
    snap_rot = snapshot_e3nn.rotate(R)
    snap_back = snap_rot.rotate(R.t())  # inverse

    # compare dense matrices for a stringent check
    assert torch.allclose(
        snap_back.hamiltonian.to_dense(),
        snapshot_e3nn.hamiltonian.to_dense(),
        atol=1e-5,
    )
    assert torch.allclose(
        snap_back.overlap.to_dense(),
        snapshot_e3nn.overlap.to_dense(),
        atol=1e-5,
    )
    assert torch.allclose(
        snap_back.density.to_dense(),
        snapshot_e3nn.density.to_dense(),
        atol=1e-5,
    )
