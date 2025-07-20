"""
Tests for FHI-AIMS parser and basis conversion.
"""

import torch
import numpy as np

from data.snapshot import Snapshot


def test_fhiaims_parsing():
    """Test that FHI-AIMS files are parsed correctly."""
    geometry_path = "data/fhi-aims/original/basis_small/geometry.in"
    basis_path = "data/fhi-aims/original/basis_small/basis-indices.out"
    hamiltonian_path = "data/fhi-aims/original/basis_small/H_spin_01_kpt_000001.csc"
    overlap_path = "data/fhi-aims/original/basis_small/S_spin_01_kpt_000001.csc"
    density_path = "data/fhi-aims/original/basis_small/D_spin_01_kpt_000001.csc"

    snap = Snapshot.from_fhiaims(
        geometry_path,
        basis_path,
        hamiltonian_path,
        overlap_path,
        density_path,
        convention="fhi-aims",
    )

    assert snap.hamiltonian.basis == "fhi-aims"
    assert snap.positions is not None
    assert snap.box is not None
    assert len(snap.hamiltonian.atoms) == 6


def test_fhiaims_basis_conversion():
    """Test the FHI-AIMS to e3nn basis conversion."""
    # Load original and rotated snapshots
    snap_orig = Snapshot.from_fhiaims(
        "data/fhi-aims/original/basis_small/geometry.in",
        "data/fhi-aims/original/basis_small/basis-indices.out",
        "data/fhi-aims/original/basis_small/H_spin_01_kpt_000001.csc",
        "data/fhi-aims/original/basis_small/S_spin_01_kpt_000001.csc",
        "data/fhi-aims/original/basis_small/D_spin_01_kpt_000001.csc",
        convention="e3nn",
    )
    snap_rot = Snapshot.from_fhiaims(
        "data/fhi-aims/rotated/basis_small/geometry.in",
        "data/fhi-aims/rotated/basis_small/basis-indices.out",
        "data/fhi-aims/rotated/basis_small/H_spin_01_kpt_000001.csc",
        "data/fhi-aims/rotated/basis_small/S_spin_01_kpt_000001.csc",
        "data/fhi-aims/rotated/basis_small/D_spin_01_kpt_000001.csc",
        convention="e3nn",
    )

    # Define the rotation matrix (pi/6 around y-axis)
    theta = np.pi / 6
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)

    # Rotate the original snapshot
    snap_orig_rotated = snap_orig.rotate(R)

    # Compare the dense matrices
    ham_rot_dense = snap_rot.hamiltonian.to_dense()
    ham_orig_rot_dense = snap_orig_rotated.hamiltonian.to_dense()

    assert torch.allclose(ham_rot_dense, ham_orig_rot_dense, atol=0.0005)
