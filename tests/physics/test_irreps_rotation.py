"""
IrrepsBlockData Rotation Tests
================================

Verifies that rotating IrrepsBlockData is equivalent to the roundtrip:
    to_blocks() -> rotate() -> to_vectors()

This ensures that the rotation operation on the irrep vector representation
is consistent with the rotation on the block matrix representation.
"""

import math
import pytest
import torch

from core.block_irrep_mapper import BlockIrrepMapper


pytestmark = pytest.mark.physics


def _rotation_z(theta_rad: float) -> torch.Tensor:
    """Return active rotation around the z-axis (right-handed)."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _rotation_x(theta_rad: float) -> torch.Tensor:
    """Return active rotation around the x-axis (right-handed)."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32
    )


def _rotation_y(theta_rad: float) -> torch.Tensor:
    """Return active rotation around the y-axis (right-handed)."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32
    )


def mapper_for(water_snapshot):
    """Create a BlockIrrepMapper for the water snapshot."""
    return BlockIrrepMapper(water_snapshot.hamiltonian.orbital_cfg)


def test_irreps_rotation_equivalence_identity(small_angular_snapshot_e3nn):
    """Test that identity rotation on IrrepsBlockData matches the roundtrip."""
    R = torch.eye(3)

    # Test for Hamiltonian
    water_snapshot = small_angular_snapshot_e3nn
    mapper = mapper_for(water_snapshot)
    ham = water_snapshot.hamiltonian
    ham_irreps = ham.to_vectors(mapper)

    # Direct rotation of irreps
    ham_irreps_rot_direct = ham_irreps.rotate(R, mapper)

    # Roundtrip: to_blocks -> rotate -> to_vectors
    ham_blocks_rot = ham.rotate(R)
    ham_irreps_rot_roundtrip = ham_blocks_rot.to_vectors(mapper)

    # Compare all pair vectors
    for key in ham_irreps.pair_vectors.keys():
        assert torch.allclose(
            ham_irreps_rot_direct.pair_vectors[key],
            ham_irreps_rot_roundtrip.pair_vectors[key],
            atol=1e-6,
        ), f"Identity rotation mismatch for H key {key}"


def test_irreps_rotation_equivalence_z_rotation(small_angular_snapshot_e3nn):
    """Test rotation around z-axis on IrrepsBlockData matches the roundtrip."""
    theta = math.pi / 7.0
    R = _rotation_z(theta)

    # Test for Hamiltonian
    water_snapshot = small_angular_snapshot_e3nn
    mapper = mapper_for(water_snapshot)
    ham = water_snapshot.hamiltonian
    ham_irreps = ham.to_vectors(mapper)

    # Direct rotation of irreps
    ham_irreps_rot_direct = ham_irreps.rotate(R, mapper)

    # Roundtrip: to_blocks -> rotate -> to_vectors
    ham_blocks_rot = ham.rotate(R)
    ham_irreps_rot_roundtrip = ham_blocks_rot.to_vectors(mapper)

    # Compare all pair vectors
    for key in ham_irreps.pair_vectors.keys():
        assert torch.allclose(
            ham_irreps_rot_direct.pair_vectors[key],
            ham_irreps_rot_roundtrip.pair_vectors[key],
            atol=2e-4,
        ), f"Z-rotation mismatch for H key {key}"


def test_irreps_rotation_equivalence_all_matrices(small_angular_snapshot_e3nn):
    """Test rotation equivalence for H, S, and D matrices."""
    theta = math.pi / 4.0
    R = _rotation_x(theta)

    water_snapshot = small_angular_snapshot_e3nn
    mapper = mapper_for(water_snapshot)
    for matrix_name, matrix in [
        ("hamiltonian", water_snapshot.hamiltonian),
        ("overlap", water_snapshot.overlap),
        ("density", water_snapshot.density),
    ]:
        matrix_irreps = matrix.to_vectors(mapper)

        # Direct rotation
        matrix_irreps_rot_direct = matrix_irreps.rotate(R, mapper)

        # Roundtrip
        matrix_blocks_rot = matrix.rotate(R)
        matrix_irreps_rot_roundtrip = matrix_blocks_rot.to_vectors(mapper)

        for key in matrix_irreps.pair_vectors.keys():
            assert torch.allclose(
                matrix_irreps_rot_direct.pair_vectors[key],
                matrix_irreps_rot_roundtrip.pair_vectors[key],
                atol=1e-5,
            ), f"Rotation mismatch for {matrix_name} key {key}"


def test_irreps_rotation_roundtrip(small_angular_snapshot_e3nn):
    """Test that R · Rᵀ on IrrepsBlockData returns to original."""
    theta = math.pi / 5.0
    R = _rotation_y(theta)

    water_snapshot = small_angular_snapshot_e3nn
    mapper = mapper_for(water_snapshot)
    ham_irreps = water_snapshot.hamiltonian.to_vectors(mapper)

    # Rotate and rotate back
    ham_irreps_rot = ham_irreps.rotate(R, mapper)
    ham_irreps_back = ham_irreps_rot.rotate(R.t(), mapper)

    # Should match original
    for key in ham_irreps.pair_vectors.keys():
        assert torch.allclose(
            ham_irreps_back.pair_vectors[key],
            ham_irreps.pair_vectors[key],
            atol=1e-5,
        ), f"Roundtrip failed for key {key}"


def test_irreps_rotation_preserves_structure(small_angular_snapshot_e3nn):
    """Test that rotation preserves the data structure of IrrepsBlockData."""
    R = _rotation_z(math.pi / 3.0)

    water_snapshot = small_angular_snapshot_e3nn
    mapper = mapper_for(water_snapshot)
    ham_irreps = water_snapshot.hamiltonian.to_vectors(mapper)
    ham_irreps_rot = ham_irreps.rotate(R, mapper)

    # Check that structure is preserved
    assert ham_irreps_rot.atoms == ham_irreps.atoms
    assert ham_irreps_rot.atom_counts == ham_irreps.atom_counts
    assert ham_irreps_rot.orbital_cfg == ham_irreps.orbital_cfg
    assert ham_irreps_rot.basis == ham_irreps.basis

    # Check that edge information is unchanged
    for key in ham_irreps.pair_edges.keys():
        assert torch.equal(ham_irreps_rot.pair_edges[key], ham_irreps.pair_edges[key])

    # Check that lookup is unchanged
    assert ham_irreps_rot.lookup == ham_irreps.lookup

    # Check that vector shapes are unchanged
    for key in ham_irreps.pair_vectors.keys():
        assert (
            ham_irreps_rot.pair_vectors[key].shape == ham_irreps.pair_vectors[key].shape
        )


def test_irreps_rotation_multiple_angles(small_angular_snapshot_e3nn):
    """Test rotation equivalence at various angles."""
    angles = [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]

    water_snapshot = small_angular_snapshot_e3nn
    mapper = mapper_for(water_snapshot)
    ham = water_snapshot.hamiltonian
    ham_irreps = ham.to_vectors(mapper)

    for theta in angles:
        R = _rotation_z(theta)

        # Direct rotation
        ham_irreps_rot_direct = ham_irreps.rotate(R, mapper)

        # Roundtrip
        ham_blocks_rot = ham.rotate(R)
        ham_irreps_rot_roundtrip = ham_blocks_rot.to_vectors(mapper)

        for key in ham_irreps.pair_vectors.keys():
            assert torch.allclose(
                ham_irreps_rot_direct.pair_vectors[key],
                ham_irreps_rot_roundtrip.pair_vectors[key],
                atol=2e-4,
            ), f"Rotation mismatch at angle {theta} for key {key}"
