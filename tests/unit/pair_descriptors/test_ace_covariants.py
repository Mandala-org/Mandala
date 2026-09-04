import pytest
import torch
from e3nn import o3

from pair_descriptors.ace_covariants import (
    ACECovariantBasis,
    AtomicNeighborDensity,
    TaggedBondACEBasis,
)


def _rotate_features(values, irreps, rotation):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return values @ irreps.D_from_matrix(rotation).T
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.unit
def test_density_zero_neighbors_and_permutation_invariance():
    density = AtomicNeighborDensity((8, 14), n_radial=2, l_max=2, cutoff=5.0)
    empty = density(torch.empty(0, 3), torch.empty(0, dtype=torch.long), num_centers=2)
    assert empty.shape == (2, density.irreps_out.dim)
    assert torch.count_nonzero(empty) == 0
    vectors = torch.tensor([[1.0, 0.2, -0.1], [-0.3, 1.1, 0.4]], dtype=torch.float64)
    species = torch.tensor([8, 14])
    assert torch.equal(
        density(vectors, species), density(vectors.flip(0), species.flip(0))
    )


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("determinant", [1, -1])
def test_density_and_onsite_ace_are_o3_equivariant(determinant):
    torch.manual_seed(12)
    density = AtomicNeighborDensity((8, 14), n_radial=2, l_max=2, cutoff=5.0)
    ace = ACECovariantBasis(
        density.layout, correlation_order=2, max_degree=5, dtype=torch.float64
    )
    vectors = torch.randn(7, 3, dtype=torch.float64)
    species = torch.tensor([8, 14, 8, 8, 14, 14, 8])
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant == -1:
        rotation = -rotation
    base_density = density(vectors, species)
    rotated_density = density(vectors @ rotation.T, species)
    expected_density = _rotate_features(base_density, density.irreps_out, rotation)
    assert torch.allclose(rotated_density, expected_density, atol=1e-10, rtol=1e-10)
    actual = ace(rotated_density)
    expected = _rotate_features(ace(base_density), ace.irreps_out, rotation)
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )
    assert relative < 1e-9
    assert any(channel.irrep == o3.Irrep("1e") for channel in ace.layout.channels)


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("determinant", [1, -1])
def test_tagged_bond_ace_is_o3_equivariant(determinant):
    torch.manual_seed(34)
    density = AtomicNeighborDensity((8, 14), n_radial=1, l_max=2, cutoff=5.0)
    basis = TaggedBondACEBasis(
        density.layout,
        bond_n_radial=2,
        bond_l_max=3,
        bond_cutoff=6.5,
        max_degree=5,
        dtype=torch.float64,
    )
    descriptor_i = density(
        torch.randn(6, 3, dtype=torch.float64), torch.tensor([8, 8, 14, 8, 14, 8])
    )
    descriptor_j = density(
        torch.randn(5, 3, dtype=torch.float64), torch.tensor([14, 8, 14, 14, 8])
    )
    bond = torch.tensor([[1.4, -0.2, 0.3]], dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant == -1:
        rotation = -rotation
    rotated_i = _rotate_features(descriptor_i, density.irreps_out, rotation)
    rotated_j = _rotate_features(descriptor_j, density.irreps_out, rotation)
    actual = basis(rotated_i, bond @ rotation.T, rotated_j)
    expected = _rotate_features(
        basis(descriptor_i, bond, descriptor_j), basis.irreps_out, rotation
    )
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )
    assert relative < 1e-9
    assert all(
        channel.correlation_order < 2 or channel.family == "bond_environment"
        for channel in basis.layout.channels
    )
