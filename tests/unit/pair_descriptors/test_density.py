import pytest
import torch
from e3nn import o3

from pair_descriptors import RawNeighborDensityDescriptor


@pytest.mark.unit
@pytest.mark.parametrize("radial_basis", ["spherical_bessel", "zernike"])
def test_d1_additivity_puncturing_and_determinism(radial_basis):
    descriptor = RawNeighborDensityDescriptor(
        (8, 14), radial_basis=radial_basis, n_radial=3, l_max=3, cutoff=6.5
    ).to(dtype=torch.float64)
    displacement = torch.tensor(
        [[1.0, 0.2, -0.3], [-0.4, 1.2, 0.5], [0.7, -0.2, 1.4]],
        dtype=torch.float64,
    )
    species = torch.tensor([8, 14, 8])
    full = descriptor(displacement, species)
    permuted = descriptor(displacement[[2, 0, 1]], species[[2, 0, 1]])
    assert torch.equal(full, permuted)
    punctured = descriptor.puncture(full, species[1:2], displacement[1:2])
    expected = descriptor(displacement[[0, 2]], species[[0, 2]])
    assert torch.allclose(punctured, expected, atol=1e-12, rtol=1e-12)
    assert descriptor.metadata == descriptor.metadata


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("radial_basis", ["spherical_bessel", "zernike"])
@pytest.mark.parametrize("determinant", [1, -1])
def test_d1_full_o3_equivariance(radial_basis, determinant):
    torch.manual_seed(7)
    descriptor = RawNeighborDensityDescriptor(
        (8, 14), radial_basis=radial_basis, n_radial=2, l_max=4, cutoff=7.5
    ).to(dtype=torch.float64)
    displacement = torch.randn(12, 3, dtype=torch.float64)
    species = torch.tensor([8, 14] * 6)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant < 0:
        rotation = -rotation
    reference = descriptor(displacement, species)
    actual = descriptor(displacement @ rotation.T, species)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        action = descriptor.irreps_out.D_from_matrix(rotation)
    finally:
        torch.set_default_dtype(previous)
    assert torch.allclose(actual, reference @ action.T, atol=2e-10, rtol=2e-10)


@pytest.mark.unit
@pytest.mark.parametrize("radial_basis", ["spherical_bessel", "zernike"])
def test_d1_ragged_empty_and_production_precision(radial_basis):
    descriptor = RawNeighborDensityDescriptor(
        (8, 14), radial_basis=radial_basis, n_radial=2, l_max=3, cutoff=8.5
    )
    empty = descriptor(
        torch.empty(0, 3), torch.empty(0, dtype=torch.long), num_centers=3
    )
    assert empty.shape == (3, descriptor.irreps_out.dim)
    assert torch.count_nonzero(empty) == 0
    values = descriptor(
        torch.randn(8, 3),
        torch.tensor([8, 14] * 4),
        torch.tensor([0, 0, 1, 1, 1, 2, 2, 2]),
        num_centers=3,
    )
    assert torch.isfinite(values).all()
