import pytest
import torch
from e3nn import o3

from pair_descriptors import FourierBesselDescriptor, IrreducibleMomentDescriptor


def _descriptors():
    return (
        IrreducibleMomentDescriptor((8, 14), max_degree=4, cutoff=6.5),
        FourierBesselDescriptor((8, 14), frequency_count=3, l_max=4, cutoff=6.5),
    )


@pytest.mark.unit
@pytest.mark.parametrize("descriptor", _descriptors())
def test_d2_d3_additivity_puncturing_and_metadata(descriptor):
    descriptor = descriptor.to(dtype=torch.float64)
    vectors = torch.tensor(
        [[1.0, 0.2, -0.3], [-0.4, 1.2, 0.5], [0.7, -0.2, 1.4]],
        dtype=torch.float64,
    )
    species = torch.tensor([8, 14, 8])
    full = descriptor(vectors, species)
    permuted = descriptor(vectors[[2, 0, 1]], species[[2, 0, 1]])
    assert torch.equal(full, permuted)
    punctured = descriptor.puncture(full, species[1:2], vectors[1:2])
    expected = descriptor(vectors[[0, 2]], species[[0, 2]])
    assert torch.allclose(punctured, expected, atol=1e-12, rtol=1e-12)
    assert len(descriptor.metadata["content_hash"]) == 64


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("descriptor", _descriptors())
@pytest.mark.parametrize("determinant", [1, -1])
def test_d2_d3_full_o3_equivariance(descriptor, determinant):
    descriptor = descriptor.to(dtype=torch.float64)
    torch.manual_seed(17)
    vectors = torch.randn(8, 3, dtype=torch.float64)
    species = torch.tensor([8, 14] * 4)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant < 0:
        rotation = -rotation
    reference = descriptor(vectors, species)
    actual = descriptor(vectors @ rotation.T, species)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        action = descriptor.irreps_out.D_from_matrix(rotation)
    finally:
        torch.set_default_dtype(previous)
    assert torch.allclose(actual, reference @ action.T, atol=3e-10, rtol=3e-10)


@pytest.mark.unit
@pytest.mark.parametrize("descriptor", _descriptors())
def test_d2_d3_jacobian_matches_finite_difference(descriptor):
    descriptor = descriptor.to(dtype=torch.float64)
    vectors = torch.tensor([[1.1, 0.3, -0.2], [-0.5, 1.3, 0.4]], dtype=torch.float64)
    species = torch.tensor([8, 14])
    jacobian = descriptor.jacobian(vectors, species)
    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.1, 0.2]], dtype=torch.float64)
    epsilon = 3e-4
    finite = (
        descriptor(vectors + epsilon * direction, species)
        - descriptor(vectors - epsilon * direction, species)
    )[0] / (2 * epsilon)
    analytic = torch.einsum("kna,na->k", jacobian, direction)
    assert torch.allclose(analytic, finite, atol=2e-8, rtol=2e-7)


@pytest.mark.unit
@pytest.mark.parametrize(
    "descriptor",
    [
        IrreducibleMomentDescriptor((8, 14), max_degree=10, cutoff=10.5),
        FourierBesselDescriptor((8, 14), frequency_count=8, l_max=6, cutoff=10.5),
    ],
)
def test_largest_planned_d2_d3_configuration_is_finite(descriptor):
    values = descriptor.to(dtype=torch.float64)(
        torch.tensor([[1e-3, 0.0, 0.0], [4.0, 2.0, -1.0]], dtype=torch.float64),
        torch.tensor([8, 14]),
    )
    assert values.shape == (1, descriptor.irreps_out.dim)
    assert torch.isfinite(values).all()
