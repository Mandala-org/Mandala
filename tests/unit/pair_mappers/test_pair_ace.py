import pytest
import torch
from e3nn import o3

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_descriptors import AtomicNeighborDensity
from pair_hamiltonian.output_schema import (
    FullBlockIrrepTransform,
    o3_representation_matrix,
)
from pair_mappers import EquivariantRidgeAccumulator, NativeACEPairMapper


@pytest.fixture(scope="module")
def components():
    config = OrbitalIrrepConfig.from_dict({"A": "1s1p", "B": "1s1p"})
    transform = FullBlockIrrepTransform(config, dtype=torch.float64)
    density = AtomicNeighborDensity((8, 14), n_radial=1, l_max=1, cutoff=4.0)
    model = NativeACEPairMapper(
        transform,
        density.layout,
        onsite_correlation_order=2,
        onsite_max_degree=4,
        bond_n_radial=1,
        bond_l_max=2,
        bond_cutoff=5.0,
        offsite_max_degree=4,
        ridge=1e-8,
        dtype=torch.float64,
    )
    for name, buffer in model.named_buffers():
        if "weight_" in name:
            buffer.copy_(torch.randn_like(buffer))
    return transform, density, model


@pytest.mark.unit
def test_pair_mapper_returns_one_complete_full_block(components):
    transform, density, model = components
    descriptor = density(
        torch.tensor([[1.0, 0.2, 0.3]], dtype=torch.float64), torch.tensor([8])
    )
    output = model(
        descriptor,
        descriptor,
        torch.zeros(1, 3, dtype=torch.float64),
        "A",
        "A",
        {"onsite": True, "ignored_graph_connectivity": object()},
    )
    assert output.shape == (1, transform.schema("A-A").vector_dimension)


@pytest.mark.unit
def test_heterogeneous_pair_reversal_is_exact(components):
    transform, density, model = components
    descriptor_i = density(
        torch.tensor([[1.0, 0.2, 0.3]], dtype=torch.float64), torch.tensor([8])
    )
    descriptor_j = density(
        torch.tensor([[-0.4, 0.8, 0.1]], dtype=torch.float64), torch.tensor([14])
    )
    displacement = torch.tensor([[1.3, 0.2, -0.1]], dtype=torch.float64)
    forward = model.predict_offsite(
        ("A", "B"), descriptor_i, displacement, descriptor_j
    )
    reverse = model.predict_offsite(
        ("B", "A"), descriptor_j, -displacement, descriptor_i
    )
    assert torch.allclose(
        reverse, transform.reverse(("A", "B"), forward), atol=1e-12, rtol=1e-12
    )


@pytest.mark.unit
def test_homonuclear_pair_reversal_is_exact(components):
    transform, density, model = components
    descriptor_i = density(
        torch.tensor([[1.0, 0.2, 0.3]], dtype=torch.float64), torch.tensor([8])
    )
    descriptor_j = density(
        torch.tensor([[-0.4, 0.8, 0.1]], dtype=torch.float64), torch.tensor([14])
    )
    displacement = torch.tensor([[1.3, 0.2, -0.1]], dtype=torch.float64)
    forward = model.predict_offsite(
        ("A", "A"), descriptor_i, displacement, descriptor_j
    )
    reverse = model.predict_offsite(
        ("A", "A"), descriptor_j, -displacement, descriptor_i
    )
    assert torch.allclose(
        reverse, transform.reverse(("A", "A"), forward), atol=1e-12, rtol=1e-12
    )


@pytest.mark.unit
def test_onsite_hermiticity_is_exact(components):
    transform, density, model = components
    descriptor = density(
        torch.tensor([[1.0, 0.2, 0.3]], dtype=torch.float64), torch.tensor([8])
    )
    onsite = model.predict_onsite("A", descriptor)
    assert torch.allclose(
        onsite,
        transform.reverse(("A", "A"), onsite),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("determinant", [1, -1])
def test_pair_mapper_is_full_o3_equivariant(components, determinant):
    transform, density, model = components
    descriptor_i = density(
        torch.tensor([[1.0, 0.2, 0.3]], dtype=torch.float64), torch.tensor([8])
    )
    descriptor_j = density(
        torch.tensor([[-0.4, 0.8, 0.1]], dtype=torch.float64), torch.tensor([14])
    )
    displacement = torch.tensor([[1.3, 0.2, -0.1]], dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant == -1:
        rotation = -rotation
    descriptor_action = o3_representation_matrix(density.irreps_out, rotation)
    actual = model.predict_offsite(
        ("A", "B"),
        descriptor_i @ descriptor_action.T,
        displacement @ rotation.T,
        descriptor_j @ descriptor_action.T,
    )
    expected = (
        model.predict_offsite(("A", "B"), descriptor_i, displacement, descriptor_j)
        @ transform.output_action("A-B", rotation).T
    )
    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.unit
def test_pair_mapper_finalizes_streaming_offsite_statistics(components):
    transform, density, model = components
    descriptor_i = torch.randn(20, density.irreps_out.dim, dtype=torch.float64)
    descriptor_j = torch.randn_like(descriptor_i)
    displacement = torch.randn(20, 3, dtype=torch.float64)
    features = model.offsite_features(descriptor_i, displacement, descriptor_j)
    targets = torch.randn(20, transform.irreps("A-B").dim, dtype=torch.float64)
    accumulator = EquivariantRidgeAccumulator(
        model.offsite_basis.irreps_out,
        transform.irreps("A-B"),
        dtype=torch.float64,
    )
    accumulator.update(features[:7], targets[:7])
    accumulator.update(features[7:], targets[7:])
    diagnostics = model.fit_offsite_from_accumulator(("A", "B"), accumulator)
    assert diagnostics.sample_count == 20
    with pytest.raises(ValueError, match="canonical"):
        model.fit_offsite_from_accumulator(("B", "A"), accumulator)
