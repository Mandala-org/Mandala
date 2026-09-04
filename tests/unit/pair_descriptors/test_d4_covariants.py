import pytest
import torch
from e3nn import o3

from pair_descriptors import DeterministicCovariantLiftDescriptor


TARGET_IRREPS = ((0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (4, 1))


def _descriptor(body_degree: int = 3):
    return DeterministicCovariantLiftDescriptor(
        (8, 14),
        radial_basis="spherical_bessel",
        n_radial=2,
        l_max=2,
        cutoff=5.0,
        body_degree=body_degree,
        lift_max_input_l=2,
        lift_max_radial_index=1,
        lift_radial_degree_budget=2,
        lift_max_intermediate_l=3,
        target_irreps=TARGET_IRREPS,
        max_paths_per_degree_irrep=8,
    )


def _action(irreps, matrix):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return irreps.D_from_matrix(matrix)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.unit
def test_d4_retains_d1_and_manifest_is_deterministic():
    first = _descriptor()
    second = _descriptor()
    vectors = torch.randn(7, 3, dtype=torch.float64)
    species = torch.tensor([8, 14, 8, 8, 14, 8, 14])
    raw = first.base(vectors, species)
    actual = first(vectors, species)
    assert torch.equal(actual[:, : raw.shape[-1]], raw)
    assert first.metadata["content_hash"] == second.metadata["content_hash"]
    assert first.metadata["paths"] == second.metadata["paths"]
    assert any(path.body_degree == 2 for path in first.paths)
    assert any(path.body_degree == 3 for path in first.paths)


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("determinant", [1, -1])
def test_d4_is_o3_equivariant(determinant):
    torch.manual_seed(82)
    descriptor = _descriptor()
    vectors = torch.randn(7, 3, dtype=torch.float64)
    species = torch.tensor([8, 14, 8, 8, 14, 8, 14])
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant == -1:
        rotation = -rotation
    actual = descriptor(vectors @ rotation.T, species)
    expected = descriptor(vectors, species) @ _action(descriptor.irreps_out, rotation).T
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )
    assert relative < 1e-9


@pytest.mark.unit
def test_d4_puncture_rebuilds_nonadditive_lifts_exactly():
    descriptor = _descriptor()
    vectors = torch.tensor(
        [[0.8, -0.1, 0.2], [0.2, 1.1, -0.4], [-0.4, 0.1, 1.3]],
        dtype=torch.float64,
    )
    species = torch.tensor([8, 14, 8])
    full = descriptor(vectors, species)
    punctured = descriptor.puncture(full, species[:1], vectors[:1])
    direct = descriptor(vectors[1:], species[1:])
    assert torch.allclose(punctured, direct, atol=2e-12, rtol=2e-12)


@pytest.mark.unit
def test_d4_equal_species_neighbor_permutation_is_deterministic():
    descriptor = _descriptor(body_degree=2)
    vectors = torch.randn(8, 3, dtype=torch.float64)
    species = torch.tensor([8, 8, 14, 8, 14, 14, 8, 14])
    permutation = torch.tensor([6, 1, 3, 0, 7, 4, 5, 2])
    assert torch.allclose(
        descriptor(vectors, species),
        descriptor(vectors[permutation], species[permutation]),
        atol=2e-14,
        rtol=2e-14,
    )
