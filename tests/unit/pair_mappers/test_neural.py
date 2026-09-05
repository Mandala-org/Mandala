import pytest
import torch
from e3nn import o3

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.output_schema import (
    FullBlockIrrepTransform,
    o3_representation_matrix,
)
from pair_mappers.neural import FullBlockNeuralPairMapper


ARCHITECTURES = ("m0", "m2", "m3", "m5", "m7")


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(104)
    orbital_config = OrbitalIrrepConfig.from_dict({"A": "1s1p", "B": "1s1p"})
    transform = FullBlockIrrepTransform(orbital_config, dtype=torch.float64)
    descriptor_irreps = o3.Irreps("2x0e + 2x1o + 1x1e + 1x2e")
    descriptor_i = torch.randn(3, descriptor_irreps.dim, dtype=torch.float64)
    descriptor_j = torch.randn_like(descriptor_i)
    displacement = torch.randn(3, 3, dtype=torch.float64)
    return transform, descriptor_irreps, descriptor_i, descriptor_j, displacement


def make_model(architecture, transform, descriptor_irreps, dtype=torch.float64):
    return FullBlockNeuralPairMapper(
        architecture,
        transform,
        descriptor_irreps,
        bond_n_radial=2,
        bond_l_max=2,
        bond_cutoff=6.0,
        hidden_multiplicity=1,
        hidden_l_max=2,
        invariant_hidden=8,
        factorization_rank=2,
        dtype=dtype,
    )


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_complete_full_block_and_irrelevant_metadata_independence(setup, architecture):
    transform, irreps, descriptor_i, descriptor_j, displacement = setup
    model = make_model(architecture, transform, irreps)
    first = model(
        descriptor_i,
        descriptor_j,
        displacement,
        "A",
        "B",
        {"onsite": False, "irrelevant_graph": torch.randn(13)},
    )
    second = model(
        descriptor_i,
        descriptor_j,
        displacement,
        "A",
        "B",
        {"onsite": False, "irrelevant_graph": torch.randn(2, 7)},
    )
    assert first.shape == (3, transform.schema("A-B").vector_dimension)
    assert torch.equal(first, second)


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("pair", [("A", "B"), ("A", "A")])
def test_pair_reversal_is_exact(setup, architecture, pair):
    transform, irreps, descriptor_i, descriptor_j, displacement = setup
    model = make_model(architecture, transform, irreps)
    forward = model.predict_offsite(pair, descriptor_i, displacement, descriptor_j)
    reverse_pair = (pair[1], pair[0])
    reverse = model.predict_offsite(
        reverse_pair, descriptor_j, -displacement, descriptor_i
    )
    assert torch.allclose(
        reverse, transform.reverse(pair, forward), atol=1e-12, rtol=1e-12
    )


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("determinant", [1, -1])
def test_random_global_o3_equivariance(setup, architecture, determinant):
    transform, irreps, descriptor_i, descriptor_j, displacement = setup
    model = make_model(architecture, transform, irreps)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant == -1:
        rotation = -rotation
    descriptor_action = o3_representation_matrix(irreps, rotation)
    target_action = transform.output_action("A-B", rotation)
    expected = (
        model.predict_offsite(("A", "B"), descriptor_i, displacement, descriptor_j)
        @ target_action.T
    )
    actual = model.predict_offsite(
        ("A", "B"),
        descriptor_i @ descriptor_action.T,
        displacement @ rotation.T,
        descriptor_j @ descriptor_action.T,
    )
    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.unit
@pytest.mark.equivariance
def test_m7_is_independent_of_arbitrary_bond_frame_roll(setup):
    transform, irreps, descriptor_i, descriptor_j, displacement = setup
    model = make_model("m7", transform, irreps)
    reference = model.predict_offsite(
        ("A", "B"), descriptor_i, displacement, descriptor_j, roll=0.0
    )
    rolled = model.predict_offsite(
        ("A", "B"), descriptor_i, displacement, descriptor_j, roll=0.731
    )
    assert torch.allclose(rolled, reference, atol=1e-8, rtol=1e-8)


@pytest.mark.unit
@pytest.mark.equivariance
def test_m7_matches_global_cg_path_at_low_ell(setup):
    transform, irreps, descriptor_i, descriptor_j, displacement = setup
    global_model = make_model("m3", transform, irreps)
    local_model = make_model("m7", transform, irreps)
    global_kernel = global_model.offsite_kernels["A__B"]
    local_kernel = local_model.offsite_kernels["A__B"].local_kernel
    local_kernel.load_state_dict(global_kernel.state_dict())
    expected = global_model.predict_offsite(
        ("A", "B"), descriptor_i, displacement, descriptor_j
    )
    actual = local_model.predict_offsite(
        ("A", "B"), descriptor_i, displacement, descriptor_j
    )
    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_float32_forward_is_finite(setup, architecture):
    orbital_config = OrbitalIrrepConfig.from_dict({"A": "1s1p", "B": "1s1p"})
    transform = FullBlockIrrepTransform(orbital_config, dtype=torch.float32)
    irreps = setup[1]
    model = make_model(architecture, transform, irreps, dtype=torch.float32)
    descriptor_i, descriptor_j, displacement = (value.float() for value in setup[2:])
    output = model.predict_offsite(("A", "B"), descriptor_i, displacement, descriptor_j)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_m5_reduces_m3_parameter_count(setup):
    transform, irreps = setup[:2]
    m3 = make_model("m3", transform, irreps)
    m5 = make_model("m5", transform, irreps)
    assert sum(p.numel() for p in m5.parameters()) < sum(
        p.numel() for p in m3.parameters()
    )
