import pytest
import torch
from e3nn import o3

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.output_schema import FullBlockIrrepTransform


@pytest.fixture(scope="module")
def transform() -> FullBlockIrrepTransform:
    config = OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"})
    return FullBlockIrrepTransform(config, dtype=torch.float64)


@pytest.mark.unit
def test_full_block_schema_is_complete_and_has_pseudo_irreps(transform):
    schema = transform.schema("O-Si")
    assert schema.row_dimension == schema.column_dimension == 13
    assert schema.vector_dimension == 169
    assert transform.irreps("O-Si").dim == 169
    assert any(copy.irrep_label == "1e" for copy in schema.copies)
    assert len(schema.content_hash) == 64


@pytest.mark.unit
def test_shell_pair_provenance_covers_each_matrix_coordinate_once(transform):
    schema = transform.schema("O-Si")
    coverage = torch.zeros(
        schema.row_dimension, schema.column_dimension, dtype=torch.int64
    )
    for row_shell in schema.row_shells:
        for column_shell in schema.column_shells:
            coverage[
                row_shell.ao_start : row_shell.ao_stop,
                column_shell.ao_start : column_shell.ao_stop,
            ] += 1
    assert torch.equal(coverage, torch.ones_like(coverage))


@pytest.mark.unit
@pytest.mark.parametrize("pair", ["O-O", "O-Si", "Si-O", "Si-Si"])
def test_float64_roundtrip_and_norm_isometry(transform, pair):
    generator = torch.Generator().manual_seed(1234)
    block = torch.randn(7, 13, 13, generator=generator, dtype=torch.float64)
    vector = transform.blocks_to_irreps(pair, block)
    restored = transform.irreps_to_blocks(pair, vector)
    relative = torch.linalg.vector_norm(restored - block) / torch.linalg.vector_norm(
        block
    )
    assert relative < 1.0e-12
    assert torch.allclose(
        torch.sum(block.square(), dim=(-2, -1)),
        torch.sum(vector.square(), dim=-1),
        atol=1.0e-11,
        rtol=1.0e-11,
    )


@pytest.mark.unit
@pytest.mark.parametrize("pair", ["O-O", "O-Si", "Si-O", "Si-Si"])
def test_pair_reversal_is_derived_from_exact_transpose(transform, pair):
    generator = torch.Generator().manual_seed(5678)
    block = torch.randn(5, 13, 13, generator=generator, dtype=torch.float64)
    vector = transform.blocks_to_irreps(pair, block)
    reversed_vector = transform.reverse(pair, vector)
    expected = transform.blocks_to_irreps(
        tuple(reversed(pair.split("-"))), block.transpose(-1, -2)
    )
    assert torch.allclose(reversed_vector, expected, atol=1.0e-12, rtol=1.0e-12)
    restored = transform.reverse(tuple(reversed(pair.split("-"))), reversed_vector)
    assert torch.allclose(restored, vector, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.unit
@pytest.mark.equivariance
@pytest.mark.parametrize("determinant", [1, -1])
def test_full_o3_action_commutes_with_target_transform(transform, determinant):
    generator = torch.Generator().manual_seed(9012 + determinant)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if determinant == -1:
        rotation = -rotation
    assert round(torch.linalg.det(rotation).item()) == determinant

    orbital_action = transform.orbital_action("O", rotation)
    output_action = transform.output_action("O-Si", rotation)
    block = torch.randn(13, 13, generator=generator, dtype=torch.float64)
    rotated_block = orbital_action @ block @ orbital_action.T
    actual = transform.blocks_to_irreps("O-Si", rotated_block)
    expected = output_action @ transform.blocks_to_irreps("O-Si", block)
    error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )
    assert error < 1.0e-10
