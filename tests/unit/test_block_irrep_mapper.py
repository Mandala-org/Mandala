import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper, MappingKeyError


@pytest.fixture(scope="module")
def mapper():
    cfg = OrbitalIrrepConfig.from_dict(
        {
            "H": ["1x0e"],  # 1 s orbital
            "Si": ["2x0e", "2x1o", "1x2e"],  # 13-dim basis
        }
    )
    return BlockIrrepMapper(cfg)


@pytest.mark.unit
def test_roundtrip_si_si(mapper):
    d_i, d_j = mapper.block_dims(("Si", "Si"))
    assert d_i == d_j == 13

    blocks = torch.randn(5, d_i, d_j)
    vec = mapper.blocks_to_vectors("Si-Si", blocks)
    assert vec.shape[-1] == mapper.vector_dim(("Si", "Si"))

    rec = mapper.vectors_to_blocks(("Si", "Si"), vec)
    assert torch.allclose(blocks, rec, atol=1e-6)


@pytest.mark.unit
def test_unknown_pair(mapper):
    with pytest.raises(MappingKeyError):
        mapper.vector_dim(("C", "H"))
