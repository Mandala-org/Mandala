import pytest
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.irrep_tools import (
    compute_irrep_metrics,
    build_irrep_projector_cache,
    get_all_irreps,
    project_irrep_vectors_to_blocks,
)


@pytest.mark.unit
def test_irrep_projectors_reconstruct_blocks():
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": "1x0e + 1x1o"}))
    all_irreps = get_all_irreps(mapper)
    cache = build_irrep_projector_cache(mapper, all_irreps)
    key = "H-H"

    dim_i, dim_j = mapper.block_dims(key)
    target_blocks = torch.randn(4, dim_i, dim_j)
    target_irreps = mapper.blocks_to_vectors(key, target_blocks)

    reconstructed = torch.zeros_like(target_blocks)
    for irrep in all_irreps:
        irrep_key = str(irrep)
        if key not in cache[irrep_key]:
            continue
        reconstructed = reconstructed + project_irrep_vectors_to_blocks(
            target_irreps,
            cache[irrep_key][key],
            mapper,
        )

    assert torch.allclose(reconstructed, target_blocks, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_compute_irrep_metrics_match_hand_computed_nonzero_error_and_weighting():
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": "1x0e + 1x1o"}))
    key = "H-H"
    edges = torch.tensor(
        [[0, 1], [0, 0], [0, 0], [0, 1], [0, 1]],
        dtype=torch.long,
    )
    lookup = {tuple(edge.tolist()): (key, idx) for idx, edge in enumerate(edges.t())}
    all_irreps = get_all_irreps(mapper)
    pair_irreps = mapper.get_pair_irreps(key)
    widths = {
        str(irrep): sum(
            mul * ir.dim for mul, ir in pair_irreps if str(ir) == str(irrep)
        )
        for irrep in all_irreps
    }
    # Every target channel is one in both blocks. The two predictions add
    # offsets 1 and 2, giving closed-form absolute and relative metrics.
    target_vectors = torch.ones(2, mapper.vector_dim(key))
    predicted_vectors = target_vectors + torch.tensor([1.0, 2.0])[:, None]
    target_blocks = mapper.vectors_to_blocks(key, target_vectors)
    predicted_blocks = mapper.vectors_to_blocks(key, predicted_vectors)

    def make_matrix(blocks):
        return BlockMatrix(
            atoms=("H", "H"),
            atom_counts={"H": 2},
            pair_blocks={key: blocks},
            pair_edges={key: edges},
            lookup=lookup,
            orbital_cfg=mapper.orbital_cfg,
            basis="e3nn",
        )

    metrics = compute_irrep_metrics(
        make_matrix(predicted_blocks), make_matrix(target_blocks), all_irreps, mapper
    )
    expected_relative = (1.0**2 + 2.0**2) ** 0.5 / 2**0.5
    for irrep in all_irreps:
        width = widths[str(irrep)]
        assert width > 0
        assert metrics[f"{irrep}_l2_block_abs"] == pytest.approx(
            width**0.5 * expected_relative
        )
        assert metrics[f"{irrep}_l2_block_rel"] == pytest.approx(expected_relative)
