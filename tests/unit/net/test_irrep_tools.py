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
def test_compute_irrep_metrics_zero_for_identical_blocks():
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": "1x0e + 1x1o"}))
    key = "H-H"
    dim_i, dim_j = mapper.block_dims(key)
    edges = torch.tensor(
        [[0, 1], [0, 0], [0, 0], [0, 1], [0, 1]],
        dtype=torch.long,
    )
    blocks = torch.randn(2, dim_i, dim_j)
    lookup = {tuple(edge.tolist()): (key, idx) for idx, edge in enumerate(edges.t())}
    bm = BlockMatrix(
        atoms=("H", "H"),
        atom_counts={"H": 2},
        pair_blocks={key: blocks},
        pair_edges={key: edges},
        lookup=lookup,
        orbital_cfg=mapper.orbital_cfg,
        basis="e3nn",
    )
    metrics = compute_irrep_metrics(bm, bm, get_all_irreps(mapper), mapper)

    assert metrics
    assert all(value == pytest.approx(0.0, abs=1e-8) for value in metrics.values())
