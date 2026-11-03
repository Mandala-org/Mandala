"""Targeted contracts of the MACE stages and blocks.

The decomposed MACE-H blocks are compared against the pinned upstream source
through ``tests/fixtures/reference_operators`` (whole default model) and
against the former monolithic layers through
``tests/physics/test_reference_decomposition.py`` (every layer variant).
"""

from __future__ import annotations


import pytest
import torch
from e3nn import o3

from net.pipeline import FeaturePipeline, FeatureSpec, build_transform
from net.transforms.cluster import (
    ChannelIrrepReshape,
    CutoffRadialExpansion,
    angular_expansion_projection_irreps,
)


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _inputs(node_irreps="2x0e", seed=123):
    generator = torch.Generator().manual_seed(seed)
    attrs = torch.nn.functional.one_hot(torch.tensor([0, 1, 0, 1]), 2).double()
    nodes = torch.randn(4, o3.Irreps(node_irreps).dim, generator=generator)
    edges = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]])
    vectors = torch.randn(6, 3, generator=generator)
    harmonics = o3.spherical_harmonics(
        "0e+1o", vectors, normalize=True, normalization="component"
    )
    radial = torch.randn(6, 3, generator=generator)
    return attrs, nodes, harmonics, radial, edges, torch.zeros(4, dtype=torch.long)


def _density_pipeline(**options):
    """Interaction, separate species skip and product, as in the MACE-H recipe."""
    from net.block_library import library_blocks

    stages = [
        {
            "id": "density",
            "transform": "density_interaction",
            "inputs": ["hidden", "harmonics", "radial", "edge_index", "node_batch"],
            "outputs": ["density", "message"],
            "options": {
                "irreps_out": "2x0e+2x1o",
                "radial_layers": [4],
                "graph_norm": False,
                "degree_scale": True,
                "avg_num_neighbors": 2.0,
                **options,
            },
        },
        {
            "id": "skip",
            "transform": "species_tensor_product",
            "inputs": ["hidden", "species"],
            "outputs": ["skip"],
            "options": {"irreps_out": "2x0e+2x1o"},
        },
        {
            "id": "product",
            "transform": "symmetric_product",
            "inputs": ["density", "species", "skip"],
            "outputs": ["product"],
            "options": {"irreps_out": "2x0e+2x1o", "correlation": 3},
        },
    ]
    return FeaturePipeline(
        {
            "blocks": library_blocks(["mace"]),
            "inputs": {
                "species": {"domain": "node", "irreps": "2x0e", "width": 2},
                "hidden": {"domain": "node", "irreps": "2x0e"},
                "harmonics": {"domain": "edge", "irreps": "0e+1o"},
                "radial": {"domain": "edge", "irreps": "3x0e"},
            },
            "stages": stages,
            "outputs": {"density": "density", "skip": "skip", "product": "product"},
        },
        num_species=2,
    )


def _run(model, attrs, hidden, harmonics, radial, edge_index, batch):
    return model(
        {
            "species": attrs,
            "hidden": hidden,
            "harmonics": harmonics,
            "radial": radial,
            "edge_index": edge_index,
            "node_batch": batch,
        }
    )


def test_density_products_are_rotation_covariant_and_differentiable():
    torch.manual_seed(42)
    model = _density_pipeline()
    attrs, hidden, harmonics, radial, edge_index, batch = _inputs()
    rotation = o3.rand_matrix()
    d_edge = o3.Irreps("0e+1o").D_from_matrix(rotation)
    d_hidden = o3.Irreps("2x0e+2x1o").D_from_matrix(rotation)
    before = _run(model, attrs, hidden, harmonics, radial, edge_index, batch)
    after = _run(model, attrs, hidden, harmonics @ d_edge.T, radial, edge_index, batch)
    torch.testing.assert_close(
        after["product"], before["product"] @ d_hidden.T, atol=2e-8, rtol=2e-8
    )
    before["product"].square().sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_empty_neighbor_set_preserves_isolated_nodes_and_raw_species_skip():
    model = _density_pipeline()
    attrs, hidden, _, _, _, batch = _inputs()
    output = _run(
        model,
        attrs,
        hidden,
        torch.empty(0, 4),
        torch.empty(0, 3),
        torch.empty(2, 0, dtype=torch.long),
        batch,
    )
    assert output["density"].shape == (4, 2, 4)
    assert torch.count_nonzero(output["density"]) == 0
    torch.testing.assert_close(output["skip"], model.transforms["skip"](hidden, attrs))


def test_density_layout_and_frame_contracts_reject_invalid_mixes():
    with pytest.raises(ValueError, match="equal irrep multiplicities"):
        ChannelIrrepReshape("2x0e+3x1o")
    with pytest.raises(ValueError, match="channel-first global node density"):
        build_transform(
            "symmetric_contraction",
            {"irreps_out": "2x0e", "correlation": 2},
            [
                FeatureSpec("node", "local_components", frame="bond", width=2),
                FeatureSpec("node", irreps="2x0e"),
            ],
        )
    with pytest.raises(ValueError, match="scalar channel count"):
        build_transform(
            "symmetric_contraction",
            {"irreps_out": "3x0e", "correlation": 2},
            [
                FeatureSpec("node", "channel_irreps", "1x0e+1x1o", width=2),
                FeatureSpec("node", irreps="2x0e"),
            ],
        )


def test_radial_cutoff_matches_native_source_and_has_nonzero_length_gradients():
    radial = CutoffRadialExpansion(4.0, num_basis=3, cutoff_order=5)
    distances = torch.tensor([[0.5], [3.9], [4.0], [4.1]], requires_grad=True)
    values = radial(distances)
    assert values.shape == (4, 3)
    assert torch.count_nonzero(values[2:]) == 0
    values.sum().backward()
    assert torch.isfinite(distances.grad).all()
    assert distances.grad[0].abs() > 0


def test_expansion_projection_policy_retains_reachable_high_angular_channels():
    output = angular_expansion_projection_irreps("2x0e+2x1o+2x2e", "3x0e+3x1o")
    assert output == o3.Irreps("3x0e+3x1o+3x2e")
