"""Targeted contracts and optional executable-source differential checks.

Set MANDALA_MACEH_REFERENCE_ROOT to a checkout of Xiaoxun-Gong/MACE-H at
3d6248b715eef302a4fd674d52bcd69254596a51 to run the upstream oracle checks.
The optional checkout is used only by tests, never by production inference.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
from functools import lru_cache

import pytest
import torch
from e3nn import o3

from net.pipeline import FeatureSpec
from net.transforms.cluster import (
    AngularDegreeExpansion,
    ChannelIrrepReshape,
    CutoffRadialExpansion,
    NeighborDensityInteraction,
    SymmetricProductContraction,
    angular_expansion_projection_irreps,
    build_cluster_transform,
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


def _density_kwargs(node_irreps="2x0e"):
    return dict(
        node_attrs_irreps=o3.Irreps("2x0e"),
        node_feats_irreps=o3.Irreps(node_irreps),
        edge_attrs_irreps=o3.Irreps("0e+1o"),
        edge_feats_irreps=o3.Irreps("3x0e"),
        target_irreps=o3.Irreps("2x0e+2x1o"),
        hidden_irreps=o3.Irreps("2x0e+2x1o"),
        avg_num_neighbors=2.0,
    )


@lru_cache(None)
def _upstream():
    location = os.getenv("MANDALA_MACEH_REFERENCE_ROOT")
    if not location:
        pytest.skip(
            "Pinned upstream differential checks need MANDALA_MACEH_REFERENCE_ROOT."
        )
    package_dir = Path(location) / "maceh"
    if not (package_dir / "from_mfn" / "blocks.py").is_file():
        raise ValueError(
            "Reference root must contain the MACE-H maceh/ source package."
        )
    package = types.ModuleType("_mandala_maceh_oracle")
    package.__path__ = [str(package_dir)]
    sys.modules[package.__name__] = package
    blocks = importlib.import_module(package.__name__ + ".from_mfn.blocks")
    expansions = importlib.import_module(package.__name__ + ".macehmodules")
    return blocks, expansions


def _compare_parameters_and_state(left, right):
    assert list(left.state_dict()) == list(right.state_dict())
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


def _compare_step(reference, actual, args, reference_args=None):
    """Compare native outputs, input/parameter gradients and one Adam step."""

    def leaves(values):
        return [
            x.detach().clone().requires_grad_(x.is_floating_point()) for x in values
        ]

    reference_values, actual_values = leaves(reference_args or args), leaves(args)
    expected, observed = reference(*reference_values), actual(*actual_values)
    if isinstance(observed, dict):
        observed = tuple(observed.values())
    if isinstance(expected, torch.Tensor):
        expected, observed = (expected,), (observed,)
    assert len(expected) == len(observed)
    for before, after in zip(expected, observed):
        torch.testing.assert_close(before, after, rtol=2e-12, atol=2e-12)
    reference_loss = sum(x.square().mean() + x.sum() * 0.031 for x in expected)
    actual_loss = sum(x.square().mean() + x.sum() * 0.031 for x in observed)
    reference_loss.backward()
    actual_loss.backward()
    for before, after in zip(reference_values, actual_values):
        if before.requires_grad:
            assert (before.grad is None) == (after.grad is None)
            if before.grad is not None:
                torch.testing.assert_close(
                    before.grad, after.grad, rtol=1e-10, atol=1e-11
                )
    assert list(dict(reference.named_parameters())) == list(
        dict(actual.named_parameters())
    )
    for name, before in reference.named_parameters():
        after = dict(actual.named_parameters())[name]
        assert (before.grad is None) == (after.grad is None), name
        if before.grad is not None:
            torch.testing.assert_close(
                before.grad, after.grad, rtol=1e-10, atol=1e-11, msg=name
            )
    torch.optim.Adam(reference.parameters(), lr=1e-3).step()
    torch.optim.Adam(actual.parameters(), lr=1e-3).step()
    for name, before in reference.named_parameters():
        torch.testing.assert_close(
            before,
            dict(actual.named_parameters())[name],
            rtol=1e-10,
            atol=1e-11,
            msg=name,
        )


@pytest.mark.parametrize("normalization", ["none", "graph_component"])
@pytest.mark.parametrize("emit_expansion", [False, True])
def test_density_seeded_initialization_outputs_gradients_and_adam_match_source(
    normalization, emit_expansion
):
    blocks, _ = _upstream()
    suffix = ("Up" if emit_expansion else "") + (
        "Norm" if normalization != "none" else ""
    )
    reference_cls = getattr(blocks, "RealAgnosticResidualInteractionBlock" + suffix)
    kwargs = _density_kwargs("2x0e+2x1o")
    torch.manual_seed(735)
    reference = reference_cls(**kwargs, radial_MLP=[4, 4])
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(735)
    actual = NeighborDensityInteraction(
        **kwargs,
        radial_layers=[4, 4],
        normalization=normalization,
        emit_expansion=emit_expansion,
    )
    assert torch.equal(expected_rng, torch.get_rng_state())
    _compare_parameters_and_state(reference, actual)
    args = _inputs("2x0e+2x1o")
    _compare_step(reference, actual, args if normalization != "none" else args[:5])


@pytest.mark.parametrize("correlation", [1, 2, 3])
def test_symmetric_product_seeded_initialization_outputs_gradients_and_adam_match_source(
    correlation,
):
    blocks, _ = _upstream()
    irreps = o3.Irreps("2x0e+2x1o")
    torch.manual_seed(915)
    reference = blocks.EquivariantProductBasisBlock(
        irreps, irreps, correlation, num_elements=2
    )
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(915)
    actual = SymmetricProductContraction(irreps, irreps, correlation, num_elements=2)
    assert torch.equal(expected_rng, torch.get_rng_state())
    _compare_parameters_and_state(reference, actual)
    generator = torch.Generator().manual_seed(81)
    density = torch.randn(4, 2, 4, generator=generator)
    attrs = _inputs()[0]
    skip = torch.randn(4, irreps.dim, generator=generator)

    # The shared pipeline uses density/attrs/skip, the original block density/skip/attrs.
    class _OriginalOrder(torch.nn.Module):
        def forward(self, value, attributes, connection):
            return reference(value, connection, attributes)

    wrapper = _OriginalOrder()
    wrapper._parameters = reference._parameters
    wrapper._modules = reference._modules
    _compare_step(wrapper, actual, (density, attrs, skip))


@pytest.mark.parametrize("normalization", ["none", "graph_component"])
@pytest.mark.parametrize("nonlinearity", [False, True])
def test_angular_expansion_seeded_initialization_outputs_gradients_and_adam_match_source(
    normalization, nonlinearity
):
    _, expansions = _upstream()
    irreps = o3.Irreps("2x0e+2x1o")
    kwargs = dict(
        hidden_irreps=irreps,
        hidden_post_irreps=irreps,
        up_irreps=irreps,
        up_post_irreps=irreps,
    )
    torch.manual_seed(37)
    reference = expansions.NodeDegreeExpansionBlock(
        **kwargs,
        irreps_post_node=irreps,
        num_species=2,
        expand_nonlin=nonlinearity,
        norm="e3LayerNorm" if normalization != "none" else "",
    )
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(37)
    actual = AngularDegreeExpansion(
        **kwargs,
        irreps_out=irreps,
        num_species=2,
        nonlinearity=nonlinearity,
        normalization=normalization,
    )
    assert torch.equal(expected_rng, torch.get_rng_state())
    _compare_parameters_and_state(reference, actual)
    generator = torch.Generator().manual_seed(91)
    args = (
        torch.randn(4, irreps.dim, generator=generator),
        torch.randn(4, irreps.dim, generator=generator),
        _inputs()[0],
        torch.tensor([0, 0, 1, 1]),
    )
    _compare_step(reference, actual, args)


def test_density_products_are_rotation_covariant_and_differentiable():
    torch.manual_seed(42)
    interaction = NeighborDensityInteraction(
        **_density_kwargs(), radial_layers=[4], emit_expansion=True
    )
    product = SymmetricProductContraction("2x0e+2x1o", "2x0e+2x1o", 3, 2)
    attrs, hidden, harmonics, radial, edge_index, batch = _inputs()
    rotation = o3.rand_matrix()
    d_edge = o3.Irreps("0e+1o").D_from_matrix(rotation)
    d_hidden = o3.Irreps("2x0e+2x1o").D_from_matrix(rotation)
    original = interaction(attrs, hidden, harmonics, radial, edge_index, batch)
    rotated = interaction(
        attrs, hidden, harmonics @ d_edge.T, radial, edge_index, batch
    )
    before = product(original["density"], attrs, original["skip"])
    after = product(rotated["density"], attrs, rotated["skip"])
    torch.testing.assert_close(after, before @ d_hidden.T, atol=2e-8, rtol=2e-8)
    before.square().sum().backward()
    for parameter in list(interaction.parameters()) + list(product.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_empty_neighbor_set_preserves_isolated_nodes_and_raw_species_skip():
    interaction = NeighborDensityInteraction(**_density_kwargs(), radial_layers=[4])
    attrs, hidden, _, _, _, _ = _inputs()
    output = interaction(
        attrs,
        hidden,
        torch.empty(0, 4),
        torch.empty(0, 3),
        torch.empty(2, 0, dtype=torch.long),
    )
    assert output["density"].shape == (4, 2, 4)
    assert torch.count_nonzero(output["density"]) == 0
    torch.testing.assert_close(output["skip"], interaction.skip_tp(hidden, attrs))


def test_density_layout_and_frame_contracts_reject_invalid_mixes():
    with pytest.raises(ValueError, match="equal irrep multiplicities"):
        ChannelIrrepReshape("2x0e+3x1o")
    with pytest.raises(ValueError, match="channel-first global node density"):
        build_cluster_transform(
            "symmetric_product",
            {"irreps_out": "2x0e", "correlation": 2},
            [
                FeatureSpec("node", "local_components", frame="bond", width=2),
                FeatureSpec("node", irreps="2x0e"),
            ],
        )
    with pytest.raises(ValueError, match="scalar channel count"):
        SymmetricProductContraction("2x0e+2x1o", "3x0e", 2, 2)


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
