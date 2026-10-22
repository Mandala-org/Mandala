"""Adversarial combinations across conventions, supports and batching."""

import copy
import pytest
import torch

from data.graph_topology import prepare_graph_topology
from data.graph_features import evaluate_graph_geometry
from data.sample import union_graph_inputs
from net.pipeline import FeatureSpec, FeaturePipeline, build_transform
from net.transforms.orbital import IrrepCartesianBasis
from net.transforms.composition import (
    ScalarGatedMessage,
    IndexedAggregation,
    GaussianEnvelope,
)
from net.transforms.local_coordinates import ScalarGatedNodeUpdate
from net.training_policies import MilestoneDecay


def graph(n=3):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.2, 1.3, 0.4], [0.6, 0.4, 1.2]],
        dtype=torch.float64,
    )[:n]
    index = torch.tensor([(i, j) for i in range(n) for j in range(n)]).T
    return {
        "positions": positions,
        "box": torch.eye(3, dtype=torch.float64) * 5,
        "edge_index": index,
        "edge_shift": torch.zeros(3, index.shape[1], dtype=torch.long),
        "node_type_idx": torch.zeros(n, dtype=torch.long),
        "edge_type_idx": torch.zeros(index.shape[1], dtype=torch.long),
    }


def test_original_components_reject_noncanonical_features_and_preserve_support():
    node = FeatureSpec("node", "irreps", "2x0e+1x1o", angular_basis="xyz")
    edge = FeatureSpec("edge", "irreps", "2x0e+1x1o", support="message")
    inputs = [
        node,
        edge,
        FeatureSpec("relation", "index", support="message"),
        FeatureSpec("edge", "irreps", "0e+1o", support="message"),
        FeatureSpec("edge", irreps="2x0e", support="message"),
        FeatureSpec("edge", irreps="1x0e", support="message"),
        FeatureSpec("edge", "index", support="message"),
    ]
    options = {
        "num_species": 1,
        "irreps_out": "2x0e+1x1o",
        "config": {"l_max": 1, "n_radial": 2, "radial_layers": [3]},
    }
    _, specs = build_transform("endpoint_irrep_edge", options, inputs)
    assert specs[0].support == "message"
    assert specs[0].angular_basis == "xyz"
    from dataclasses import replace

    cyclic = [
        replace(spec, angular_basis="yzx") if spec.irreps else spec for spec in inputs
    ]
    with pytest.raises(ValueError, match="canonical xyz"):
        build_transform("endpoint_irrep_edge", options, cyclic)


def test_external_irreps_normalize_once_at_representation_boundary():
    pipeline = FeaturePipeline(
        {
            "inputs": {
                "external": {
                    "domain": "node",
                    "representation": "irreps",
                    "irreps": "0e+1o",
                    "angular_basis": "yzx",
                }
            },
            "stages": [
                {
                    "id": "identity",
                    "transform": "irrep_cartesian_basis",
                    "inputs": ["external"],
                    "outputs": ["canonical"],
                    "options": {"target": "xyz"},
                }
            ],
            "outputs": {"feature": "canonical"},
        },
        num_species=1,
    )
    values = torch.randn(4, 4, requires_grad=True)
    output, trace = pipeline({"external": values}, return_trace=True)
    expected = IrrepCartesianBasis("0e+1o", "yzx", "xyz")(values)
    torch.testing.assert_close(trace["external"], expected)
    torch.testing.assert_close(output["feature"], expected)
    assert pipeline.feature_specs["external"].angular_basis == "xyz"
    output["feature"].square().sum().backward()
    assert torch.isfinite(values.grad).all()


def test_batched_custom_relations_use_declared_entity_offsets_and_axes():
    first, second = graph(2), graph(3)
    for x in (first, second):
        prepare_graph_topology(x)
        n = len(x["positions"])
        x["custom_index"] = torch.tensor([[0, n - 1], [n - 1, 0]])
        x["mixed_relation"] = torch.tensor([[0, n - 1], [0, n * n - 1]])
        x["custom_to_output"] = torch.tensor([0, -1])
    layouts = {
        "custom_index": {"entity": "custom", "cat_dim": 1, "references": "node"},
        "mixed_relation": {
            "entity": "custom",
            "cat_dim": 1,
            "references": ["node", "output"],
        },
        "custom_to_output": {"entity": "custom", "references": "output"},
    }
    union = union_graph_inputs([(first, {}), (second, {})], layouts=layouts)
    torch.testing.assert_close(
        union["custom_index"], torch.tensor([[0, 1, 2, 4], [1, 0, 4, 2]])
    )
    torch.testing.assert_close(
        union["mixed_relation"], torch.tensor([[0, 1, 2, 4], [0, 3, 4, 12]])
    )
    assert union["custom_to_output"].tolist() == [0, -1, 4, -1]
    with pytest.raises(ValueError, match="explicit tensor layout"):
        union_graph_inputs([(first, {}), (second, {})])


def test_feature_contract_exposes_custom_relation_layout_to_collator():
    pipeline = FeaturePipeline(
        {
            "inputs": {
                "short_edges": {
                    "domain": "relation",
                    "representation": "index",
                    "support": "short",
                    "batch": {"cat_dim": 1, "references": "node"},
                }
            },
            "stages": [],
            "outputs": {"indices": "short_edges"},
        },
        num_species=1,
    )
    a, b = graph(2), graph(3)
    for x in (a, b):
        x["short_edges"] = torch.tensor([[0], [1]])
    union = union_graph_inputs([(a, {}), (b, {})], layouts=pipeline.input_layouts)
    assert union["short_edges"].tolist() == [[0, 2], [1, 3]]


def test_explicit_geometry_stages_preserve_support_and_coordinate_gradients():
    x = graph()
    prepare_graph_topology(
        x, {"message": {"kind": "subset", "source": "output", "cutoff": 1.2}}
    )
    pipeline = FeaturePipeline(
        {
            "stages": [
                {
                    "id": "geometry",
                    "transform": "periodic_displacement",
                    "inputs": [
                        "positions",
                        "box",
                        "message_edge_index",
                        "message_shift",
                        "node_batch",
                    ],
                    "outputs": ["vectors"],
                },
                {
                    "id": "lengths",
                    "transform": "vector_lengths",
                    "inputs": ["vectors"],
                    "outputs": ["lengths"],
                },
            ],
            "outputs": {"distance": "lengths"},
        },
        num_species=1,
    )
    x["positions"].requires_grad_()
    edges = x["message_edge_index"].clone()
    distance = pipeline(x)["distance"]
    gradient = torch.autograd.grad(distance.square().sum(), x["positions"])[0]
    assert gradient.abs().sum() > 0
    assert torch.equal(x["message_edge_index"], edges)


@pytest.mark.parametrize("permutation", [[0, 0, 1], [0, 1]])
def test_canonical_harmonic_compilation_does_not_hide_invalid_options(permutation):
    with pytest.raises(ValueError, match="permutation"):
        FeaturePipeline(
            {
                "stages": [
                    {
                        "id": "sh",
                        "transform": "spherical_harmonics",
                        "inputs": ["edge_vectors"],
                        "outputs": ["sh"],
                        "options": {"irreps": "0e+1o", "permutation": permutation},
                    }
                ],
                "outputs": {"features": "sh"},
            },
            num_species=1,
        )


def test_geometry_and_derivatives_preserve_selected_support_even_after_motion():
    x = graph()
    policy = {
        "message": {
            "kind": "subset",
            "source": "output",
            "cutoff": 1.2,
            "exclude_self": True,
        }
    }
    prepare_graph_topology(x, policy)
    topology = x["message_edge_index"].clone()
    assert topology.shape[1] < 6
    x["positions"] = (x["positions"] * 2).requires_grad_()
    x["box"].requires_grad_()
    evaluate_graph_geometry(x, recompute_geometry=True)
    assert torch.equal(x["message_edge_index"], topology)
    energy = x["message_lengths"].square().sum()
    force, stress = torch.autograd.grad(energy, (x["positions"], x["box"]))
    expected = torch.zeros_like(force)
    displacements = x["positions"][topology[1]] - x["positions"][topology[0]]
    expected.index_add_(0, topology[0], -2 * displacements)
    expected.index_add_(0, topology[1], 2 * displacements)
    torch.testing.assert_close(force, expected)
    assert torch.isfinite(stress).all()
    from data.graph_topology import rebuild_graph_supports

    changed_policy = {
        "message": {
            "kind": "subset",
            "source": "output",
            "exclude_self": True,
            "cutoff": 5.0,
        }
    }
    with pytest.raises(ValueError, match="explicitly rebuild"):
        prepare_graph_topology(x, changed_policy)
    rebuilt = rebuild_graph_supports(x, changed_policy)
    assert rebuilt["message_edge_index"].shape[1] > topology.shape[1]
    assert torch.equal(x["message_edge_index"], topology)


def test_independent_radius_support_retains_periodic_images_and_batches():
    x = graph(2)
    x["box"] = torch.eye(3, dtype=torch.float64) * 2
    # Output consists only of onsite blocks, messages have independent support.
    x["edge_index"] = torch.tensor([[0, 1], [0, 1]])
    x["edge_shift"] = torch.zeros(3, 2, dtype=torch.long)
    x["edge_type_idx"] = torch.zeros(2, dtype=torch.long)
    prepare_graph_topology(x, {"message": {"kind": "radius", "cutoff": 1.5}})
    assert (x["message_to_output"] == -1).all()
    assert x["message_shift"].any()
    evaluate_graph_geometry(x)
    union = union_graph_inputs([(x, {}), (copy.deepcopy(x), {})])
    assert union["message_edge_index"].shape[1] == 2 * x["message_edge_index"].shape[1]
    evaluate_graph_geometry(union, recompute_geometry=True)
    torch.testing.assert_close(
        union["message_vectors"], x["message_vectors"].repeat(2, 1)
    )


def test_scalar_message_envelope_and_reducer_compose_to_reference_update():
    composite = ScalarGatedNodeUpdate(3, 2, normalization="none").double()
    message = ScalarGatedMessage(8, 3).double()
    message.load_state_dict(
        {
            key: value
            for key, value in composite.state_dict().items()
            if key.startswith(("lin_f.", "lin_s."))
        }
    )
    node, edge = torch.randn(3, 3, dtype=torch.float64), torch.randn(
        4, 2, dtype=torch.float64
    )
    index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    lengths = torch.rand(4, dtype=torch.float64)
    z = torch.cat((node[index[1]], node[index[0]], edge), -1)
    values = message(z) * GaussianEnvelope()(lengths)
    expected = node + IndexedAggregation()(values, index, node)
    observed = composite(node, index, edge, torch.zeros(3, dtype=torch.long), lengths)
    torch.testing.assert_close(observed, expected)


def test_milestone_policy_matches_source_epoch_order_and_resumes():
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([p], lr=0.001)
    scheduler = MilestoneDecay(optimizer, [500, 2000, 3000], [0.4, 0.5, 0.5])
    trajectory = []
    for epoch in range(3002):
        trajectory.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        if epoch == 1999:
            opt_state, state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(
                scheduler.state_dict()
            )
    assert trajectory[500] == 0.001
    assert trajectory[501] == pytest.approx(0.0004)
    assert trajectory[2001] == pytest.approx(0.0002)
    assert trajectory[3001] == pytest.approx(0.0001)
    resumed = torch.optim.Adam([p], lr=0.001)
    continuation = MilestoneDecay(resumed, [500, 2000, 3000], [0.4, 0.5, 0.5])
    resumed.load_state_dict(opt_state)
    continuation.load_state_dict(state)
    for epoch in range(2000, 3002):
        assert resumed.param_groups[0]["lr"] == trajectory[epoch]
        resumed.step()
        continuation.step()
