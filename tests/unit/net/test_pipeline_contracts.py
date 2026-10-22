"""Index-set, frame and output-layout contracts for composable stages."""

from __future__ import annotations

import pytest
import torch

from core.operator_layout import ShellTensorCodec
from net.pipeline import (
    FeaturePipeline,
    FeatureSpec,
    build_transform,
    default_input_specs,
)


def test_builtins_distinguish_message_and_output_edge_index_sets():
    specs = default_input_specs(2)
    assert specs["edge_index"].support == specs["edge_vectors"].support == "output"
    assert (
        specs["message_edge_index"].support
        == specs["message_vectors"].support
        == "message"
    )
    assert specs["message_lengths"].support == "message"


@pytest.mark.parametrize(
    "transform,options,representation",
    [
        ("gaussian_basis", {"start": 0.0, "stop": 4.0, "n_gaussians": 3}, "scalar"),
        ("cutoff_radial_expansion", {"r_max": 4.0, "num_basis": 3}, "scalar"),
        ("spherical_harmonics", {"irreps": "0e+1o"}, "geometry"),
    ],
)
def test_geometric_expansions_preserve_edge_support(transform, options, representation):
    spec = FeatureSpec(
        "edge",
        representation,
        width=3 if representation == "geometry" else 1,
        support="message",
    )
    _, produced = build_transform(transform, options, [spec])
    assert produced[0].support == "message"


@pytest.mark.parametrize("transform", ["add", "concat"])
def test_equal_shape_features_from_distinct_edge_supports_cannot_be_combined(transform):
    with pytest.raises(ValueError, match="support mismatch"):
        build_transform(
            transform,
            {},
            [
                FeatureSpec("edge", irreps="2x0e"),
                FeatureSpec("edge", irreps="2x0e", support="message"),
            ],
        )


def test_linear_layers_preserve_edge_support():
    spec = FeatureSpec("edge", "irreps", "2x0e", support="message")
    for transform, options in (
        ("irrep_linear", {"irreps_out": "3x0e"}),
        ("scalar_linear", {"out_features": 3}),
    ):
        _, produced = build_transform(transform, options, [spec])
        assert produced[0].support == "message"


@pytest.mark.parametrize(
    "name", ["endpoint_tensor_product_node", "endpoint_tensor_product_edge"]
)
def test_endpoint_tensor_products_reject_wrong_edge_index_support(name):
    node = FeatureSpec("node", "irreps", "2x0e")
    attrs = FeatureSpec("node" if name.endswith("node") else "edge", irreps="2x0e")
    edge = FeatureSpec("edge", "irreps", "2x0e")
    specs = [
        node,
        attrs,
        edge,
        edge,
        edge,
        FeatureSpec("relation", "index", frame="none", support="message"),
        FeatureSpec("node", "index", frame="none"),
    ]
    with pytest.raises(ValueError, match="support mismatch"):
        build_transform(name, {}, specs)


def test_neighbor_density_rejects_equal_width_radial_features_from_another_graph():
    specs = [
        FeatureSpec("node", irreps="2x0e"),
        FeatureSpec("node", "irreps", "2x0e"),
        FeatureSpec("edge", "irreps", "0e+1o", support="message"),
        FeatureSpec("edge", irreps="3x0e"),
        FeatureSpec("relation", "index", frame="none", support="message"),
    ]
    with pytest.raises(ValueError, match="support mismatch"):
        build_transform("neighbor_density", {}, specs)


def test_graph_norm_rejects_batch_indices_from_another_edge_support():
    with pytest.raises(ValueError, match="support mismatch"):
        build_transform(
            "graph_irrep_norm",
            {},
            [
                FeatureSpec("edge", "irreps", "2x0e", support="message"),
                FeatureSpec("edge", "index", frame="none"),
            ],
        )


def test_local_message_rejects_bond_neighborhoods_for_another_output_graph():
    specs = default_input_specs(2)
    ports = [
        specs["node_one_hot"],
        specs["edge_one_hot"],
        specs["sub_atom_idx"],
        specs["sub_edge_idx"],
        FeatureSpec(
            "bond_neighbor", "local_components", frame="bond", support="message"
        ),
        specs["sub_index"],
        specs["edge_lengths"],
    ]
    with pytest.raises(ValueError, match="support mismatch"):
        build_transform(
            "local_coordinate_message", {"angular_dim": 4, "out_dim": 2}, ports
        )


def test_local_orbital_output_spinless_identity_and_bond_frame_contract():
    source = FeatureSpec("edge", "local_components", frame="bond", width=3)
    module, output = build_transform(
        "local_orbital_output", {"packed_dim": 3}, [source]
    )
    value = torch.randn(2, 3)
    assert module(value) is value
    assert output == [FeatureSpec("edge", "matrix", frame="bond", width=3)]
    with pytest.raises(ValueError, match="bond-frame"):
        build_transform(
            "local_orbital_output",
            {"packed_dim": 3},
            [FeatureSpec("edge", irreps="3x0e")],
        )


def test_local_orbital_soc_output_matches_original_reader_component_order_and_gradients():
    # Native reader packs per orbital: uu.real/imag, dd.real/imag, ud.real/imag, du.real/imag.
    source = FeatureSpec("edge", "local_components", frame="bond", width=16)
    module, output = build_transform(
        "local_orbital_output", {"packed_dim": 2, "spinful": True}, [source]
    )
    value = torch.arange(16, dtype=torch.float64).reshape(1, 16).requires_grad_()
    observed = module(value)
    expected = torch.tensor(
        [
            [
                [0 + 1j, 8 + 9j],
                [4 + 5j, 12 + 13j],
                [6 + 7j, 14 + 15j],
                [2 + 3j, 10 + 11j],
            ]
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(observed, expected)
    observed.real.sum().backward()
    torch.testing.assert_close(
        value.grad, torch.tensor([[1.0, 0.0] * 8], dtype=torch.float64)
    )
    assert output[0].representation == "matrix"


def test_shell_decoder_uses_current_default_precision_and_preserves_support():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected = ShellTensorCodec([(1, 1)], dtype=torch.float64)
        spec = FeatureSpec(
            "edge",
            "irreps",
            str(expected.irreps_out),
            support="message",
            angular_basis="yzx",
        )
        actual, output = build_transform(
            "shell_tensor_decode", {"shell_pairs": [(1, 1)]}, [spec]
        )
        value = torch.randn(2, expected.irreps_out.dim)
        torch.testing.assert_close(
            actual(value), expected.decode(value), rtol=0, atol=0
        )
        assert output[0].support == "message"
    finally:
        torch.set_default_dtype(old)


def test_explicit_message_view_pipeline_excludes_zero_distance_output_self_edges():
    configuration = {
        "stages": [
            {
                "id": "nodes",
                "transform": "irrep_linear",
                "inputs": ["node_one_hot"],
                "outputs": ["hidden"],
                "options": {"irreps_out": "2x0e"},
            },
            {
                "id": "radial",
                "transform": "cutoff_radial_expansion",
                "inputs": ["message_lengths"],
                "outputs": ["message_radial"],
                "options": {"r_max": 4.0, "num_basis": 3},
            },
            {
                "id": "sh",
                "transform": "spherical_harmonics",
                "inputs": ["message_vectors"],
                "outputs": ["message_sh"],
                "options": {"irreps": "0e+1o"},
            },
            {
                "id": "density",
                "transform": "neighbor_density",
                "inputs": [
                    "node_one_hot",
                    "hidden",
                    "message_sh",
                    "message_radial",
                    "message_edge_index",
                    "node_batch",
                ],
                "outputs": ["rho", "skip"],
                "options": {
                    "target_irreps": "2x0e+2x1o",
                    "hidden_irreps": "2x0e+2x1o",
                    "avg_num_neighbors": 1.0,
                    "radial_layers": [4],
                },
            },
        ],
        "outputs": {"density": "rho"},
    }
    pipeline = FeaturePipeline(configuration, num_species=2)
    dtype = next(pipeline.parameters()).dtype
    inputs = {
        "node_one_hot": torch.eye(2, dtype=dtype),
        "edge_lengths": torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=dtype),
        "message_edge_index": torch.tensor([[0, 1], [1, 0]]),
        "message_lengths": torch.ones(2, dtype=dtype),
        "message_vectors": torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=dtype
        ),
        "node_batch": torch.zeros(2, dtype=torch.long),
    }
    output = pipeline(inputs)["density"]
    assert output.shape == (2, 2, 4)
    assert torch.isfinite(output).all()


def test_stage_unknown_keys_are_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="Unknown stage keys"):
        FeaturePipeline(
            {
                "stages": [
                    {
                        "id": "a",
                        "transform": "irrep_linear",
                        "inputs": ["node_one_hot"],
                        "outputs": ["node"],
                        "options": {"irreps_out": "2x0e"},
                        "typo_option": True,
                    }
                ],
                "outputs": {"node": "node"},
            },
            num_species=2,
        )


def test_shell_readout_converts_canonical_xyz_at_reconstruction_boundary():
    codec = ShellTensorCodec([(0, 1)])
    decoder, _ = build_transform(
        "shell_tensor_decode",
        {"shell_pairs": [(0, 1)]},
        [FeatureSpec("edge", "irreps", str(codec.irreps_out), angular_basis="xyz")],
    )
    from net.transforms.orbital import IrrepCartesianBasis

    values = torch.randn(3, codec.irreps_out.dim)
    conversion = IrrepCartesianBasis(codec.irreps_out, "xyz", "yzx")
    torch.testing.assert_close(decoder(values), codec.decode(conversion(values)))


def test_bond_linear_refuses_global_irreps_without_frame_rotation():
    with pytest.raises(ValueError, match="explicitly rotated"):
        build_transform(
            "bond_local_linear",
            {"out_features": 4},
            [FeatureSpec("edge", "irreps", "0e+1o")],
        )


def test_scalar_local_readout_requires_global_to_bond_adapter():
    with pytest.raises(ValueError, match="bond-frame"):
        build_transform(
            "local_orbital_output",
            {"packed_dim": 4},
            [FeatureSpec("edge", "irreps", "4x0e")],
        )


@pytest.mark.parametrize("irreps", ["1x1o", "1x0o"])
def test_scalar_endpoint_edge_rejects_global_tensor_and_pseudoscalar_channels(irreps):
    specs = default_input_specs(1)
    with pytest.raises(ValueError, match="invariant edge"):
        build_transform(
            "endpoint_scalar_edge",
            {"out_dim": 3},
            [
                specs["node_one_hot"],
                specs["edge_index"],
                FeatureSpec("edge", "irreps", irreps),
            ],
        )


@pytest.mark.parametrize("irreps", ["1x1o", "1x0o"])
def test_local_messages_reject_unrotated_tensor_and_pseudoscalar_edges(irreps):
    specs = default_input_specs(1)
    ports = [
        specs["node_one_hot"],
        FeatureSpec("edge", "irreps", irreps),
        specs["sub_atom_idx"],
        specs["sub_edge_idx"],
        specs["sub_edge_ang"],
        specs["sub_index"],
        specs["edge_lengths"],
    ]
    with pytest.raises(ValueError, match="invariant edge"):
        build_transform(
            "local_coordinate_message", {"angular_dim": 4, "out_dim": 3}, ports
        )


@pytest.mark.parametrize("irreps", ["1x1o", "1x0o"])
def test_channel_affine_rejects_global_tensor_and_pseudoscalar_channels(irreps):
    source = FeatureSpec("edge", "irreps", irreps)
    with pytest.raises(ValueError, match="invariant scalars"):
        build_transform(
            "channel_linear",
            {"num_channels": 2, "in_features": source.dim, "out_features": 3},
            [source],
        )


@pytest.mark.parametrize("local", [False, True])
def test_channel_affine_preserves_entity_axis_and_frame_contract(local):
    source = (
        FeatureSpec("edge", "local_components", frame="bond", width=3)
        if local
        else FeatureSpec("edge", irreps="3x0e", width=3)
    )
    module, outputs = build_transform(
        "channel_linear",
        {"num_channels": 2, "in_features": 3, "out_features": 4},
        [source],
    )
    value = torch.randn(5, 3, requires_grad=True)
    expected = (
        (torch.matmul(value, module.weight) + module.bias[:, None, :])
        .transpose(0, 1)
        .reshape(5, 8)
    )
    actual = module(value)
    assert actual.shape == (5, 8) and outputs[0].dim == 8
    assert outputs[0].frame == ("bond" if local else "global")
    assert outputs[0].representation == ("local_components" if local else "scalar")
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize(
    "feature",
    [
        {
            "domain": "edge",
            "representation": "matrix",
            "width": 1,
            "support": "message",
        },
        {"domain": "node", "representation": "matrix", "width": 1},
        {"domain": "edge", "representation": "scalar", "width": 1, "support": "output"},
    ],
)
def test_e3gnn_rejects_nonoperator_and_wrong_support_output_slots(feature):
    from core.block_irrep_mapper import BlockIrrepMapper
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from net.common import Config
    from net.e3gnn import E3GNN

    config = {
        "inputs": {"custom": feature},
        "stages": [],
        "outputs": {"hamiltonian": "custom"},
    }
    # Generic pipelines may expose intermediate tensors; operator models cannot
    # silently train these against a different edge index set or representation.
    FeaturePipeline(config, num_species=1)
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": "1s"}))
    with pytest.raises(ValueError, match="matrix features on the output-edge support"):
        E3GNN(mapper, Config(pipeline=config))
