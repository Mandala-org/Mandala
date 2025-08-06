"""
Tests for the E(3) equivariance of all network components.

This test suite verifies that each layer and sub-layer in the E3GNN framework
correctly adheres to rotational equivariance. The core methodology for each test is:

1.  Initialize the layer, often with various configurations using pytest.mark.parametrize.
2.  Generate a random input tensor `x` that conforms to the layer's input Irreps.
3.  Generate a random SO(3) rotation matrix `R`.
4.  Calculate the Wigner-D matrices `D_in` and `D_out` for the input and output Irreps.
5.  Transform the input tensor with the rotation: `x_rotated = x @ D_in.T`.
6.  Pass both the original and the transformed inputs through the layer to get `y` and `y_transformed_input`.
7.  The core assertion checks if transforming the original output `y` is equivalent to the output from the transformed input.
    Mathematically: `y_transformed_input` should be approximately equal to `y @ D_out.T`.

This ensures that for a rotation in the input space, the output transforms predictably
according to its own representation, which is the definition of equivariance.
"""

import sys
from pathlib import Path
import pytest
import torch
from scipy.spatial.transform import Rotation as R
from e3nn.o3 import Irreps

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from net.common import Config, build_hidden_irreps  # noqa: E402
from net.encoders import NodeEncoder, EdgeEncoder  # noqa: E402
from net.layers import EdgeUpdateBlock, NodeUpdateBlock, MessageBlock  # noqa: E402
from net.heads import DeepHead  # noqa: E402
from net.activations import make_nonlinearity  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402

# --- Test Helpers ---


def random_rotation_matrix() -> torch.Tensor:
    """Generates a random 3x3 SO(3) rotation matrix."""
    return torch.tensor(R.random().as_matrix(), dtype=torch.float32)


def generate_equivariant_input(irreps: Irreps, batch_size: int = 1) -> torch.Tensor:
    """Creates a random tensor conforming to the given Irreps."""
    return irreps.randn(batch_size, -1)


# --- Equivariance Tests ---


def test_node_encoder_equivariance():
    """
    Tests the NodeEncoder. Since it only produces scalars, the output
    should be invariant to rotation (a special case of equivariance).
    """
    cfg = Config()
    layer = NodeEncoder(node_one_hot_dim=10, cfg=cfg)
    node_type_idx = torch.randint(0, 10, (5,))

    # The input is not geometric, so there's no rotation to apply to it.
    # We just check that the output is scalar and therefore invariant.
    output = layer(node_type_idx)

    assert layer.irreps_out.lmax == 0, "NodeEncoder should only output scalars"
    # No assertion needed beyond this, as scalar output is inherently invariant.
    # The test serves to document this property.
    assert output.shape == (5, cfg.hidden_base_dim)


@pytest.mark.parametrize("l_max_gnn", [1, 2, 3])
def test_edge_encoder_equivariance(l_max_gnn):
    """Tests the EdgeEncoder for equivariance."""
    E, N_edge_types, n_radial = 5, 7, 16
    cfg = Config(l_max_gnn=l_max_gnn, n_radial=n_radial)
    layer = EdgeEncoder(
        n_edge_types=N_edge_types,
        irreps_out=build_hidden_irreps(l_max_gnn, 32),
        cfg=cfg,
    )

    # Inputs
    edge_type_idx = torch.randint(0, N_edge_types, (E,))
    length_emb = torch.randn(E, n_radial)
    sh = generate_equivariant_input(layer.sh_irreps, batch_size=E)

    # Rotation
    rot = random_rotation_matrix()
    D_in = layer.sh_irreps.D_from_matrix(rot)
    D_out = layer.irreps_out.D_from_matrix(rot)

    # Transform inputs
    sh_rotated = sh @ D_in.T

    # Apply layer
    y = layer(edge_type_idx, length_emb, sh)
    y_rotated_input = layer(edge_type_idx, length_emb, sh_rotated)

    # Check equivariance
    y_rotated_output = y @ D_out.T
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=1e-5)


@pytest.mark.parametrize(
    "nonlin_kind", ["normact", "gate_scalars_mlp", "gate_magnitudes"]
)
@pytest.mark.parametrize("irreps_str", ["32x0e+16x1o+8x2e", "16x0e+16x0o+8x1e+8x1o"])
def test_activations_equivariance(nonlin_kind, irreps_str):
    """Tests all activation layers for equivariance."""
    B = 2
    cfg = Config(nonlin_kind=nonlin_kind)
    irreps = Irreps(irreps_str)
    layer = make_nonlinearity(irreps, cfg)

    # Input and Rotation
    x = generate_equivariant_input(irreps, batch_size=B)
    rot = random_rotation_matrix()
    D_in = irreps.D_from_matrix(rot)
    D_out = irreps.D_from_matrix(rot)  # Activations don't change irreps

    # Transform input
    x_rotated = x @ D_in.T

    # Apply layer
    y = layer(x)
    y_rotated_input = layer(x_rotated)

    # Check equivariance
    y_rotated_output = y @ D_out.T
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=1e-5)


@pytest.mark.parametrize("edge_update_node_combine", ["concat", "sum"])
@pytest.mark.parametrize("edge_update", ["tensor_product", "concat", "replace"])
@pytest.mark.parametrize("edge_update_residual", [True, False])
def test_edge_update_block_equivariance(
    edge_update_node_combine, edge_update, edge_update_residual
):
    """Tests the EdgeUpdateBlock in isolation."""
    N, E = 10, 20
    cfg = Config(
        edge_update_node_combine=edge_update_node_combine,
        edge_update=edge_update,
        edge_update_residual=edge_update_residual,
    )
    hidden_irreps = build_hidden_irreps(cfg.l_max_gnn, 32)
    layer = EdgeUpdateBlock(hidden_irreps, cfg)

    # Inputs and Rotation
    node = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))
    rot = random_rotation_matrix()
    D_hidden = hidden_irreps.D_from_matrix(rot)

    # Transform inputs
    node_rotated = node @ D_hidden.T
    edge_rotated = edge @ D_hidden.T

    # Apply layer
    y = layer(node, edge, edge_index)
    y_rotated_input = layer(node_rotated, edge_rotated, edge_index)

    # Check equivariance
    y_rotated_output = y @ D_hidden.T
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=1e-5)


@pytest.mark.parametrize("node_update_message_agg", ["attention", "sum"])
@pytest.mark.parametrize("node_update", ["tensor_product", "concat", "replace", "sum"])
@pytest.mark.parametrize("node_update_residual", [True, False])
def test_node_update_block_equivariance(
    node_update_message_agg, node_update, node_update_residual
):
    """Tests the NodeUpdateBlock in isolation."""
    N, E = 10, 20
    cfg = Config(
        node_update_message_agg=node_update_message_agg,
        node_update=node_update,
        node_update_residual=node_update_residual,
    )
    hidden_irreps = build_hidden_irreps(cfg.l_max_gnn, 32)
    layer = NodeUpdateBlock(hidden_irreps, cfg)

    # Inputs and Rotation
    node = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))
    rot = random_rotation_matrix()
    D_hidden = hidden_irreps.D_from_matrix(rot)

    # Transform inputs
    node_rotated = node @ D_hidden.T
    edge_rotated = edge @ D_hidden.T

    # Apply layer
    y = layer(node, edge, edge_index)
    y_rotated_input = layer(node_rotated, edge_rotated, edge_index)

    # Check equivariance
    y_rotated_output = y @ D_hidden.T
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=1e-5)


def test_message_block_equivariance():
    """Tests the full MessageBlock as an integration test."""
    N, E = 10, 20
    cfg = Config()
    hidden_irreps = build_hidden_irreps(cfg.l_max_gnn, 32)
    layer = MessageBlock(hidden_irreps, cfg)

    # Inputs and Rotation
    node = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))
    rot = random_rotation_matrix()
    D_hidden = hidden_irreps.D_from_matrix(rot)

    # Transform inputs
    node_rotated = node @ D_hidden.T
    edge_rotated = edge @ D_hidden.T

    # Apply layer
    node_out, edge_out = layer(node, edge, edge_index)
    node_out_rot, edge_out_rot = layer(node_rotated, edge_rotated, edge_index)

    # Check equivariance for both outputs
    assert torch.allclose(node_out_rot, node_out @ D_hidden.T, atol=1e-5)
    assert torch.allclose(edge_out_rot, edge_out @ D_hidden.T, atol=1e-5)


@pytest.mark.parametrize("head_use_mlp_log_scale", [True, False])
@pytest.mark.parametrize("neck_depth", [1, 2])
@pytest.mark.parametrize("head_depth", [1, 2])
def test_deep_head_equivariance(head_use_mlp_log_scale, neck_depth, head_depth):
    """Tests the DeepHead for equivariance."""
    E = 30
    cfg = Config(
        head_use_mlp_log_scale=head_use_mlp_log_scale,
        neck_depth=neck_depth,
        head_depth=head_depth,
        l_max_gnn=2,
        l_max_matrix=3,
        hidden_base_dim=16,
    )

    # Mock mapper
    orbital_cfg = OrbitalIrrepConfig.from_dict(
        {"H": ["1x0e", "1x1o"], "O": ["2x0e", "1x1o", "1x2e"]}
    )
    mapper = BlockIrrepMapper(orbital_cfg)
    pair_keys = list(map(lambda pair: f"{pair[0]}-{pair[1]}", mapper._maps.keys()))

    # Layer
    hidden_irreps = build_hidden_irreps(cfg.l_max_gnn, cfg.hidden_base_dim)
    neck_irreps = build_hidden_irreps(cfg.l_max_matrix, cfg.hidden_base_dim)
    layer = DeepHead(hidden_irreps, neck_irreps, pair_keys, mapper, cfg)

    # Inputs and Rotation
    edge_feat = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_type_idx = torch.randint(0, len(pair_keys), (E,))
    edge_index = torch.randint(0, 10, (2, E))  # Dummy node indices
    rot = random_rotation_matrix()
    D_in = hidden_irreps.D_from_matrix(rot)

    # Transform input
    edge_feat_rotated = edge_feat @ D_in.T

    # Apply layer
    y_dict = layer(edge_feat, edge_type_idx, edge_index)
    y_dict_rotated_input = layer(edge_feat_rotated, edge_type_idx, edge_index)

    # Check equivariance for each output pair
    for key in y_dict:
        y_vec = y_dict[key]["vectors"]
        y_vec_rotated_input = y_dict_rotated_input[key]["vectors"]

        out_irreps = mapper.get_pair_irreps(key)
        D_out = out_irreps.D_from_matrix(rot)

        y_vec_rotated_output = y_vec @ D_out.T

        assert torch.allclose(y_vec_rotated_input, y_vec_rotated_output, atol=1e-4)
