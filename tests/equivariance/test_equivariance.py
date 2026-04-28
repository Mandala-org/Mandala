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
from data.edge_alignment import build_prediction_edge_metadata  # noqa: E402

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
    cfg = Config(
        safety_checks=True,
    )
    layer = NodeEncoder(node_one_hot_dim=10, cfg=cfg)
    node_type_idx = torch.randint(0, 10, (5,))

    # The input is not geometric, so there's no rotation to apply to it.
    # We just check that the output is scalar and therefore invariant.
    output = layer(node_type_idx)

    assert layer.irreps_out.lmax == 0, "NodeEncoder should only output scalars"
    # No assertion needed beyond this, as scalar output is inherently invariant.
    # The test serves to document this property.
    assert output.shape == (5, cfg.hidden_base_dim)


@pytest.mark.parametrize("l_max", [1, 2, 3])
def test_edge_encoder_equivariance(l_max):
    """Tests the EdgeEncoder for equivariance."""
    E, N_edge_types, n_radial = 5, 7, 16
    cfg = Config(l_max=l_max, n_radial=n_radial, safety_checks=True)
    layer = EdgeEncoder(
        n_edge_types=N_edge_types,
        irreps_out=build_hidden_irreps(l_max, 32),
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
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=2e-4)


@pytest.mark.parametrize(
    "nonlin_kind", ["normact", "gate_scalars_mlp", "gate_magnitudes"]
)
@pytest.mark.parametrize("irreps_str", ["32x0e+16x1o+8x2e", "16x0e+16x0o+8x1e+8x1o"])
def test_activations_equivariance(nonlin_kind, irreps_str):
    """Tests all activation layers for equivariance."""
    B = 2
    cfg = Config(nonlin_kind=nonlin_kind, safety_checks=True)
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
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=2e-4)


@pytest.mark.parametrize("edge_update_node_combine", ["concat", "sum"])
@pytest.mark.parametrize("edge_update", ["tensor_product", "concat", "replace"])
@pytest.mark.parametrize("edge_update_residual", [True, False])
@pytest.mark.parametrize("mlp_layers", [1, 2])
def test_edge_update_block_equivariance(
    edge_update_node_combine, edge_update, edge_update_residual, mlp_layers
):
    """Tests the EdgeUpdateBlock in isolation."""
    N, E = 10, 20
    cfg = Config(
        edge_update_node_combine=edge_update_node_combine,
        edge_update=edge_update,
        edge_update_residual=edge_update_residual,
        edge_update_pre_lin_mlp_n_layers=mlp_layers,
        edge_update_post_lin_mlp_n_layers=mlp_layers,
        safety_checks=True,
        l_max=2,
    )
    hidden_irreps = build_hidden_irreps(cfg.l_max, 32)
    num_species = 2
    layer = EdgeUpdateBlock(hidden_irreps, hidden_irreps, num_species, cfg)

    # Inputs and Rotation
    node = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = generate_equivariant_input(sh_irreps, batch_size=E)
    edge_length_emb = torch.randn(E, cfg.n_radial)
    rot = random_rotation_matrix()
    D_hidden = hidden_irreps.D_from_matrix(rot)
    D_sh = sh_irreps.D_from_matrix(rot)

    # Transform inputs
    node_rotated = node @ D_hidden.T
    edge_rotated = edge @ D_hidden.T
    edge_sh_rotated = edge_sh @ D_sh.T

    # Apply layer
    y = layer(node, edge, edge_index, edge_sh, edge_length_emb)
    y_rotated_input = layer(
        node_rotated, edge_rotated, edge_index, edge_sh_rotated, edge_length_emb
    )

    # Check equivariance
    y_rotated_output = y @ D_hidden.T
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=2e-4)


@pytest.mark.parametrize("node_update_message_agg", ["attention", "sum"])
@pytest.mark.parametrize("node_update", ["tensor_product", "concat", "replace", "sum"])
@pytest.mark.parametrize("node_update_residual", [True, False])
@pytest.mark.parametrize("mlp_layers", [1, 2])
def test_node_update_block_equivariance(
    node_update_message_agg, node_update, node_update_residual, mlp_layers
):
    """Tests the NodeUpdateBlock in isolation."""
    N, E = 10, 20
    cfg = Config(
        node_update_message_agg=node_update_message_agg,
        node_update=node_update,
        node_update_residual=node_update_residual,
        node_update_pre_lin_mlp_n_layers=mlp_layers,
        node_update_attention_mlp_n_layers=mlp_layers,
        node_update_post_lin_mlp_n_layers=mlp_layers,
        safety_checks=True,
        l_max=2,
    )
    hidden_irreps = build_hidden_irreps(cfg.l_max, 32)
    num_species = 2
    layer = NodeUpdateBlock(hidden_irreps, hidden_irreps, num_species, cfg)

    # Inputs and Rotation
    node = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = generate_equivariant_input(sh_irreps, batch_size=E)
    edge_length_emb = torch.randn(E, cfg.n_radial)
    rot = random_rotation_matrix()
    D_hidden = hidden_irreps.D_from_matrix(rot)
    D_sh = sh_irreps.D_from_matrix(rot)

    # Transform inputs
    node_rotated = node @ D_hidden.T
    edge_rotated = edge @ D_hidden.T
    edge_sh_rotated = edge_sh @ D_sh.T

    # Apply layer
    y = layer(node, edge, edge_index, edge_sh, edge_length_emb)
    y_rotated_input = layer(
        node_rotated, edge_rotated, edge_index, edge_sh_rotated, edge_length_emb
    )

    # Check equivariance
    y_rotated_output = y @ D_hidden.T
    assert torch.allclose(y_rotated_input, y_rotated_output, atol=2e-4)


def test_message_block_equivariance():
    """Tests the full MessageBlock as an integration test."""
    N, E = 10, 20
    cfg = Config(safety_checks=True, l_max=2)
    hidden_irreps = build_hidden_irreps(cfg.l_max, 32)
    num_species = 2
    layer = MessageBlock(hidden_irreps, hidden_irreps, num_species, cfg)

    # Inputs and Rotation
    node = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = generate_equivariant_input(sh_irreps, batch_size=E)
    edge_length_emb = torch.randn(E, cfg.n_radial)
    rot = random_rotation_matrix()
    D_hidden = hidden_irreps.D_from_matrix(rot)
    D_sh = sh_irreps.D_from_matrix(rot)

    # Transform inputs
    node_rotated = node @ D_hidden.T
    edge_rotated = edge @ D_hidden.T
    edge_sh_rotated = edge_sh @ D_sh.T

    # Apply layer
    node_out, edge_out = layer(node, edge, edge_index, edge_sh, edge_length_emb)
    node_out_rot, edge_out_rot = layer(
        node_rotated, edge_rotated, edge_index, edge_sh_rotated, edge_length_emb
    )

    # Check equivariance for both outputs
    assert torch.allclose(node_out_rot, node_out @ D_hidden.T, atol=2e-4)
    assert torch.allclose(edge_out_rot, edge_out @ D_hidden.T, atol=2e-4)


@pytest.mark.parametrize("head_use_mlp_log_scale", [True, False])
@pytest.mark.parametrize("mlp_layers", [1, 2])
def test_deep_head_equivariance(head_use_mlp_log_scale, mlp_layers):
    """Tests the DeepHead for equivariance."""
    E = 30
    cfg = Config(
        head_use_mlp_log_scale=head_use_mlp_log_scale,
        neck_depth=mlp_layers,
        head_e3mlp_layers=mlp_layers,
        head_log_scale_mlp_n_layers=mlp_layers,
        l_max=3,
        hidden_base_dim=16,
        safety_checks=True,
    )

    # Mock mapper
    orbital_cfg = OrbitalIrrepConfig.from_dict(
        {"H": ["1x0e", "1x1o"], "O": ["2x0e", "1x1o", "1x2e"]}
    )
    mapper = BlockIrrepMapper(orbital_cfg)
    pair_keys = list(map(lambda pair: f"{pair[0]}-{pair[1]}", mapper._maps.keys()))

    # Layer
    hidden_irreps = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    neck_irreps = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    layer = DeepHead(
        irreps_diag_in=hidden_irreps,
        irreps_edge_in=hidden_irreps,
        irreps_neck=neck_irreps,
        pair_keys=pair_keys,
        mapper=mapper,
        cfg=cfg,
    )

    # Inputs and Rotation
    N = 10
    node_feat = generate_equivariant_input(hidden_irreps, batch_size=N)
    edge_feat = generate_equivariant_input(hidden_irreps, batch_size=E)
    edge_index = torch.randint(0, N, (2, E))  # Dummy node indices
    edge_shift = torch.zeros(3, E, dtype=torch.long)
    atoms = tuple("H" if i % 2 == 0 else "O" for i in range(N))
    edge_type_idx = torch.tensor(
        [
            mapper.edge_type2idx[f"{atoms[i]}-{atoms[j]}"]
            for i, j in edge_index.t().tolist()
        ],
        dtype=torch.long,
    )
    metadata = build_prediction_edge_metadata(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms=atoms,
        edge_types=mapper.edge_types,
        edge_type2idx=mapper.edge_type2idx,
        separate_shifted_self=bool(cfg.separate_shifted_self),
    )
    rot = random_rotation_matrix()
    D_in = hidden_irreps.D_from_matrix(rot)

    # Transform input
    node_feat_rotated = node_feat @ D_in.T
    edge_feat_rotated = edge_feat @ D_in.T

    # Apply layer
    y_dict = layer(
        node_feat,
        edge_feat,
        metadata["pred_pair_edges_static"],
        metadata["edge_partitions"],
    )
    y_dict_rotated_input = layer(
        node_feat_rotated,
        edge_feat_rotated,
        metadata["pred_pair_edges_static"],
        metadata["edge_partitions"],
    )

    # Check equivariance for each output pair
    for key in y_dict:
        y_vec = y_dict[key]
        y_vec_rotated_input = y_dict_rotated_input[key]

        out_irreps = mapper.get_pair_irreps(key)
        D_out = out_irreps.D_from_matrix(rot)

        y_vec_rotated_output = y_vec @ D_out.T

        assert torch.allclose(y_vec_rotated_input, y_vec_rotated_output, atol=2e-3)
