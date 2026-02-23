"""
Test suite for DeepH-E3 inspired architecture changes.

Tests:
- SeparateWeightTensorProduct
- EquiConv module
- Updated EdgeUpdateBlock and NodeUpdateBlock
- MessageBlock with new signature
- E3GNN with unified message-passing (no small/large split)
"""

import pytest
import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps, rand_matrix

from net.common import Config, build_hidden_irreps, SeparateWeightTensorProduct
from net.layers import EquiConv, EdgeUpdateBlock, NodeUpdateBlock, MessageBlock


# ════════════════════════════════════════════════════════════════════════
# Test Helpers
# ════════════════════════════════════════════════════════════════════════
def make_dummy_graph(E=10, N=5, node_dim=32, edge_dim=32, n_radial=64, sh_dim=9):
    """Create dummy graph data for testing."""
    ei = torch.randint(0, N, (2, E))
    node = torch.randn(N, node_dim)
    edge = torch.randn(E, edge_dim)
    edge_sh = torch.randn(E, sh_dim)
    edge_length_emb = torch.randn(E, n_radial)
    return node, edge, ei, edge_sh, edge_length_emb


# ════════════════════════════════════════════════════════════════════════
# SeparateWeightTensorProduct Tests
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_separate_weight_tp_initialization():
    """Test that SeparateWeightTensorProduct initializes correctly."""
    irreps_in1 = Irreps("8x0e + 4x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("8x0e + 4x1o")

    tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_out)

    assert tp is not None
    assert len(tp.weights1) > 0
    assert len(tp.weights2) > 0
    assert len(tp.weights1) == len(tp.weights2)


@pytest.mark.unit
def test_separate_weight_tp_forward():
    """Test forward pass of SeparateWeightTensorProduct."""
    irreps_in1 = Irreps("8x0e + 4x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("8x0e + 4x1o")

    tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_out)

    batch_size = 10
    x1 = irreps_in1.randn(batch_size, -1)
    x2 = irreps_in2.randn(batch_size, -1)

    out = tp(x1, x2)

    assert out.shape == (batch_size, irreps_out.dim)


@pytest.mark.unit
def test_separate_weight_tp_equivariance():
    """Test that SeparateWeightTensorProduct is equivariant."""
    irreps_in1 = Irreps("8x0e + 4x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("8x0e + 4x1o")

    tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_out)

    batch_size = 5
    x1 = irreps_in1.randn(batch_size, -1)
    x2 = irreps_in2.randn(batch_size, -1)

    # Random rotation
    R = rand_matrix()
    D_in1 = irreps_in1.D_from_matrix(R)
    D_in2 = irreps_in2.D_from_matrix(R)
    D_out = irreps_out.D_from_matrix(R)

    # Forward on original inputs
    out = tp(x1, x2)

    # Rotate inputs, then forward
    x1_rot = x1 @ D_in1.T
    x2_rot = x2 @ D_in2.T
    out_rot = tp(x1_rot, x2_rot)

    # Forward, then rotate output
    out_expected = out @ D_out.T

    assert torch.allclose(out_rot, out_expected, atol=1e-4)


# ════════════════════════════════════════════════════════════════════════
# EquiConv Tests
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
@pytest.mark.parametrize("tp_type", ["separate_weight", "fully_connected"])
def test_equiconv_initialization(tp_type):
    """Test EquiConv initialization with different TP types."""
    cfg = Config(tp_type=tp_type, n_radial=64)

    # Use irreps that are compatible with the TP
    # in1 (x) in2 should produce paths to out
    irreps_in1 = Irreps("16x0e + 8x1e")  # Changed to 1e (even parity)
    irreps_in2 = Irreps("1x0e + 1x1o")  # sh irreps
    irreps_out = Irreps(
        "16x0e + 8x1o"
    )  # 0e (x) 1o = 1o, 1e (x) 1o = 0e+1e+2e (includes 0e and 1o via gates)

    conv = EquiConv(
        n_radial=cfg.n_radial,
        irreps_in1=irreps_in1,
        irreps_in2=irreps_in2,
        irreps_out=irreps_out,
        cfg=cfg,
        nonlin=True,
    )

    assert conv is not None
    assert conv.irreps_out.dim > 0


@pytest.mark.unit
def test_equiconv_forward():
    """Test EquiConv forward pass."""
    cfg = Config(tp_type="separate_weight", n_radial=64)

    irreps_in1 = Irreps("16x0e + 8x1e")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("16x0e + 8x1o")

    conv = EquiConv(
        n_radial=cfg.n_radial,
        irreps_in1=irreps_in1,
        irreps_in2=irreps_in2,
        irreps_out=irreps_out,
        cfg=cfg,
        nonlin=True,
    )

    batch_size = 10
    fea_in1 = irreps_in1.randn(batch_size, -1)
    fea_in2 = irreps_in2.randn(batch_size, -1)
    edge_length_emb = torch.randn(batch_size, cfg.n_radial)

    out = conv(fea_in1, fea_in2, edge_length_emb)

    assert out.shape == (batch_size, conv.irreps_out.dim)


@pytest.mark.unit
def test_equiconv_equivariance():
    """Test that EquiConv maintains equivariance."""
    cfg = Config(tp_type="separate_weight", n_radial=64)

    irreps_in1 = Irreps("16x0e + 8x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("16x0e + 8x1o")

    conv = EquiConv(
        n_radial=cfg.n_radial,
        irreps_in1=irreps_in1,
        irreps_in2=irreps_in2,
        irreps_out=irreps_out,
        cfg=cfg,
        nonlin=False,  # No nonlinearity for strict equivariance test
    )

    batch_size = 5
    fea_in1 = irreps_in1.randn(batch_size, -1)
    fea_in2 = irreps_in2.randn(batch_size, -1)
    edge_length_emb = torch.randn(batch_size, cfg.n_radial)

    # Random rotation
    R = rand_matrix()
    D_in1 = irreps_in1.D_from_matrix(R)
    D_in2 = irreps_in2.D_from_matrix(R)
    D_out = conv.irreps_out.D_from_matrix(R)

    # Forward on original
    out = conv(fea_in1, fea_in2, edge_length_emb)

    # Rotate inputs, then forward
    fea_in1_rot = fea_in1 @ D_in1.T
    fea_in2_rot = fea_in2 @ D_in2.T
    out_rot = conv(fea_in1_rot, fea_in2_rot, edge_length_emb)

    # Forward, then rotate
    out_expected = out @ D_out.T

    assert torch.allclose(out_rot, out_expected, atol=1e-4)


# ════════════════════════════════════════════════════════════════════════
# EdgeUpdateBlock Tests
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
@pytest.mark.parametrize("use_self_connection", [True, False])
def test_edge_update_block_new(use_self_connection):
    """Test new EdgeUpdateBlock architecture."""
    cfg = Config(
        use_self_connection=use_self_connection,
        tp_type="separate_weight",
        n_radial=64,
        edge_update_residual=True,
        l_max=2,
    )

    node_irreps = build_hidden_irreps(2, 16)
    edge_irreps = build_hidden_irreps(2, 16)
    sh_irreps = Irreps.spherical_harmonics(2)
    num_species = 2

    edge_blk = EdgeUpdateBlock(
        node_irreps=node_irreps,
        edge_irreps=edge_irreps,
        num_species=num_species,
        cfg=cfg,
    )

    node, edge, ei, edge_sh, edge_length_emb = make_dummy_graph(
        N=5,
        E=10,
        node_dim=node_irreps.dim,
        edge_dim=edge_irreps.dim,
        n_radial=cfg.n_radial,
        sh_dim=sh_irreps.dim,
    )

    # Create edge one-hot if using self-connection
    edge_one_hot = None
    if use_self_connection:
        src_type = torch.randint(0, num_species, (ei.shape[1],))
        dst_type = torch.randint(0, num_species, (ei.shape[1],))
        edge_one_hot = F.one_hot(
            src_type * num_species + dst_type, num_classes=num_species * num_species
        ).float()

    edge_out = edge_blk(node, edge, ei, edge_sh, edge_length_emb, edge_one_hot)

    assert edge_out.shape[0] == edge.shape[0]
    assert edge_out.shape[1] == edge_blk.irreps_out.dim


# ════════════════════════════════════════════════════════════════════════
# NodeUpdateBlock Tests
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
@pytest.mark.parametrize("use_self_connection", [True, False])
def test_node_update_block_new(use_self_connection):
    """Test new NodeUpdateBlock architecture."""
    cfg = Config(
        use_self_connection=use_self_connection,
        tp_type="separate_weight",
        n_radial=64,
        node_update_residual=True,
        l_max=2,
    )

    node_irreps = build_hidden_irreps(2, 16)
    edge_irreps = build_hidden_irreps(2, 16)
    sh_irreps = Irreps.spherical_harmonics(2)
    num_species = 2

    node_blk = NodeUpdateBlock(
        node_irreps=node_irreps,
        edge_irreps=edge_irreps,
        num_species=num_species,
        cfg=cfg,
    )

    node, edge, ei, edge_sh, edge_length_emb = make_dummy_graph(
        N=5,
        E=10,
        node_dim=node_irreps.dim,
        edge_dim=edge_irreps.dim,
        n_radial=cfg.n_radial,
        sh_dim=sh_irreps.dim,
    )

    # Create node one-hot if using self-connection
    node_one_hot = None
    if use_self_connection:
        node_type = torch.randint(0, num_species, (node.shape[0],))
        node_one_hot = F.one_hot(node_type, num_classes=num_species).float()

    node_out = node_blk(node, edge, ei, edge_sh, edge_length_emb, node_one_hot)

    assert node_out.shape[0] == node.shape[0]
    assert node_out.shape[1] == node_blk.irreps_out.dim


# ════════════════════════════════════════════════════════════════════════
# MessageBlock Tests
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_message_block_new_signature():
    """Test MessageBlock with new signature and unified architecture."""
    cfg = Config(
        use_self_connection=True,
        tp_type="separate_weight",
        n_radial=64,
        edge_update_residual=True,
        node_update_residual=True,
        l_max=2,
    )

    node_irreps = build_hidden_irreps(2, 16)
    edge_irreps = build_hidden_irreps(2, 16)
    sh_irreps = Irreps.spherical_harmonics(2)
    num_species = 2

    msg_blk = MessageBlock(
        node_irreps=node_irreps,
        edge_irreps=edge_irreps,
        num_species=num_species,
        cfg=cfg,
        info={"layer": 0},
    )

    node, edge, ei, edge_sh, edge_length_emb = make_dummy_graph(
        N=5,
        E=10,
        node_dim=node_irreps.dim,
        edge_dim=edge_irreps.dim,
        n_radial=cfg.n_radial,
        sh_dim=sh_irreps.dim,
    )

    # Create one-hot encodings
    node_type = torch.randint(0, num_species, (node.shape[0],))
    node_one_hot = F.one_hot(node_type, num_classes=num_species).float()

    src_type = node_type[ei[0]]
    dst_type = node_type[ei[1]]
    edge_one_hot = F.one_hot(
        src_type * num_species + dst_type, num_classes=num_species * num_species
    ).float()

    node_out, edge_out = msg_blk(
        node, edge, ei, edge_sh, edge_length_emb, node_one_hot, edge_one_hot
    )

    assert node_out.shape[0] == node.shape[0]
    assert edge_out.shape[0] == edge.shape[0]


@pytest.mark.unit
def test_message_block_activation_magnitudes():
    """Test that MessageBlock tracks activation magnitudes correctly."""
    cfg = Config(
        use_self_connection=True,
        tp_type="separate_weight",
        n_radial=64,
        log_activation_mag=True,
        l_max=2,
    )

    node_irreps = build_hidden_irreps(2, 16)
    edge_irreps = build_hidden_irreps(2, 16)
    sh_irreps = Irreps.spherical_harmonics(2)
    num_species = 2

    msg_blk = MessageBlock(
        node_irreps=node_irreps,
        edge_irreps=edge_irreps,
        num_species=num_species,
        cfg=cfg,
        info={"layer": 0, "graph": "test"},
    )

    node, edge, ei, edge_sh, edge_length_emb = make_dummy_graph(
        N=5,
        E=10,
        node_dim=node_irreps.dim,
        edge_dim=edge_irreps.dim,
        n_radial=cfg.n_radial,
        sh_dim=sh_irreps.dim,
    )

    node_type = torch.randint(0, num_species, (node.shape[0],))
    node_one_hot = F.one_hot(node_type, num_classes=num_species).float()

    src_type = node_type[ei[0]]
    dst_type = node_type[ei[1]]
    edge_one_hot = F.one_hot(
        src_type * num_species + dst_type, num_classes=num_species * num_species
    ).float()

    activation_mags = {}
    node_out, edge_out = msg_blk(
        node,
        edge,
        ei,
        edge_sh,
        edge_length_emb,
        node_one_hot,
        edge_one_hot,
        activation_mags=activation_mags,
    )

    # Check that activation magnitudes were recorded
    assert len(activation_mags) > 0
    assert any("mag_edge_test_layer_0" in key for key in activation_mags.keys())
    assert any("mag_node_test_layer_0" in key for key in activation_mags.keys())


# ════════════════════════════════════════════════════════════════════════
# Config Tests
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_config_new_parameters():
    """Test that Config has new parameters."""
    cfg = Config()

    # Check new parameters exist
    assert hasattr(cfg, "tp_type")
    assert hasattr(cfg, "use_self_connection")

    # Check defaults
    assert cfg.tp_type in ["separate_weight", "fully_connected"]
    assert isinstance(cfg.use_self_connection, bool)


@pytest.mark.unit
def test_config_deephe3_mode():
    """Test creating a separate configuration."""
    cfg = Config(
        tp_type="separate_weight",
        use_self_connection=True,
        node_update_message_agg="sum",
        num_layers_gnn=3,
    )

    assert cfg.tp_type == "separate_weight"
    assert cfg.use_self_connection is True
    assert cfg.node_update_message_agg == "sum"
    assert cfg.num_layers_gnn == 3
