import pytest
import torch
from itertools import product
from e3nn.o3 import Irreps

from net.common import Config, build_hidden_irreps
from net.layers import EdgeUpdateBlock, NodeUpdateBlock, MessageBlock


def make_dummy_graph(E=10, N=5, hid_dim=32):
    ei = torch.randint(0, N, (2, E))
    node = torch.randn(N, hid_dim)
    edge = torch.randn(E, hid_dim)
    return node, edge, ei


EDGE_UPDATE_CONFIGS = list(
    product(
        ["sum", "concat"],  # edge_update_node_combine
        ["tensor_product", "concat", "replace"],  # edge_update
        [True, False],  # edge_update_residual
    )
)


@pytest.mark.parametrize(
    "node_combine, edge_update, residual",
    EDGE_UPDATE_CONFIGS,
)
@pytest.mark.unit
def test_edge_update_block_variants(node_combine, edge_update, residual):
    """Tests the EdgeUpdateBlock with various configurations."""
    cfg = Config(
        edge_update_node_combine=node_combine,
        edge_update=edge_update,
        edge_update_residual=residual,
        safety_checks=True,
        l_max=2,
    )
    hid = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    num_species = 2
    edge_blk = EdgeUpdateBlock(hid, hid, num_species, cfg)

    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(edge.shape[0], sh_irreps.dim)
    edge_length_emb = torch.randn(edge.shape[0], cfg.n_radial)
    edge_out = edge_blk(node, edge, ei, edge_sh, edge_length_emb)

    assert edge_out.shape == edge.shape


NODE_UPDATE_CONFIGS = list(
    product(
        ["attention", "sum"],  # node_update_message_agg
        ["tensor_product", "concat", "replace", "sum"],  # node_update
        [True, False],  # node_update_residual
    )
)


@pytest.mark.parametrize(
    "message_agg, node_update, residual",
    NODE_UPDATE_CONFIGS,
)
@pytest.mark.unit
def test_node_update_block_variants(message_agg, node_update, residual):
    """Tests the NodeUpdateBlock with various configurations."""
    cfg = Config(
        node_update_message_agg=message_agg,
        node_update=node_update,
        node_update_residual=residual,
        safety_checks=True,
        l_max=2,
    )
    hid = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    num_species = 2
    node_blk = NodeUpdateBlock(hid, hid, num_species, cfg)

    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(edge.shape[0], sh_irreps.dim)
    edge_length_emb = torch.randn(edge.shape[0], cfg.n_radial)
    node_out = node_blk(node, edge, ei, edge_sh, edge_length_emb)

    assert node_out.shape == node.shape


@pytest.mark.unit
def test_message_block_roundtrip():
    """Tests the full MessageBlock forward pass."""
    cfg = Config(safety_checks=True, l_max=2)
    hid = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    num_species = 2

    blk = MessageBlock(hid, hid, num_species, cfg)
    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(edge.shape[0], sh_irreps.dim)
    edge_length_emb = torch.randn(edge.shape[0], cfg.n_radial)
    n2, e2 = blk(node, edge, ei, edge_sh, edge_length_emb)

    assert n2.shape == node.shape
    assert e2.shape == edge.shape


@pytest.mark.unit
def test_message_block_tracks_post_refine_irreps():
    """Downstream edge blocks must be built against the post-refine node irreps."""
    cfg = Config(
        safety_checks=True,
        l_max=4,
        hidden_base_dim=32,
        hidden_irreps="32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e",
        internal_e3mlp_layers=1,
        e3layernorm=False,
        edge_encoder_style="distance",
    )
    hid = Irreps(cfg.hidden_irreps)
    num_species = 2

    blk = MessageBlock(
        node_irreps=Irreps("32x0e"),
        edge_irreps=Irreps("32x0e"),
        num_species=num_species,
        cfg=cfg,
        node_irreps_out=hid,
        edge_irreps_out=hid,
    )

    assert blk.node_upd.irreps_out == hid
    assert blk.edge_upd.node_irreps == hid
