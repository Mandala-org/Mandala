import pytest
import torch
from itertools import product

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
    )
    hid = build_hidden_irreps(cfg.l_max_gnn, cfg.hidden_base_dim)
    edge_blk = EdgeUpdateBlock(hid, cfg)

    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    edge_out = edge_blk(node, edge, ei)

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
    )
    hid = build_hidden_irreps(cfg.l_max_gnn, cfg.hidden_base_dim)
    node_blk = NodeUpdateBlock(hid, cfg)

    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    node_out = node_blk(node, edge, ei)

    assert node_out.shape == node.shape


@pytest.mark.unit
def test_message_block_roundtrip():
    """Tests the full MessageBlock forward pass."""
    cfg = Config()
    hid = build_hidden_irreps(cfg.l_max_gnn, cfg.hidden_base_dim)

    blk = MessageBlock(hid, cfg)
    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    n2, e2 = blk(node, edge, ei)

    assert n2.shape == node.shape
    assert e2.shape == edge.shape
