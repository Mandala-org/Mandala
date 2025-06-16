import torch

from net.common import HyperParams, build_hidden_irreps
from net.layers import EdgeUpdateBlock, NodeUpdateBlock, MessageBlock


def make_dummy_graph(E=10, N=5, hid_dim=32):
    ei = torch.randint(0, N, (2, E))
    node = torch.randn(N, hid_dim)
    edge = torch.randn(E, hid_dim)
    return node, edge, ei


def test_edge_node_update_shapes():
    hp = HyperParams(dropout=0.1, batch_norm=True)
    hid = build_hidden_irreps(hp.l_max, hp.hidden_base_dim)

    edge_blk = EdgeUpdateBlock(hid, hp)
    node_blk = NodeUpdateBlock(hid, hp)

    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    edge2 = edge_blk(node, edge, ei)
    node2 = node_blk(node, edge2, ei)

    assert edge2.shape == edge.shape
    assert node2.shape == node.shape


def test_message_block_roundtrip():
    hp = HyperParams(use_edge_updates=False, dropout=0.0)
    hid = build_hidden_irreps(hp.l_max, hp.hidden_base_dim)

    blk = MessageBlock(hid, hp)
    node, edge, ei = make_dummy_graph(hid_dim=hid.dim)
    n2, e2 = blk(node, edge, ei)

    assert n2.shape == node.shape
    assert e2.shape == edge.shape
