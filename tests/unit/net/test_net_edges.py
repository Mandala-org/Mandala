import torch
from net.e3gnn import E3GNN
from net.layers import EdgeUpdateBlock, NodeUpdateBlock
from net.common import Config
from e3nn.o3 import Irreps
from data.block_matrix import IrrepsBlockData
from core.orbital_irrep_config import OrbitalIrrepConfig


def test_wrap_head_output():
    # Test _wrap_head_output logic (rebuilding lookup)
    # Mock E3GNN
    class MockE3GNN:
        def __init__(self):
            self.mapper = type(
                "MockMapper",
                (),
                {"orbital_cfg": OrbitalIrrepConfig.from_dict({"H": "1s"})},
            )()

        _wrap_head_output = E3GNN._wrap_head_output

    model = MockE3GNN()

    # Raw output from head
    # Key "H-H"
    # Edges: (0,0,0,0,1)
    edges = torch.tensor([[0, 0, 0, 0, 1]]).t()
    vectors = torch.randn(1, 1)

    raw = {"H-H": {"vectors": vectors, "edges": edges}}
    atoms = ("H", "H")

    ibd = model._wrap_head_output(raw, atoms)

    assert isinstance(ibd, IrrepsBlockData)
    assert (0, 0, 0, 0, 1) in ibd.lookup
    assert ibd.lookup[(0, 0, 0, 0, 1)] == ("H-H", 0)


def test_edge_update_block():
    # Test EdgeUpdateBlock edge unpacking
    cfg = Config(
        edge_update_node_combine="concat",
        edge_update="concat",
        edge_update_pre_lin_mlp_n_layers=1,
        edge_update_post_lin_mlp_n_layers=1,
        dropout=0.0,
        edge_update_residual=False,
    )
    hidden_irreps = Irreps("1x0e")

    block = EdgeUpdateBlock(hidden_irreps, cfg)

    # Mock input
    node = torch.randn(2, 1)
    edge = torch.randn(1, 1)
    edge_index = torch.tensor([[0], [1]])  # src=0, dst=1

    # Forward
    out = block(node, edge, edge_index)

    assert out.shape == (1, 1)


def test_node_update_block():
    # Test NodeUpdateBlock edge unpacking
    cfg = Config(
        node_update_message_agg="sum",
        node_update="concat",
        node_update_pre_lin_mlp_n_layers=1,
        node_update_post_lin_mlp_n_layers=1,
        dropout=0.0,
        node_update_residual=False,
    )
    hidden_irreps = Irreps("1x0e")

    block = NodeUpdateBlock(hidden_irreps, cfg)

    # Mock input
    node = torch.randn(2, 1)
    edge = torch.randn(1, 1)
    edge_index = torch.tensor([[0], [1]])  # src=0, dst=1

    # Forward
    out = block(node, edge, edge_index)

    assert out.shape == (2, 1)
