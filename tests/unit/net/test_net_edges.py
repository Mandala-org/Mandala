import torch
from torch import nn
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

    raw = {"H-H": vectors}
    atoms = ("H", "H")
    x = {
        "atoms_tuple": atoms,
        "atom_counts": {"H": 2},
        "pred_pair_edges_static": {"H-H": edges},
        "pred_lookup_static": {(0, 0, 0, 0, 1): ("H-H", 0)},
    }

    ibd = model._wrap_head_output(raw, x)

    assert isinstance(ibd, IrrepsBlockData)
    assert (0, 0, 0, 0, 1) in ibd.lookup
    assert ibd.lookup[(0, 0, 0, 0, 1)] == ("H-H", 0)


def test_edge_update_block():
    # Replace only the equivariant convolution with a fixed linear probe. This
    # isolates the block's endpoint ordering and gradient flow from random TP
    # weights while exercising the real forward method.
    cfg = Config(
        edge_update_node_combine="concat",
        dropout=0.0,
        edge_update_residual=False,
        l_max=2,
    )
    hidden_irreps = Irreps("1x0e")
    num_species = 2

    block = EdgeUpdateBlock(hidden_irreps, hidden_irreps, num_species, cfg)

    # Mock input
    node = torch.tensor([[1.0], [2.0], [-1.0]], requires_grad=True)
    edge = torch.tensor([[0.5], [-0.5]], requires_grad=True)
    edge_index = torch.tensor([[0, 2], [1, 0]])  # src, dst
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.zeros(2, sh_irreps.dim)
    edge_length_emb = torch.zeros(2, cfg.n_radial)

    class EndpointProbe(nn.Module):
        def forward(self, features, spherical_harmonics, radial, **kwargs):
            # concat mode is [source, destination, edge].
            return features[:, :1] + 2 * features[:, 1:2] + 3 * features[:, 2:3]

    block.lin_pre = nn.Identity()
    block.conv = EndpointProbe()
    block.lin_post = nn.Identity()
    block.norm_act = nn.Identity()
    block.refine = nn.Identity()

    # Forward
    out = block(node, edge, edge_index, edge_sh, edge_length_emb)

    expected = node[edge_index[0]] + 2 * node[edge_index[1]] + 3 * edge
    torch.testing.assert_close(out, expected)
    node_grad, edge_grad = torch.autograd.grad(out.sum(), (node, edge))
    expected_node_grad = torch.tensor([[3.0], [2.0], [1.0]])
    expected_edge_grad = torch.full_like(edge, 3.0)
    torch.testing.assert_close(node_grad, expected_node_grad)
    torch.testing.assert_close(edge_grad, expected_edge_grad)


def test_node_update_block():
    # Isolate the real receiver scatter from equivariant message construction
    # with a deterministic scalar message probe.
    cfg = Config(
        node_update_message_agg="sum",
        dropout=0.0,
        node_update_residual=False,
        l_max=2,
    )
    hidden_irreps = Irreps("1x0e")
    num_species = 2

    block = NodeUpdateBlock(hidden_irreps, hidden_irreps, num_species, cfg)

    # Mock input
    node = torch.tensor([[1.0], [2.0], [-1.0]], requires_grad=True)
    edge = torch.tensor([[0.5], [-0.5], [1.5]], requires_grad=True)
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 0]])  # src, receiver
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.zeros(3, sh_irreps.dim)
    edge_length_emb = torch.zeros(3, cfg.n_radial)

    class MessageProbe(nn.Module):
        def forward(self, features, spherical_harmonics, radial, **kwargs):
            # node-update inputs are [source, receiver, edge].
            return features[:, :1] + 2 * features[:, 1:2] + 3 * features[:, 2:3]

    block.lin_pre = nn.Identity()
    block.conv = MessageProbe()
    block.lin_post = nn.Identity()
    block.norm_act = nn.Identity()
    block.refine = nn.Identity()

    # Forward
    out = block(node, edge, edge_index, edge_sh, edge_length_emb)

    messages = node[edge_index[0]] + 2 * node[edge_index[1]] + 3 * edge
    expected = torch.zeros_like(node).index_add(0, edge_index[1], messages)
    torch.testing.assert_close(out, expected)
    node_grad, edge_grad = torch.autograd.grad(out.sum(), (node, edge))
    # Node 2 is a source only; node 0 is the receiver of one message; node 1
    # receives two messages and also participates as a source once.
    torch.testing.assert_close(node_grad, torch.tensor([[3.0], [5.0], [1.0]]))
    torch.testing.assert_close(edge_grad, torch.full_like(edge, 3.0))
