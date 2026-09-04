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
    # Test EdgeUpdateBlock edge unpacking
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
    node = torch.randn(2, 1)
    edge = torch.randn(1, 1)
    edge_index = torch.tensor([[0], [1]])  # src=0, dst=1
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(1, sh_irreps.dim)
    edge_length_emb = torch.randn(1, cfg.n_radial)
    edge_length = torch.rand(1) * cfg.cutoff_radius

    # Forward
    out = block(node, edge, edge_index, edge_sh, edge_length_emb, edge_length)

    assert out.shape == (1, 1)


def test_node_update_block():
    # Test NodeUpdateBlock edge unpacking
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
    node = torch.randn(2, 1)
    edge = torch.randn(1, 1)
    edge_index = torch.tensor([[0], [1]])  # src=0, dst=1
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(1, sh_irreps.dim)
    edge_length_emb = torch.randn(1, cfg.n_radial)
    edge_length = torch.rand(1) * cfg.cutoff_radius

    # Forward
    out = block(node, edge, edge_index, edge_sh, edge_length_emb, edge_length)

    assert out.shape == (2, 1)
