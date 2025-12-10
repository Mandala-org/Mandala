import torch
from torch import nn
import pytorch_lightning as pl
from net.e3gnn import E3GNN
from net.layers import EdgeUpdateBlock, NodeUpdateBlock
from net.common import Config
from e3nn.o3 import Irreps
from data.block_matrix import IrrepsBlockData
from core.orbital_irrep_config import OrbitalIrrepConfig


def test_e3gnn_forward_edge_slicing():
    # Test that forward slices edges correctly into small and large

    # Mock Config
    cfg = Config(
        precompute_edge_features=True,  # Assume precomputed for simplicity
        cutoff_matrix=5.0,
        cutoff_gnn=3.0,
        hidden_base_dim=4,
        edge_type_emb_dim=4,
        l_max_gnn=1,
        l_max_matrix=1,
    )
    # E3GNN uses cfg.r_max and cfg.r_max_gnn which are aliases or properties?
    # Let's check E3GNN code. It uses self.cfg.r_max.
    # But Config definition in common.py has cutoff_matrix.
    # Maybe E3GNN aliases them? Or maybe I should check Config again.
    # Config has cutoff_gnn and cutoff_matrix.
    # E3GNN.py:
    # (..., index_gnn_cutoff, ...) = compute_graph_features(..., self.cfg.r_max, self.cfg.r_max_gnn, ...)
    # Wait, in my previous read of e3gnn.py, it was:
    # compute_graph_features(..., cfg=self.cfg, ...)
    # So it passes cfg object.
    # And compute_graph_features uses cfg.cutoff_matrix.

    # So E3GNN forward uses:
    # if not self.cfg.precompute_edge_features:
    #    compute_graph_features(..., cfg=self.cfg, ...)

    # But wait, I am mocking E3GNN forward logic?
    # No, I am calling model.forward(x).
    # And model.forward calls compute_graph_features ONLY if precompute_edge_features is False.
    # In my test I set precompute_edge_features=True.
    # So compute_graph_features is NOT called.
    # But forward uses `index_gnn_cutoff` from x.

    # Then forward does:
    # ei_small = x["edge_index"][:, num_self_edges:index_gnn_cutoff]

    # So I don't need r_max in Config if I don't call compute_graph_features.
    # But I need to fix the Config init in the test anyway.

    # Also fix num_layers=1 for MLPs.

    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

    class MockE3GNN(E3GNN):
        def __init__(self, cfg, orbital_cfg):
            super(pl.LightningModule, self).__init__()  # Skip E3GNN init
            self.cfg = cfg
            self.mapper = type(
                "MockMapper",
                (),
                {"orbital_cfg": orbital_cfg, "edge_type2idx": {"H-H": 0}},
            )()
            self.sh_irreps = Irreps("1x0e")
            # Mock submodules called in forward
            self.node_enc = lambda *args, **kwargs: torch.randn(2, 4)
            self.edge_enc = lambda *args, **kwargs: torch.randn(3, 4)  # 3 edges
            self.layers = nn.ModuleList(
                [
                    type(
                        "MockLayer",
                        (nn.Module,),
                        {"forward": lambda s, n, e, ei, **kwargs: (n, e)},
                    )()
                ]
            )
            self.mp_small = self.layers
            self.mp_large = nn.ModuleList([])
            self.heads = nn.ModuleDict(
                {
                    "density": type(
                        "MockHead",
                        (nn.Module,),
                        {
                            "forward": lambda *args: {
                                "H-H": {
                                    "vectors": torch.randn(1, 1),
                                    "edges": torch.randn(5, 1),
                                }
                            }
                        },
                    )()
                }
            )
            self._activation_mags = {}

    model = MockE3GNN(cfg, orbital_cfg)

    # Input x
    # Edges:
    # 0. Self (dist 0) -> Small
    # 1. Short (dist 2) -> Small (cutoff_gnn=3)
    # 2. Long (dist 4) -> Large (cutoff_matrix=5)

    edge_index = torch.tensor([[0, 0], [0, 1], [0, 1]]).t()  # Self  # Short  # Long
    edge_shift = torch.zeros(3, 3)

    # index_gnn_cutoff should be 2 (first 2 edges are <= 3.0)
    index_gnn_cutoff = 2
    num_self_edges = 1

    x = {
        "node_type_idx": torch.zeros(2, dtype=torch.long),
        "positions": torch.randn(2, 3),
        "box": torch.eye(3),
        "atoms": ("H", "H"),
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_type_idx": torch.zeros(3, dtype=torch.long),
        "edge_length_emb": torch.randn(3, 5),
        "edge_sh": torch.randn(3, 1),
        "index_gnn_cutoff": index_gnn_cutoff,
        "num_self_edges": num_self_edges,
    }

    captured_ei = []
    # Capture the edge_index passed to the first layer
    model.layers[0].forward = lambda n, e, ei, **kwargs: (
        captured_ei.append(ei),
        (n, e),
    )[1]

    model.forward(x)

    assert len(captured_ei) == 1
    ei = captured_ei[0]

    # ei passed to layer should be `ei_small`.
    # ei_small = edge_index[:, num_self_edges:index_gnn_cutoff]
    # num_self=1, index_gnn=2. Slice [1:2].
    # Should contain edge 1 (Short).

    assert ei.shape[1] == 1
    assert torch.equal(ei, edge_index[:, 1:2])


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
