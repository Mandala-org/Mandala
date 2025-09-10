import pytest
import torch
from torch import nn
from unittest.mock import MagicMock

from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from core.block_irrep_mapper import BlockIrrepMapper
from net.e3gnn import E3GNN
from e3nn.o3 import Irreps


class MockHead(nn.Module):
    def __init__(self, mock_impl):
        super().__init__()
        self.mock_impl = mock_impl

    def forward(self, *args, **kwargs):
        return self.mock_impl(*args, **kwargs)


@pytest.mark.unit
def test_forward_smoke():
    # ------- dummy orbital config ------------------
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})

    cfg = Config(num_layers_gnn=1, num_layers_matrix=1)

    model = E3GNN(BlockIrrepMapper(orb_cfg), cfg)

    # ------- fake batch ----------------------------
    N, E = 4, 7
    node_type_idx = torch.zeros(N, dtype=torch.long)  # all H
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # H-H
    edge_len = torch.randn(E, cfg.n_radial)
    sh = Irreps.spherical_harmonics(cfg.l_max_gnn)
    edge_sh = torch.randn(E, sh.dim)  # random SH features

    x = {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx,
        "edge_length_emb": edge_len,
        "edge_sh": edge_sh,
        "edge_index": torch.tensor([[0, 1, 2, 3, 0, 1, 2], [0, 1, 2, 3, 1, 2, 3]]),
        "atoms": ("H", "H", "H", "H"),
        "num_self_edges": N,
        "index_gnn_cutoff": E,
        "is_closest_edge": torch.ones(E, dtype=torch.bool),
        "box": torch.eye(3),
    }
    atoms = ("H", "H", "H", "H")

    preds = model(x)
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}
    for v in preds.values():
        assert v.atoms == atoms


@pytest.mark.unit
def test_pbc_aggregation_closest():
    # Setup
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    cfg = Config(pbc_aggregation="closest")
    mapper = BlockIrrepMapper(orb_cfg)
    model = E3GNN(mapper, cfg)

    # Mock the head
    mock_forward = MagicMock()
    for name in model.heads.keys():
        model.heads[name] = MockHead(mock_forward)

    # Mock input data
    N = 2
    # Edges: (0,0), (1,1), (0,1) closest, (0,1) image
    edge_index = torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1]])
    is_closest_edge = torch.tensor([True, True, True, False])
    num_self_edges = 2

    x = {
        "node_type_idx": torch.zeros(N, dtype=torch.long),
        "edge_type_idx": torch.zeros(4, dtype=torch.long),
        "edge_length_emb": torch.randn(4, cfg.n_radial),
        "edge_sh": torch.randn(4, Irreps.spherical_harmonics(cfg.l_max_gnn).dim),
        "edge_index": edge_index,
        "atoms": ("H", "H"),
        "num_self_edges": num_self_edges,
        "index_gnn_cutoff": 4,
        "is_closest_edge": is_closest_edge,
        "box": torch.eye(3),
    }

    # Action
    model.forward(x)

    # Assertion
    # Check that the head was called with filtered inputs
    mock_forward.assert_called()
    call_args, _ = mock_forward.call_args
    head_embeddings, head_edge_type_idx, head_edge_index = call_args

    assert head_edge_index.shape[1] == 3  # 2 self-edges + 1 closest off-diag
    assert torch.all(head_edge_index == edge_index[:, is_closest_edge])
    assert len(head_edge_type_idx) == 3
    assert len(head_embeddings) == 3  # 2 nodes + 1 closest edge


@pytest.mark.unit
def test_pbc_aggregation_sum():
    # Setup
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    cfg = Config(pbc_aggregation="sum")
    mapper = BlockIrrepMapper(orb_cfg)
    model = E3GNN(mapper, cfg)

    # Mock the head to return predictable values
    v1 = torch.tensor([[1.0]])
    v2 = torch.tensor([[2.0]])
    v_self0 = torch.tensor([[10.0]])
    v_self1 = torch.tensor([[20.0]])

    # The head will be called with all 4 edges.
    # The mock needs to return values for each edge type.
    # In this test, all edges are H-H.
    raw_preds = {
        "H-H": {
            "vectors": torch.cat([v_self0, v_self1, v1, v2], dim=0),
            "edges": torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1]]),
        }
    }
    mock_forward = MagicMock(return_value=raw_preds)
    for name in model.heads.keys():
        model.heads[name] = MockHead(mock_forward)

    # Mock input data
    N = 2
    edge_index = torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1]])
    is_closest_edge = torch.tensor([True, True, True, False])
    num_self_edges = 2

    x = {
        "node_type_idx": torch.zeros(N, dtype=torch.long),  # all H
        "edge_type_idx": torch.zeros(4, dtype=torch.long),  # all H-H
        "edge_length_emb": torch.randn(4, cfg.n_radial),
        "edge_sh": torch.randn(4, Irreps.spherical_harmonics(cfg.l_max_gnn).dim),
        "edge_index": edge_index,
        "atoms": ("H", "H"),
        "num_self_edges": num_self_edges,
        "index_gnn_cutoff": 4,
        "is_closest_edge": is_closest_edge,
        "box": torch.eye(3),
    }

    # Action
    preds = model.forward(x)

    # Assertion
    for name in preds.keys():
        result_vectors = preds[name].pair_vectors["H-H"]
        result_edges = preds[name].pair_edges["H-H"]

        assert result_vectors.shape[0] == 3  # 3 unique pairs: (0,0), (1,1), (0,1)

        # Find the summed vector for (0,1)
        for i in range(result_edges.shape[1]):
            edge = tuple(result_edges[:, i].tolist())
            if edge == (0, 1):
                assert torch.allclose(result_vectors[i], v1 + v2)
            elif edge == (0, 0):
                assert torch.allclose(result_vectors[i], v_self0)
            elif edge == (1, 1):
                assert torch.allclose(result_vectors[i], v_self1)
