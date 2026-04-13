import pytest
import torch
from e3nn.o3 import Irreps

from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config, build_hidden_irreps
from net.heads import DeepHead
from core.block_irrep_mapper import BlockIrrepMapper


@pytest.mark.unit
def test_deep_head_shapes_and_device():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    pair_keys = ["H-H"]
    cfg = Config(neck_depth=2, dropout=0.1, safety_checks=True)

    hid = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    neck = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)

    mapper = BlockIrrepMapper(orb_cfg)
    head = DeepHead(
        irreps_diag_in=hid,
        irreps_edge_in=hid,
        irreps_neck=neck,
        pair_keys=pair_keys,
        mapper=mapper,
        cfg=cfg,
    )

    E = 3
    edge_feat = torch.randn(E, hid.dim)
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # all "H-H"
    edge_index = torch.vstack([torch.arange(E), torch.flip(torch.arange(E), dims=[0])])
    edge_shift = torch.zeros(3, E, dtype=torch.long)
    edges_5d = torch.cat([edge_shift, edge_index], dim=0)

    out = head(edge_feat, edge_feat, edge_type_idx, edges_5d)
    assert "H-H" in out
    vec = out["H-H"]["vectors"]
    edges = out["H-H"]["edges"]

    # correct shapes
    assert vec.shape[0] == E
    assert edges.shape == (5, E)
    # device consistency
    assert vec.device == torch.device("cpu")


@pytest.mark.unit
def test_deep_head_splits_diag_shifted_self_and_offdiag():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)
    cfg = Config(
        neck_depth=1,
        head_depth=1,
        head_use_node_embeddings_for_self_edges=True,
        separate_shifted_self=True,
        safety_checks=True,
    )
    head = DeepHead(
        irreps_diag_in=Irreps("1x0e"),
        irreps_edge_in=Irreps("2x0e"),
        irreps_neck=Irreps("2x0e"),
        pair_keys=["H-H"],
        mapper=mapper,
        cfg=cfg,
    )

    node_feat = torch.randn(2, 1)
    edge_feat = torch.randn(3, 2)
    edge_type_idx = torch.zeros(3, dtype=torch.long)
    edges_5d = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )

    out = head(node_feat, edge_feat, edge_type_idx, edges_5d)

    assert "H-H" in out
    assert out["H-H"]["vectors"].shape[0] == 3
    returned_edges = {tuple(edge.tolist()) for edge in out["H-H"]["edges"].t()}
    expected_edges = {tuple(edge.tolist()) for edge in edges_5d.t()}
    assert returned_edges == expected_edges
