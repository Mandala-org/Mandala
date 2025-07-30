import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config, build_hidden_irreps
from net.heads import DeepHead
from core.block_irrep_mapper import BlockIrrepMapper


@pytest.mark.unit
def test_deep_head_shapes_and_device():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    pair_keys = ["H-H"]
    cfg = Config(head_depth=2, head_hidden_mul=0.8, dropout=0.1)

    hid = build_hidden_irreps(cfg.l_max_gnn, cfg.hidden_base_dim)

    mapper = BlockIrrepMapper(orb_cfg)
    head = DeepHead(
        in_irreps=hid,
        pair_keys=pair_keys,
        mapper=mapper,
        cfg=cfg,
    )

    E = 3
    edge_feat = torch.randn(E, hid.dim)
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # all "H-H"
    edge_index = torch.vstack([torch.arange(E), torch.flip(torch.arange(E), dims=[0])])

    out = head(edge_feat, edge_type_idx, edge_index)
    assert "H-H" in out
    vec = out["H-H"]["vectors"]
    edges = out["H-H"]["edges"]

    # correct shapes
    assert vec.shape[0] == E
    assert edges.shape == (2, E)
    # device consistency
    assert vec.device == torch.device("cpu")
