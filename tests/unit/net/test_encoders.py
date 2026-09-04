import pytest
import torch
from e3nn.o3 import Irreps

from net.common import Config, build_hidden_irreps
from net.encoders import NodeEncoder, EdgeEncoder


@pytest.mark.unit
def test_node_encoder_shape_no_diag():
    cfg = Config(safety_checks=True)

    enc = NodeEncoder(
        node_one_hot_dim=4,
        cfg=cfg,
    )

    node_type_idx = torch.arange(4, dtype=torch.long)
    out = enc(node_type_idx)
    assert out.shape == (4, enc.irreps_out.dim)


@pytest.mark.unit
def test_edge_encoder_with_offdiag():
    cfg = Config(radial_layers=(64, 32), safety_checks=True)
    hid = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)

    enc = EdgeEncoder(
        n_edge_types=3,
        irreps_out=hid,
        cfg=cfg,
    )

    E = 7
    edge_type_idx = torch.randint(0, 3, (E,))
    length_emb = torch.randn(E, cfg.n_radial)
    edge_length = torch.rand(E) * cfg.cutoff_radius
    sh = torch.randn(E, sh_irreps.dim)
    out = enc(edge_type_idx, length_emb, sh, edge_length)
    assert out.shape == (E, hid.dim)
