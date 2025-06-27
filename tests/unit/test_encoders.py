import pytest
import torch
from e3nn.o3 import Irreps

from net.common import HyperParams, build_hidden_irreps
from net.encoders import NodeEncoder, EdgeEncoder


@pytest.mark.unit
def test_node_encoder_shape_no_diag():
    hp = HyperParams()
    hid = build_hidden_irreps(hp.l_max, hp.hidden_base_dim)

    enc = NodeEncoder(
        node_one_hot_dim=4,
        out_irreps=hid,
        hp=hp,
    )

    one_hot = torch.eye(4)
    out = enc(one_hot)
    assert out.shape == (4, hid.dim)


@pytest.mark.unit
def test_edge_encoder_with_offdiag():
    hp = HyperParams(radial_layers=(64, 32))
    hid = build_hidden_irreps(hp.l_max, hp.hidden_base_dim)
    sh_irreps = Irreps.spherical_harmonics(hp.l_max)

    enc = EdgeEncoder(
        n_edge_types=3,
        n_radial=hp.n_radial,
        sh_irreps=sh_irreps,
        offdiag_irrep_dim=10,
        out_irreps=hid,
        hp=hp,
    )

    E = 7
    one_hot = torch.nn.functional.one_hot(torch.randint(0, 3, (E,)), 3).float()
    length_emb = torch.randn(E, hp.n_radial)
    sh = torch.randn(E, sh_irreps.dim)
    off = torch.randn(E, 10)

    out = enc(one_hot, length_emb, sh, off)
    assert out.shape == (E, hid.dim)
