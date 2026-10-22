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

    with torch.no_grad():
        enc.elem_emb.weight.copy_(
            torch.arange(4 * cfg.hidden_base_dim, dtype=torch.float32).reshape(
                4, cfg.hidden_base_dim
            )
        )
    node_type_idx = torch.tensor([3, 1, 3, 0], dtype=torch.long)
    out = enc(node_type_idx)
    assert out.shape == (4, enc.irreps_out.dim)
    expected = torch.tensor(
        [
            enc.elem_emb.weight[3].tolist(),
            enc.elem_emb.weight[1].tolist(),
            enc.elem_emb.weight[3].tolist(),
            enc.elem_emb.weight[0].tolist(),
        ]
    )
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    expected_grad = torch.zeros_like(enc.elem_emb.weight)
    expected_grad[0] = 1
    expected_grad[1] = 1
    expected_grad[3] = 2
    torch.testing.assert_close(enc.elem_emb.weight.grad, expected_grad)


@pytest.mark.unit
def test_edge_encoder_with_offdiag():
    cfg = Config(
        radial_layers=(64, 32),
        safety_checks=True,
        e3layernorm=False,
        edge_encoder_style="rich",
        edge_encoder_use_sh_tensor_square=False,
        n_radial=4,
        l_max=1,
        edge_type_emb_dim=3,
    )
    hid = build_hidden_irreps(cfg.l_max, cfg.hidden_base_dim)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)

    enc = EdgeEncoder(
        n_edge_types=4,
        irreps_out=hid,
        cfg=cfg,
    )

    E = 4
    edge_type_idx = torch.tensor([2, 0, 2, 1])
    length_emb = torch.randn(E, cfg.n_radial, requires_grad=True)
    sh = torch.randn(E, sh_irreps.dim, requires_grad=True)
    out = enc(edge_type_idx, length_emb, sh)
    assert out.shape == (E, hid.dim)

    # Independent input assembly pins edge-type routing and the two feature
    # channels consumed by the rich tensor-product encoder.
    expected = enc.tp(
        torch.nn.functional.embedding(edge_type_idx, enc.edge_emb.weight),
        torch.cat((length_emb, sh), dim=-1),
    )
    torch.testing.assert_close(out, expected)
    actual_grads = torch.autograd.grad(
        out.square().sum(), (enc.edge_emb.weight, length_emb, sh)
    )
    reference_embedding = enc.edge_emb.weight.detach().clone().requires_grad_(True)
    reference_length = length_emb.detach().clone().requires_grad_(True)
    reference_sh = sh.detach().clone().requires_grad_(True)
    reference_out = enc.tp(
        torch.nn.functional.embedding(edge_type_idx, reference_embedding),
        torch.cat((reference_length, reference_sh), dim=-1),
    )
    reference_grads = torch.autograd.grad(
        reference_out.square().sum(),
        (reference_embedding, reference_length, reference_sh),
    )
    for actual, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, reference)
    assert torch.count_nonzero(actual_grads[0][3]) == 0
