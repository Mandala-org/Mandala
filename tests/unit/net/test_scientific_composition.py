"""Matched composites and independently scheduled primitives, including training."""

import copy
import math

from e3nn import o3
import pytest
import torch

from net.transforms.composition import ProjectedAttention
from tests.contract_support import double_precision


def _compare_training(left, right, left_forward, right_forward, inputs):
    """Compare every used parameter/input VJP and two Adam updates."""
    left_opt = torch.optim.Adam(left.parameters(), lr=0.013)
    right_opt = torch.optim.Adam(right.parameters(), lr=0.013)
    changed = False
    for step in range(2):
        a = [x.detach().clone().requires_grad_() for x in inputs]
        b = [x.detach().clone().requires_grad_() for x in inputs]
        ya, yb = left_forward(left, a), right_forward(right, b)
        torch.testing.assert_close(ya, yb, atol=2e-11, rtol=2e-10)
        probe = torch.linspace(-0.7, 1.3, ya.numel()).reshape_as(ya)
        la = (ya * probe).sum() + (0.1 + step * 0.03) * ya.square().sum()
        lb = (yb * probe).sum() + (0.1 + step * 0.03) * yb.square().sum()
        left_opt.zero_grad()
        right_opt.zero_grad()
        la.backward()
        lb.backward()
        for xa, xb in zip(a, b):
            assert xa.grad is not None and torch.isfinite(xa.grad).all()
            assert xa.grad.abs().max() > 1e-10
            torch.testing.assert_close(xa.grad, xb.grad, atol=2e-10, rtol=2e-9)
        # Callers arrange identical parameter order explicitly, not by relying
        # on equal module registration order in differently composed graphs.
        pa, pb = left._contract_parameters(), right._contract_parameters()
        assert len(pa) == len(pb)
        assert {id(p) for p in pa} == {id(p) for p in left.parameters()}
        assert {id(p) for p in pb} == {id(p) for p in right.parameters()}
        before = [p.detach().clone() for p in pa]
        for x, y in zip(pa, pb):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                assert torch.isfinite(x.grad).all()
                torch.testing.assert_close(x.grad, y.grad, atol=2e-10, rtol=2e-9)
        assert sum(p.grad is not None and p.grad.abs().max() > 1e-10 for p in pa) >= 4
        left_opt.step()
        right_opt.step()
        for old, x, y in zip(before, pa, pb):
            torch.testing.assert_close(x, y, atol=2e-11, rtol=2e-10)
            changed |= not torch.equal(old, x)
    assert changed


@pytest.mark.parametrize("receiver", [0, 1])
def test_attention_matches_per_receiver_per_head_softmax_and_all_vjps(receiver):
    with double_precision():
        module = ProjectedAttention(
            o3.Irreps("3x0e+1x1o"),
            o3.Irreps("2x0e+2x1o"),
            3,
            scalar_dim=6,
            heads=3,
            receiver=receiver,
        ).double()
        oracle = copy.deepcopy(module)
        module._contract_parameters = lambda: list(module.parameters())
        oracle._contract_parameters = lambda: list(oracle.parameters())
        index = torch.tensor([[0, 0, 0, 1, 2, 2, 3], [1, 2, 3, 0, 0, 1, 0]])
        values = [torch.randn(7, 8), torch.randn(5, 6), torch.randn(7, 3)]

        def expected(m, v):
            messages, nodes, radial = v
            q = m.query_proj(nodes).reshape(5, 3, 2)
            k = m.key_proj(torch.cat([radial, messages], -1)).reshape(7, 3, 2)
            rows = []
            for node in range(5):
                selected = torch.where(index[receiver] == node)[0]
                heads = []
                for head in range(3):
                    scores = (q[node, head] * k[selected, head]).sum(-1) / math.sqrt(2)
                    weight = torch.softmax(scores, dim=0)
                    projected = m.value_projs[head](messages[selected])
                    heads.append((weight[:, None] * projected).sum(0))
                rows.append(torch.stack(heads).mean(0))
            return torch.stack(rows)

        def observed(m, v):
            return m(*v, index)

        _compare_training(module, oracle, observed, expected, values)
