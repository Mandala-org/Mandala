import torch
import pytest
from e3nn.o3 import Irreps

from net.activations import make_nonlinearity, scalar_activation


class DummyHP:
    """Mimic HyperParams subset used by make_nonlinearity."""

    def __init__(self, **kw):
        # defaults
        self.nonlin_kind = "gate"
        self.activation_scalar = "silu"
        self.batch_norm = False
        self.norm_kind = "component"
        # overrides
        for k, v in kw.items():
            setattr(self, k, v)


IR = Irreps("4x0e + 4x0o + 2x1e + 2x1o")  # simple test irreps


@pytest.mark.parametrize("kind", ["gate", "normact", "s2act", "id"])
def test_factory_builds_and_runs(kind):
    hp = DummyHP(nonlin_kind=kind)
    mod = make_nonlinearity(IR, hp)

    x = torch.randn(8, IR.dim, requires_grad=True)
    y = mod(x)
    assert y.shape == x.shape

    # gradient flows
    loss = (y**2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_scalar_activation_errors():
    with pytest.raises(ValueError):
        _ = scalar_activation("nosuchactivation")
