import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

from net.activations import make_nonlinearity, scalar_activation


class DummyCFG:
    """Mimic Config subset used by make_nonlinearity."""

    def __init__(self, **kw):
        # defaults
        self.nonlin_kind = "gate"
        self.activation_scalar = "silu"
        self.norm_kind = "component"
        # overrides
        for k, v in kw.items():
            setattr(self, k, v)


IR = Irreps("4x0e + 4x0o + 2x1e + 2x1o")  # simple test irreps


@pytest.mark.unit
def test_mixed_parity_normact_is_rotation_and_inversion_covariant():
    cfg = DummyCFG(
        nonlin_kind="normact",
        activation_scalar="tanh",
        norm_kind="component",
        s2act_res=128,
    )
    mod = make_nonlinearity(IR, cfg)

    torch.manual_seed(727)
    x = torch.randn(8, IR.dim)
    for transform in (rand_matrix(), -torch.eye(3)):
        representation = IR.D_from_matrix(transform)
        transformed_then_activated = mod(x @ representation.T)
        activated_then_transformed = mod(x) @ representation.T
        torch.testing.assert_close(
            transformed_then_activated,
            activated_then_transformed,
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.unit
def test_scalar_activation_errors():
    with pytest.raises(ValueError):
        _ = scalar_activation("nosuchactivation")
