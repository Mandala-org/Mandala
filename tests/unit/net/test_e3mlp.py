import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

from net.common import Config, E3MLP
from net.e3mlp_variants import (
    InvariantFiLMActivation,
    GateMagnitudesActivation,
    EquivariantRMSNorm,
)


@pytest.mark.parametrize(
    "activation",
    [InvariantFiLMActivation, GateMagnitudesActivation, EquivariantRMSNorm],
)
def test_zero_irrep_channels_have_finite_higher_order_parameter_gradients(activation):
    torch.manual_seed(734)
    module = activation(Irreps("1x0e+1x1o")).double()
    # Structurally zero angular channels are common for onsite output features.
    x = torch.tensor(
        [[0.3, 0.0, 0.0, 0.0], [0.7, 0.2, -0.4, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = module(x)
    probe = torch.linspace(0.2, 1.3, output.numel(), dtype=x.dtype).reshape_as(output)
    gradient = torch.autograd.grad((output * probe).sum(), x, create_graph=True)[0]
    gradients = torch.autograd.grad(
        gradient.square().sum(), (x, *module.parameters()), allow_unused=True
    )
    assert torch.isfinite(gradient).all()
    assert any(value is not None and value.abs().max() > 0 for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients if value is not None)


VARIANTS = [
    "basic",
    "normact",
    "gate",
    "gatemagnitudes",
    "film",
    "resnormact",
    "resgatemagnitudes",
    "bilinear",
]


def _cfg_for_variant(variant: str) -> Config:
    if variant == "basic":
        return Config(
            e3mlp_variant="basic", nonlin_kind="gate_magnitudes", safety_checks=True
        )
    return Config(e3mlp_variant=variant, safety_checks=True)


def random_rotation_matrix() -> torch.Tensor:
    return rand_matrix()


def generate_equivariant_input(irreps: Irreps, batch_size: int = 1) -> torch.Tensor:
    return irreps.randn(batch_size, -1)


@pytest.mark.unit
def test_e3mlp_min_layers_assertion():
    """Tests that E3MLP raises an error if num_layers < 1."""
    with pytest.raises(ValueError, match="at least 1 layer"):
        E3MLP(
            input_irreps=Irreps("1x0e"),
            hidden_irreps=Irreps("2x0e"),
            output_irreps=Irreps("1x0e"),
            num_layers=0,
            cfg=Config(),
        )


@pytest.mark.unit
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("activate_last", [True, False])
def test_e3mlp_equivariance(variant, num_layers, activate_last):
    """Tests the E3MLP for rotational equivariance."""
    cfg = _cfg_for_variant(variant)
    input_irreps = Irreps("5x0e + 5x1o")
    hidden_irreps = Irreps("10x0e + 10x1o + 5x2e")
    output_irreps = Irreps("3x0e + 3x1o")
    batch_size = 2

    mlp = E3MLP(
        input_irreps=input_irreps,
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        num_layers=num_layers,
        cfg=cfg,
        activate_last=activate_last,
    )

    # 1. Generate random input and rotation
    x = generate_equivariant_input(input_irreps, batch_size=batch_size)
    R = random_rotation_matrix()
    D_in = input_irreps.D_from_matrix(R)
    D_out = output_irreps.D_from_matrix(R)

    # 2. Transform input and pass through MLP
    x_rotated = x @ D_in.T
    y_from_rotated_input = mlp(x_rotated)

    # 3. Pass original input and transform output
    y_original = mlp(x)
    y_rotated_from_original = y_original @ D_out.T

    assert y_original.shape == (batch_size, output_irreps.dim)

    # 4. Check for equivariance
    assert torch.allclose(y_from_rotated_input, y_rotated_from_original, atol=1e-5)


@pytest.mark.unit
def test_e3mlp_invalid_variant():
    cfg = Config(e3mlp_variant="not_a_variant", safety_checks=True)
    with pytest.raises(ValueError, match="Unsupported E3MLP variant"):
        E3MLP(
            input_irreps=Irreps("1x0e"),
            hidden_irreps=Irreps("2x0e"),
            output_irreps=Irreps("1x0e"),
            num_layers=2,
            cfg=cfg,
        )
