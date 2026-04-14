import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

from net.common import Config, E3MLP

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
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("activate_last", [True, False])
def test_e3mlp_creation_and_forward(variant, num_layers, activate_last):
    """Tests that the E3MLP is created and runs a forward pass with correct shapes."""
    cfg = _cfg_for_variant(variant)
    input_irreps = Irreps("3x0e + 4x1o")
    hidden_irreps = Irreps("16x0e + 8x1o + 4x2e")
    output_irreps = Irreps("5x0e + 2x1o")
    batch_size = 2

    mlp = E3MLP(
        input_irreps=input_irreps,
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        num_layers=num_layers,
        cfg=cfg,
        activate_last=activate_last,
    )

    x = generate_equivariant_input(input_irreps, batch_size=batch_size)
    y = mlp(x)

    assert y.shape == (batch_size, output_irreps.dim)


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
