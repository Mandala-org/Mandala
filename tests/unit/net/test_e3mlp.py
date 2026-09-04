import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

from net.common import Config, E3MLP
from net.e3mlp_variants import InvariantFiLMActivation, IrrepCopyScale

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


def o3_transforms() -> list[torch.Tensor]:
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    improper = rand_matrix()
    improper[:, 0] *= -1.0
    transforms = [rand_matrix(), -torch.eye(3), reflection, improper]
    assert [round(torch.det(r).item()) for r in transforms] == [1, -1, -1, -1]
    return transforms


def assert_o3_equivariant(module, irreps: Irreps, *, atol: float = 3e-5) -> None:
    x = irreps.randn(5, -1)
    y = module(x)
    for transform in o3_transforms():
        d = irreps.D_from_matrix(transform)
        assert torch.allclose(
            module(x @ d.T), y @ d.T, atol=atol, rtol=atol
        ), f"failed for det={torch.det(transform).item():.0f}"


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


@pytest.mark.unit
def test_invariant_film_is_o3_equivariant_with_even_and_odd_scalars():
    torch.manual_seed(2026)
    irreps = Irreps("2x0e+3x0o+2x1e+2x1o+2x2e+2x2o")
    film = InvariantFiLMActivation(irreps, hidden_dim=19)
    # Avoid relying on any special initialization of the final FiLM map.
    with torch.no_grad():
        for parameter in film.parameters():
            parameter.normal_(mean=0.17, std=0.63)
    assert_o3_equivariant(film, irreps, atol=3e-4)


@pytest.mark.unit
def test_irrep_copy_scale_broadcasts_over_m_and_is_o3_equivariant():
    torch.manual_seed(2027)
    irreps = Irreps("2x0e+3x0o+2x1e+3x1o+2x2e+3x2o")
    scale = IrrepCopyScale(irreps, initial_value=1.0)
    with torch.no_grad():
        scale.scale.copy_(torch.linspace(-1.3, 2.1, irreps.num_irreps))
    assert scale.scale.numel() == irreps.num_irreps
    assert_o3_equivariant(scale, irreps)


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["resnormact", "resgatemagnitudes", "bilinear"])
def test_residual_and_bilinear_scales_remain_o3_equivariant_when_perturbed(variant):
    torch.manual_seed(2028)
    irreps = Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o")
    mlp = E3MLP(
        input_irreps=irreps,
        hidden_irreps=irreps,
        output_irreps=irreps,
        num_layers=2,
        cfg=Config(e3mlp_variant=variant, safety_checks=True),
        activate_last=True,
    )
    cursor = 0
    with torch.no_grad():
        for scale in (m for m in mlp.modules() if isinstance(m, IrrepCopyScale)):
            values = torch.linspace(-0.9 + cursor, 1.4 + cursor, scale.scale.numel())
            scale.scale.copy_(values)
            cursor += 1
    assert cursor > 0
    assert_o3_equivariant(mlp, irreps, atol=8e-5)
