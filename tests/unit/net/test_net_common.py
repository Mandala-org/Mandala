import pytest
import torch
from e3nn.o3 import Irreps
from net.common import RadialMLP, build_hidden_irreps, Config, smooth_cutoff


@pytest.mark.unit
def test_build_hidden_irreps_basic():
    ir = build_hidden_irreps(l_max=2, base_dim=4)
    # 4,4,2,2,1,1 multiplicities expected
    assert ir == Irreps("4x0e + 4x0o + 2x1e + 2x1o + 1x2e + 1x2o")


@pytest.mark.unit
def test_radial_mlp_arbitrary_layers():
    mlp = RadialMLP(16, out_dim=4, layers=(32, 16, 8))
    x = torch.randn(3, 16)
    y = mlp(x)
    assert y.shape == (3, 4)


@pytest.mark.unit
def test_hyperparams_defaults():
    cfg = Config(safety_checks=True)
    assert cfg.nonlin_kind in {
        "normact",
        "gate_scalars_mlp",
        "gate_magnitudes",
    }
    assert cfg.e3mlp_variant == "film"
    assert cfg.internal_e3mlp_variant == "resnormact"
    assert cfg.head_e3mlp_variant == "normact"
    assert cfg.loss_coef_observables == 0
    assert cfg.checkpoint_monitor == "val/hamiltonian_mae"
    assert cfg.log_grad_norm is False


@pytest.mark.unit
def test_smooth_cutoff_value_and_first_derivative_vanish_at_radius():
    radius = 5.0
    r = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
    value = smooth_cutoff(r, radius)
    derivative = torch.autograd.grad(value, r, create_graph=True)[0]
    second_derivative = torch.autograd.grad(derivative, r)[0]
    assert value.item() == pytest.approx(0.0, abs=1e-14)
    assert derivative.item() == pytest.approx(0.0, abs=1e-14)
    assert second_derivative.item() == pytest.approx(0.0, abs=1e-13)


@pytest.mark.unit
def test_smooth_cutoff_is_finite_at_zero_and_just_below_cutoff():
    values = smooth_cutoff(torch.tensor([0.0, 5.0 - 1e-5]), 5.0)
    assert torch.isfinite(values).all()
    assert values[0].item() == pytest.approx(1.0)
    assert values[1].item() > 0.0


@pytest.mark.unit
def test_smooth_cutoff_rejects_nonpositive_radius():
    with pytest.raises(ValueError, match="positive"):
        smooth_cutoff(torch.tensor(1.0), 0.0)
