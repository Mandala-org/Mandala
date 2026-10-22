import pytest
import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps
from net.common import RadialMLP, build_hidden_irreps, Config


@pytest.mark.unit
def test_build_hidden_irreps_basic():
    ir = build_hidden_irreps(l_max=2, base_dim=4)
    # 4,4,2,2,1,1 multiplicities expected
    assert ir == Irreps("4x0e + 4x0o + 2x1e + 2x1o + 1x2e + 1x2o")


@pytest.mark.unit
def test_radial_mlp_arbitrary_layers():
    torch.manual_seed(731)
    mlp = RadialMLP(16, out_dim=4, layers=(32, 16, 8))
    x = torch.randn(3, 16, requires_grad=True)
    y = mlp(x)
    assert y.shape == (3, 4)

    # Spell out the three hidden affine/SiLU steps and the final affine map.
    # This catches a missing hidden layer, an activation after the output layer,
    # and wrong layer ordering while retaining an input-gradient check.
    layers = [module for module in mlp.net if isinstance(module, torch.nn.Linear)]
    assert [layer.out_features for layer in layers] == [32, 16, 8, 4]
    expected = x
    for layer in layers[:-1]:
        expected = F.silu(F.linear(expected, layer.weight, layer.bias))
    expected = F.linear(expected, layers[-1].weight, layers[-1].bias)
    torch.testing.assert_close(y, expected)
    actual_grad = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.unit
def test_hyperparams_defaults():
    cfg = Config(safety_checks=True)
    assert cfg.nonlin_kind in {
        "normact",
        "s2act",
        "gate_scalars_mlp",
        "gate_magnitudes",
    }
    assert cfg.e3mlp_variant == "film"
    assert cfg.internal_e3mlp_variant == "resnormact"
    assert cfg.head_e3mlp_variant == "normact"
    assert cfg.loss_coef_observables == 0
    assert cfg.checkpoint_monitor == "val/hamiltonian_mae"
    assert cfg.log_grad_norm is False
