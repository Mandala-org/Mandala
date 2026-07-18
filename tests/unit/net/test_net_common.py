import pytest
import torch
from e3nn.o3 import Irreps
from net.common import RadialMLP, build_hidden_irreps, Config


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
        "s2act",
        "gate_scalars_mlp",
        "gate_magnitudes",
    }
    assert cfg.e3mlp_variant == "film"
    assert cfg.internal_e3mlp_variant == "resnormact"
    assert cfg.head_e3mlp_variant == "normact"
    assert cfg.loss_coef_observables == 0
    assert cfg.checkpoint_monitor == "val/hamiltonian_mae"
