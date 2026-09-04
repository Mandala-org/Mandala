import pytest
import torch
from e3nn.nn import NormActivation
from e3nn.o3 import Irreps, rand_matrix

from net.activations import GateScalarsMLP, GateMagnitudes, make_nonlinearity
from net.common import Config


@pytest.fixture
def sample_irreps():
    return Irreps("16x0e+8x1o+4x2e")


@pytest.fixture
def sample_tensor(sample_irreps):
    return torch.randn(5, sample_irreps.dim)


@pytest.mark.unit
def test_gate_scalars_mlp(sample_irreps, sample_tensor):
    """Tests the GateScalarsMLP activation."""
    nonlin = torch.nn.SiLU()
    activation = GateScalarsMLP(sample_irreps, nonlin)
    output = activation(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert not torch.allclose(output, sample_tensor)


@pytest.mark.unit
def test_gate_magnitudes(sample_irreps, sample_tensor):
    """Tests the GateMagnitudes activation."""
    nonlin_scalars = torch.nn.SiLU()
    nonlin_magnitudes = torch.nn.Softplus()
    activation = GateMagnitudes(sample_irreps, nonlin_scalars, nonlin_magnitudes)
    output = activation(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert not torch.allclose(output, sample_tensor)


@pytest.mark.parametrize("kind", ["normact", "gate_scalars_mlp", "gate_magnitudes"])
@pytest.mark.unit
def test_make_nonlinearity_factory(sample_irreps, sample_tensor, kind):
    """Tests the make_nonlinearity factory for all new kinds."""
    cfg = Config(nonlin_kind=kind, safety_checks=True)
    activation = make_nonlinearity(sample_irreps, cfg)
    output = activation(sample_tensor)

    assert output.shape == sample_tensor.shape


@pytest.mark.unit
def test_make_nonlinearity_rejects_removed_s2act():
    with pytest.raises(ValueError, match="Unsupported nonlin_kind 's2act'"):
        Config(nonlin_kind="s2act", safety_checks=True)


@pytest.mark.unit
def test_removed_nonlinearity_config_fields_are_rejected():
    with pytest.raises(TypeError):
        Config(s2act_res=64)
    with pytest.raises(TypeError):
        Config(norm_kind="component")


@pytest.mark.unit
def test_norm_activation_uses_explicit_norm_normalization():
    activation = make_nonlinearity(Irreps("2x0e+2x1o"), Config(nonlin_kind="normact"))
    assert isinstance(activation, NormActivation)
    assert activation.normalize is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "irreps",
    ["64x0e+8x0o+16x1o+8x2e", "32x0e+16x0e+8x1o"],
)
def test_gate_scalars_mlp_accepts_even_scalar_prefix(irreps):
    GateScalarsMLP(Irreps(irreps), torch.nn.SiLU())


@pytest.mark.unit
@pytest.mark.parametrize(
    "irreps",
    ["16x1o+32x0e+8x2e", "32x0e+8x1o+16x0e"],
)
def test_gate_scalars_mlp_rejects_late_even_scalars(irreps):
    with pytest.raises(ValueError, match="contiguous prefix"):
        GateScalarsMLP(Irreps(irreps), torch.nn.SiLU())


@pytest.mark.unit
@pytest.mark.parametrize("magnitude", [0.0, 1.0e-12, 0.7])
def test_gate_magnitudes_is_finite_at_zero_and_backward(magnitude):
    irreps = Irreps("2x0e+2x0o+2x1e+2x2o")
    x = torch.full((3, irreps.dim), magnitude, requires_grad=True)
    activation = GateMagnitudes(irreps, torch.nn.SiLU(), torch.nn.Softplus())
    y = activation(x)
    y.square().sum().backward()
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.unit
def test_gate_magnitudes_o3_equivariance():
    irreps = Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o")
    activation = GateMagnitudes(irreps, torch.nn.SiLU(), torch.nn.Softplus())
    x = irreps.randn(4, -1)
    transforms = [
        rand_matrix(),
        -torch.eye(3),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0])),
    ]
    improper = rand_matrix()
    improper[:, 0] *= -1
    transforms.append(improper)
    for transform in transforms:
        d = irreps.D_from_matrix(transform)
        assert torch.allclose(
            activation(x @ d.T), activation(x) @ d.T, atol=2e-5, rtol=2e-5
        )


@pytest.mark.unit
def test_make_nonlinearity_invalid_kind(sample_irreps):
    """Tests that the factory raises an error for an invalid kind."""
    with pytest.raises(ValueError, match="Unsupported nonlin_kind"):
        Config(nonlin_kind="invalid_kind", safety_checks=True)
