import pytest
import torch
from e3nn.o3 import Irreps

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
def test_make_nonlinearity_s2act():
    """Tests the S2Activation factory with valid irreps."""
    irreps = Irreps("1x0e+1x1o+1x2e")
    tensor = torch.randn(5, irreps.dim)
    cfg = Config(nonlin_kind="s2act", safety_checks=True)
    activation = make_nonlinearity(irreps, cfg)
    output = activation(tensor)
    assert output.shape == tensor.shape


@pytest.mark.unit
def test_make_nonlinearity_invalid_kind(sample_irreps):
    """Tests that the factory raises an error for an invalid kind."""
    cfg = Config(nonlin_kind="invalid_kind", safety_checks=True)
    with pytest.raises(ValueError, match="Unknown nonlin_kind"):
        make_nonlinearity(sample_irreps, cfg)
