import pytest
import torch
import torch.nn.functional as F
from e3nn.nn import NormActivation, S2Activation
from e3nn.o3 import Irreps, rand_matrix

from net.activations import GateScalarsMLP, GateMagnitudes, make_nonlinearity
from net.common import Config


@pytest.fixture
def sample_irreps():
    # The scalar invariant precedes the vector/tensor channels, and every
    # parity is compatible with O(3): p=(-1)^l.
    return Irreps("2x0e+1x1o+1x2e")


@pytest.fixture
def sample_tensor(sample_irreps):
    torch.manual_seed(731)
    return torch.randn(5, sample_irreps.dim)


def test_gate_scalars_mlp_uses_scalar_activation_and_gates_each_irrep():
    irreps = Irreps("2x0e+2x1o+1x2e")
    activation = GateScalarsMLP(irreps, torch.nn.SiLU(), torch.nn.SiLU())
    activation.double()
    # Fixed nontrivial weights make the scalar dependence and per-copy gate
    # order observable rather than accepting any changed output.
    with torch.no_grad():
        for parameter in activation.mlp.parameters():
            parameter.copy_(
                torch.linspace(-0.2, 0.3, parameter.numel()).reshape_as(parameter)
            )
    x = (
        torch.linspace(-0.8, 1.1, 2 * irreps.dim, dtype=torch.float64)
        .reshape(2, irreps.dim)
        .requires_grad_()
    )
    actual = activation(x)

    scalars = x[:, :2]
    vector = x[:, 2:8].reshape(2, 2, 3)
    tensor = x[:, 8:].reshape(2, 1, 5)
    first, second = activation.mlp[0], activation.mlp[2]
    hidden = F.leaky_relu(F.linear(scalars, first.weight, first.bias))
    gates = F.silu(F.linear(hidden, second.weight, second.bias))
    expected = torch.cat(
        (
            F.silu(scalars),
            (vector * gates[:, :2, None]).reshape(2, 6),
            (tensor * gates[:, 2:3, None]).reshape(2, 5),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_gate_magnitudes_uses_scalar_and_norm_gates():
    irreps = Irreps("2x0e+1x1o+1x2e")
    activation = GateMagnitudes(irreps, torch.nn.SiLU(), torch.nn.Softplus())
    x = torch.tensor(
        [[-0.7, 0.4, 1.0, -2.0, 0.5, 0.2, -0.3, 0.8, 1.1, -0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = activation(x)
    scalar = F.silu(x[:, :2])
    vector = x[:, 2:5].reshape(1, 1, 3)
    tensor = x[:, 5:].reshape(1, 1, 5)

    def magnitude_gate(block):
        norm = torch.linalg.vector_norm(block, dim=-1, keepdim=True)
        return block * (F.softplus(norm) / norm)

    expected = torch.cat(
        (scalar, magnitude_gate(vector).flatten(1), magnitude_gate(tensor).flatten(1)),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("normact", NormActivation),
        ("gate_scalars_mlp", GateScalarsMLP),
        ("gate_magnitudes", GateMagnitudes),
        ("s2act", S2Activation),
    ],
)
def test_factory_selects_functional_o3_covariant_activation(
    sample_irreps, sample_tensor, kind, expected_type
):
    # S2Activation has a strict single-copy, one-copy-per-l requirement.
    irreps = Irreps("1x0e+1x1o+1x2e") if kind == "s2act" else sample_irreps
    cfg = Config(nonlin_kind=kind, safety_checks=True, s2act_res=128)
    activation = make_nonlinearity(irreps, cfg)
    assert isinstance(activation, expected_type)
    torch.manual_seed(739)
    x = (
        torch.randn(sample_tensor.shape[0], irreps.dim, dtype=sample_tensor.dtype)
        if kind == "s2act"
        else sample_tensor
    )
    rotation = rand_matrix(dtype=x.dtype)
    # Include inversion explicitly; the chosen l/parity sequence is valid for
    # S2Activation and for the scalar-gating paths.
    for transform in (rotation, -torch.eye(3, dtype=x.dtype)):
        representation = irreps.D_from_matrix(transform)
        transformed_then_activated = activation(x @ representation.T)
        activated_then_transformed = activation(x) @ representation.T
        tolerance = 5e-5 if kind == "s2act" else 2e-5
        torch.testing.assert_close(
            transformed_then_activated,
            activated_then_transformed,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.unit
def test_make_nonlinearity_invalid_kind(sample_irreps):
    cfg = Config(nonlin_kind="invalid_kind", safety_checks=True)
    with pytest.raises(ValueError, match="Unknown nonlin_kind"):
        make_nonlinearity(sample_irreps, cfg)
