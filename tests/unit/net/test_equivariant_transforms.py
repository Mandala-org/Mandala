"""Analytic contracts and optional pinned-source differential checks.

Run the differential checks with::

    MANDALA_DEEPHE3_REFERENCE=/path/to/DeepH-E3 pytest -q \
        tests/unit/net/test_equivariant_transforms.py

The upstream code is loaded only as an external oracle, never by production
MANDALA.  Hashes pin the audited source files, including parameter coordinates.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from e3nn.o3 import Irreps

from net.transforms.equivariant import (
    EndpointTensorProductEdgeUpdate,
    EndpointTensorProductNodeUpdate,
    FactorizedTensorProduct,
    GaussianBasis,
    GatedRadialTensorProduct,
    GraphIrrepNorm,
    IrrepChannelScale,
    IrrepPermutation,
    IrrepResidual,
    SymmetricFactorizedTensorProduct,
    make_tensor_product_gate,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def reference():
    value = os.getenv("MANDALA_DEEPHE3_REFERENCE")
    if not value:
        pytest.skip("Set MANDALA_DEEPHE3_REFERENCE to enable pinned-source parity.")
    root = Path(value).resolve()
    package_dir = root / "deephe3" if (root / "deephe3").is_dir() else root
    hashes = {
        "model.py": "e27fbb04ade27371aa798da1a47ed4c4adc2bec629ef46dfa968c1e0734f8297",
        "e3modules.py": "a1dd2d12d4ddceed83a153753eba5db08bc23ecc7182255ad63eb1d1309633c0",
        "from_schnetpack/acsf.py": "4b8c345ced63ab81819d18ebb95c8602f00a27c9dd521a95a19ab8e92e19b7cd",
    }
    for name, expected in hashes.items():
        path = package_dir / name
        assert path.is_file(), f"Missing reference source: {path}"
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == expected
        ), f"Reference {path} differs from audited DeepH-E3 commit 036bf95."
    # Bypass package __init__, which imports unrelated training/data utilities.
    package_name = "_mandala_equivariant_deephe3_oracle"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package
    return types.SimpleNamespace(
        model=importlib.import_module(f"{package_name}.model"),
        e3=importlib.import_module(f"{package_name}.e3modules"),
        gaussian=importlib.import_module(f"{package_name}.from_schnetpack.acsf"),
    )


def _assert_initialization_and_vjp(ours_factory, reference_factory, arguments):
    """Compare random initialization, outputs and every input/parameter VJP."""
    torch.manual_seed(7109)
    expected = reference_factory()
    torch.manual_seed(7109)
    actual = ours_factory()
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, expected.state_dict()[name], rtol=0, atol=0)
    ours_inputs, reference_inputs = [], []
    for value in arguments:
        if isinstance(value, torch.Tensor):
            needs_grad = value.is_floating_point() or value.is_complex()
            ours_inputs.append(value.detach().clone().requires_grad_(needs_grad))
            reference_inputs.append(value.detach().clone().requires_grad_(needs_grad))
        else:
            ours_inputs.append(value)
            reference_inputs.append(value)
    ours_output = actual(*ours_inputs)
    expected_output = expected(*reference_inputs)
    torch.testing.assert_close(ours_output, expected_output, rtol=1e-12, atol=1e-12)
    cotangent = torch.linspace(0.3, 1.4, ours_output.numel()).reshape_as(ours_output)
    ours_variables = [
        x for x in ours_inputs if isinstance(x, torch.Tensor) and x.requires_grad
    ] + list(actual.parameters())
    expected_variables = [
        x for x in reference_inputs if isinstance(x, torch.Tensor) and x.requires_grad
    ] + list(expected.parameters())
    ours_gradients = torch.autograd.grad(
        ours_output, ours_variables, cotangent, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected_output, expected_variables, cotangent, allow_unused=True
    )
    assert len(ours_gradients) == len(expected_gradients)
    for ours, theirs in zip(ours_gradients, expected_gradients):
        if ours is None or theirs is None:
            assert ours is theirs
        else:
            torch.testing.assert_close(ours, theirs, rtol=1e-11, atol=1e-11)


def test_gaussian_formula_and_distance_gradient():
    basis = GaussianBasis(start=0, stop=2, n_gaussians=3)
    distances = torch.tensor([0.25, 1.5], requires_grad=True)
    difference = distances[:, None] - torch.tensor([0.0, 1.0, 2.0])
    expected = torch.exp(-0.5 * difference.square())
    torch.testing.assert_close(basis(distances), expected)
    gradient = torch.autograd.grad(basis(distances).sum(), distances)[0]
    torch.testing.assert_close(gradient, (-difference * expected).sum(dim=-1))
    assert not list(basis.parameters())
    assert len(list(GaussianBasis(0, 2, 3, trainable=True).parameters())) == 2


def test_factorized_scalar_product_retains_two_parameter_coordinates():
    product = FactorizedTensorProduct("1x0e", "1x0e", "1x0e")
    with torch.no_grad():
        product.weights1[0].fill_(2)
        product.weights2[0].fill_(3)
    x = torch.tensor([[5.0]], requires_grad=True)
    y = torch.tensor([[7.0]], requires_grad=True)
    value = product(x, y)
    torch.testing.assert_close(value, torch.tensor([[210.0]]))
    gradients = torch.autograd.grad(value.sum(), [x, y, *product.parameters()])
    expected = [42.0, 30.0, 105.0, 70.0]
    for gradient, scalar in zip(gradients, expected):
        torch.testing.assert_close(gradient, torch.full_like(gradient, scalar))


def test_graph_norm_uses_graph_rows_and_channels_and_external_epsilon():
    norm = GraphIrrepNorm("2x0e+1x1o", eps=0.1)
    batch = torch.tensor([0, 0, 1, 1, 1])
    scalars = torch.tensor([[1.0, 3], [5, 7], [10, 12], [14, 16], [18, 20]])
    vectors = torch.arange(15.0).reshape(5, 3)
    value = norm(torch.cat([scalars, vectors], dim=-1), batch)
    for indices in [slice(0, 2), slice(2, 5)]:
        group = scalars[indices]
        centered = group - group.mean()
        expected = centered / (centered.square().mean().sqrt() + 0.1)
        torch.testing.assert_close(value[indices, :2], expected)
        torch.testing.assert_close(
            value[indices, 2:], vectors[indices] - vectors[indices].mean(dim=0)
        )
    changed = torch.cat([scalars, vectors], dim=-1)
    changed[2:] *= 100
    torch.testing.assert_close(norm(changed, batch)[:2], value[:2])


def test_graph_norm_vector_equivariance_and_scalar_only_bias():
    irreps = Irreps("2x0e+1x0o+2x1o")
    norm = GraphIrrepNorm(irreps)
    assert norm.bias.numel() == 2
    x = torch.randn(7, irreps.dim)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    representation = irreps.D_from_matrix(rotation)
    torch.testing.assert_close(
        norm(x @ representation.T, batch),
        norm(x, batch) @ representation.T,
        rtol=1e-10,
        atol=1e-10,
    )


def test_channel_scale_and_irrep_permutation_contracts():
    value = torch.arange(16.0).reshape(2, 8)
    scale = IrrepChannelScale("1x1o+2x0e+1x1o")
    weights = torch.tensor([[2.0, 3, 4, 5], [6, 7, 8, 9]])
    expanded = weights[:, [0, 0, 0, 1, 2, 3, 3, 3]]
    torch.testing.assert_close(scale(value, weights), value * expanded)
    permutation = IrrepPermutation(scale.irreps_in)
    torch.testing.assert_close(permutation.inverse(permutation(value)), value)


@pytest.mark.parametrize("trainable", [False, True])
def test_gaussian_reference_initialization_forward_and_gradients(reference, trainable):
    kwargs = dict(start=0.1, stop=4, n_gaussians=7, trainable=trainable)
    _assert_initialization_and_vjp(
        lambda: GaussianBasis(**kwargs),
        lambda: reference.gaussian.GaussianBasis(**kwargs),
        [torch.tensor([0.0, 0.3, 2.7, 5.0])],
    )


def test_factorized_tp_reference_initialization_forward_and_gradients(reference):
    args = ("2x0e+1x1o", "1x0e+2x1o", "2x0e+1x1o+1x2e")
    _assert_initialization_and_vjp(
        lambda: FactorizedTensorProduct(*args),
        lambda: reference.e3.SeparateWeightTensorProduct(*args),
        [torch.randn(4, 5), torch.randn(4, 7)],
    )


def test_self_tp_reference_initialization_forward_and_gradients(reference):
    args = ("2x0e+1x1o", "2x0e+1x1o+1x2e")
    _assert_initialization_and_vjp(
        lambda: SymmetricFactorizedTensorProduct(*args),
        lambda: reference.e3.SelfTp(*args),
        [torch.randn(4, 5)],
    )


@pytest.mark.parametrize("subtract_mean,divide_norm", [(True, False), (False, True)])
def test_graph_norm_reference_initialization_forward_and_gradients(
    reference, subtract_mean, divide_norm
):
    args = (Irreps("2x0e+1x0o+2x1o"),)
    kwargs = dict(subtract_mean=subtract_mean, divide_norm=divide_norm)
    _assert_initialization_and_vjp(
        lambda: GraphIrrepNorm(*args, **kwargs),
        lambda: reference.e3.e3LayerNorm(*args, **kwargs),
        [torch.randn(7, 9), torch.tensor([0, 0, 0, 1, 1, 1, 1])],
    )


@pytest.mark.parametrize("project", [False, True])
def test_residual_reference_initialization_forward_and_gradients(reference, project):
    args = ("2x0e+1x1o", "1x0e+1x1o" if project else "2x0e+1x1o")
    _assert_initialization_and_vjp(
        lambda: IrrepResidual(*args),
        lambda: reference.e3.SkipConnection(*args),
        [torch.randn(4, Irreps(args[0]).dim), torch.randn(4, Irreps(args[1]).dim)],
    )


@pytest.mark.parametrize("nonlin", [False, True])
def test_radial_convolution_reference_initialization_forward_and_gradients(
    reference, nonlin
):
    args = (4, "2x0e+1x1o", "1x0e+1x1o", "2x0e+1x1o+1x2e")
    kwargs = dict(norm="e3LayerNorm", nonlin=nonlin)
    _assert_initialization_and_vjp(
        lambda: GatedRadialTensorProduct(*args, **kwargs),
        lambda: reference.model.EquiConv(*args, **kwargs),
        [
            torch.randn(6, 5),
            torch.randn(6, 4),
            torch.randn(6, 4),
            torch.tensor([0, 0, 0, 1, 1, 1]),
        ],
    )


def _endpoint_arguments(edge_update):
    edge_index = torch.tensor([[0, 1, 0, 2, 3, 2], [0, 1, 1, 2, 3, 3]])
    species = torch.tensor([0, 1, 1, 0])
    one_hot = torch.nn.functional.one_hot(species, 2).to(torch.get_default_dtype())
    pair_ids = 2 * species[edge_index[0]] + species[edge_index[1]]
    pair_one_hot = torch.nn.functional.one_hot(pair_ids, 4).to(
        torch.get_default_dtype()
    )
    common = [
        torch.randn(6, 4),
        torch.randn(6, 2),
        torch.randn(6, 4),
        edge_index,
        torch.tensor([0, 0, 1, 1]),
    ]
    if edge_update:
        return [torch.randn(4, 5), pair_one_hot, *common]
    return [torch.randn(4, 5), one_hot, *common, None, None]


@pytest.mark.parametrize("nonlin,sort", [(False, False), (True, True)])
def test_endpoint_node_reference_initialization_forward_and_gradients(
    reference, nonlin, sort
):
    kwargs = dict(
        num_species=2,
        fc_len_in=4,
        irreps_sh="1x0e+1x1o",
        irreps_in_node="2x0e+1x1o",
        irreps_out_node="3x0e+2x1o+1x2e",
        irreps_in_edge="2x0e",
        act={1: torch.nn.functional.silu, -1: torch.tanh},
        act_gates={1: torch.sigmoid, -1: torch.tanh},
        nonlin=nonlin,
        if_sort_irreps=sort,
    )
    _assert_initialization_and_vjp(
        lambda: EndpointTensorProductNodeUpdate(**kwargs),
        lambda: reference.model.NodeUpdateBlock(**kwargs),
        _endpoint_arguments(False),
    )


@pytest.mark.parametrize("nonlin,sort", [(False, False), (True, True)])
def test_endpoint_edge_reference_initialization_forward_and_gradients(
    reference, nonlin, sort
):
    kwargs = dict(
        num_species=2,
        fc_len_in=4,
        irreps_sh="1x0e+1x1o",
        irreps_in_node="2x0e+1x1o",
        irreps_in_edge="2x0e",
        irreps_out_edge="3x0e+2x1o+1x2e",
        act={1: torch.nn.functional.silu, -1: torch.tanh},
        act_gates={1: torch.sigmoid, -1: torch.tanh},
        nonlin=nonlin,
        if_sort_irreps=sort,
    )
    _assert_initialization_and_vjp(
        lambda: EndpointTensorProductEdgeUpdate(**kwargs),
        lambda: reference.model.EdgeUpdateBlock(**kwargs),
        _endpoint_arguments(True),
    )


def test_gate_reference_irreps_and_reachable_path_selection(reference):
    args = tuple(map(Irreps, ("1x0e+1x1o", "1x0e+1x1o", "2x0e+1x1o+1x2e+1x3o")))
    actual = make_tensor_product_gate(*args)
    expected = reference.model.get_gate_nonlin(*args)
    assert actual.irreps_in == expected.irreps_in
    assert actual.irreps_out == expected.irreps_out
    inputs = torch.randn(5, actual.irreps_in.dim)
    torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)
    _assert_initialization_and_vjp(
        lambda: make_tensor_product_gate(*args),
        lambda: reference.model.get_gate_nonlin(*args),
        [inputs],
    )


@pytest.mark.parametrize("edge_update", [False, True])
def test_endpoint_update_equivariance_without_external_source(edge_update):
    kwargs = dict(
        num_species=2,
        fc_len_in=4,
        irreps_sh="1x0e+1x1o",
        irreps_in_node="2x0e+1x1o",
        irreps_in_edge="2x0e",
    )
    output_irreps = Irreps("3x0e+2x1o+1x2e")
    constructor = (
        EndpointTensorProductEdgeUpdate
        if edge_update
        else EndpointTensorProductNodeUpdate
    )
    kwargs["irreps_out_edge" if edge_update else "irreps_out_node"] = output_irreps
    update = constructor(**kwargs)
    inputs = _endpoint_arguments(edge_update)
    rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = list(inputs)
    rotated[0] = inputs[0] @ Irreps(kwargs["irreps_in_node"]).D_from_matrix(rotation).T
    rotated[2] = inputs[2] @ Irreps(kwargs["irreps_sh"]).D_from_matrix(rotation).T
    actual = update(*rotated)
    expected = update(*inputs) @ output_irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    gradients = torch.autograd.grad(update(*inputs).square().sum(), update.parameters())
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
