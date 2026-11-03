"""Analytic contracts and optional pinned-source differential checks.

The DeepH-E3 endpoint updates are blocks of these primitives (library
``deeph_e3``); they are checked against the former monolithic layers in
``tests/physics/test_reference_decomposition.py`` and against the pinned
source through ``tests/fixtures/reference_operators``.

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
    FactorizedTensorProduct,
    GaussianBasis,
    GraphIrrepNorm,
    IrrepPermutation,
    RadialChannelScale,
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
    scale = RadialChannelScale("1x1o+2x0e+1x1o", radial_dim=1, hidden=())
    with torch.no_grad():
        scale.fc[0].weight.zero_()
        scale.fc[0].bias.copy_(torch.tensor([2.0, 3, 4, 5]))
    expanded = torch.tensor([2.0, 3, 4, 5])[[0, 0, 0, 1, 2, 3, 3, 3]]
    torch.testing.assert_close(scale(value, torch.ones(2, 1)), value * expanded)
    permutation = IrrepPermutation(scale.irreps)
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
    from net.block_library import library_blocks
    from net.pipeline import FeaturePipeline

    output_irreps = Irreps("3x0e+2x1o+1x2e")
    batch_slot = "edge_batch" if edge_update else "node_batch"
    model = FeaturePipeline(
        {
            "blocks": library_blocks(["deeph_e3"]),
            "inputs": {
                "nodes": {"domain": "node", "irreps": "2x0e+1x1o"},
                "attributes": {
                    "domain": "edge" if edge_update else "node",
                    "irreps": "4x0e" if edge_update else "2x0e",
                },
                "harmonics": {"domain": "edge", "irreps": "1x0e+1x1o"},
                "edges": {"domain": "edge", "irreps": "2x0e"},
                "radial": {"domain": "edge", "irreps": "4x0e"},
            },
            "stages": [
                {
                    "id": "update",
                    "transform": "e3_edge_update" if edge_update else "e3_node_update",
                    "inputs": [
                        "nodes",
                        "attributes",
                        "harmonics",
                        "edges",
                        "radial",
                        "edge_index",
                        batch_slot,
                    ],
                    "outputs": ["out"],
                    "options": {"irreps_out": str(output_irreps)},
                }
            ],
            "outputs": {"out": "out"},
        },
        num_species=2,
    )
    names = ["nodes", "attributes", "harmonics", "edges", "radial", "edge_index"]
    arguments = _endpoint_arguments(edge_update)
    inputs = dict(zip(names, arguments))
    batch = arguments[6]
    inputs[batch_slot] = batch[inputs["edge_index"][0]] if edge_update else batch
    rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = dict(inputs)
    rotated["nodes"] = inputs["nodes"] @ Irreps("2x0e+1x1o").D_from_matrix(rotation).T
    rotated["harmonics"] = (
        inputs["harmonics"] @ Irreps("1x0e+1x1o").D_from_matrix(rotation).T
    )
    actual = model(rotated)["out"]
    expected = model(inputs)["out"] @ output_irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    gradients = torch.autograd.grad(
        model(inputs)["out"].square().sum(), list(model.parameters())
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
