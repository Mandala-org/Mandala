"""Every MANDALA stage is O(3)-equivariant, alone and when routed.

For each stage, rotating (or reflecting) every irrep input by ``D(g)`` must
rotate every irrep output by ``D(g)``; invariant and index inputs are left
unchanged. ``g`` ranges over a proper rotation, an improper rotation and the
inversion.
"""

import pytest
import torch
from e3nn import o3

from net.features import FeatureSpec
from net.inspection import orbital_mapper
from net.registry import build_transform

pytestmark = pytest.mark.equivariance

NODES, EDGES = 5, 9
EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 4], [0, 1, 3, 2, 0, 4, 2, 1, 4]])
ROLES = torch.tensor([0, 0, 2, 2, 2, 2, 2, 2, 0])


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)
    yield
    torch.set_default_dtype(previous)


def node(irreps):
    return FeatureSpec("node", "irreps", irreps)


def edge(irreps):
    return FeatureSpec("edge", "irreps", irreps)


INDEX = FeatureSpec("relation", "index", frame="none")
ROLE = FeatureSpec("edge", "index", frame="none", width=3)
HARMONICS = "1x0e+1x1o+1x2e"
HIDDEN = "4x0e+2x0o+3x1o+2x1e+2x2e"


def _value(spec):
    if spec is INDEX:
        return EDGE_INDEX
    if spec is ROLE:
        return ROLES
    count = NODES if spec.domain == "node" else EDGES
    return o3.Irreps(spec.irreps).randn(count, -1)


def _transform(spec, value, rotation):
    if spec.representation == "index" or value.dtype == torch.long:
        return value
    return value @ o3.Irreps(spec.irreps).D_from_matrix(rotation).T


def _perturb(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    return module


CASES = {
    "e3mlp": (
        {"variant": "film", "layers": 2, "irreps_out": "3x0e+2x1o+1x2e"},
        [node(HIDDEN)],
    ),
    "e3mlp_routed": (
        {"variant": "gate", "layers": 2, "inactive_routes": [1]},
        [edge(HIDDEN), ROLE],
    ),
    "tensor_square": ({"irreps_out": "2x0e+1x1o+1x2e"}, [node("2x0e+1x1o+1x2e")]),
    "fc_tensor_product": (
        {"irreps_out": "2x0e+2x1o"},
        [node("2x0e+1x1o"), node("1x0e+1x1e")],
    ),
    "endpoint_concat": (
        {"combine": "tensor_product"},
        [node("2x0e+1x1o"), edge("1x0e+1x2e"), INDEX],
    ),
    "equivariant_convolution": (
        {"irreps_out": "3x0e+2x1o+1x2e", "radial_layers": [6]},
        [edge(HIDDEN), edge(HARMONICS), edge("5x0e")],
    ),
    "self_connection": ({}, [node(HIDDEN), node("2x0e"), node("3x0e+1x1o")]),
    "residual": ({"mode": "project"}, [node(HIDDEN), node("3x0e+1x1o")]),
    "irrep_nonlinearity": ({"nonlinearity": "gate_magnitudes"}, [node(HIDDEN)]),
    "layer_norm": ({}, [node(HIDDEN)]),
    "projected_attention": (
        {"scalar_dim": 4, "heads": 2},
        [edge(HIDDEN), node(HIDDEN), edge("3x0e"), INDEX],
    ),
    "onsite_from_nodes": ({}, [edge(HIDDEN), node(HIDDEN), INDEX, ROLE]),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("parity", [1, -1])
def test_stage_is_equivariant(case, parity):
    options, inputs = CASES[case]
    name = case.removesuffix("_routed")
    module, outputs = build_transform(name, options, inputs)
    module = _perturb(module)
    values = [_value(spec) for spec in inputs]
    rotation = parity * o3.rand_matrix()
    rotated = [_transform(spec, value, rotation) for spec, value in zip(inputs, values)]
    expected = _transform(outputs[0], module(*values), rotation)
    torch.testing.assert_close(module(*rotated), expected, rtol=1e-9, atol=1e-10)


def test_pair_projection_rotates_every_pair_with_its_orbital_irreps():
    mapper = orbital_mapper({"H": [0, 1], "O": [0, 1, 2]})
    types = torch.tensor([0, 1, 2, 3, 0, 3, 1, 2, 3])
    partitions = {
        key: {"global_idx": torch.nonzero(types == t).flatten()}
        for t, key in enumerate(mapper.edge_types)
    }
    for options in (
        {"mode": "split", "log_scale": True},
        {"mode": "shared_conditioned", "layers": 2},
    ):
        module, _ = build_transform(
            "pair_projection",
            options,
            [edge(HIDDEN), FeatureSpec("relation", "object"), ROLE],
            mapper=mapper,
        )
        module = _perturb(module)
        hidden = o3.Irreps(HIDDEN).randn(EDGES, -1)
        rotation = -o3.rand_matrix()
        reference = module(hidden, partitions, ROLES)
        rotated = module(
            hidden @ o3.Irreps(HIDDEN).D_from_matrix(rotation).T, partitions, ROLES
        )
        for key, vectors in reference.items():
            d = mapper.get_pair_irreps(key).D_from_matrix(rotation)
            torch.testing.assert_close(
                rotated[key], vectors @ d.T, rtol=1e-9, atol=1e-10
            )
