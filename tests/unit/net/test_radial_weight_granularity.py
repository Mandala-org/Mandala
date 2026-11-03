"""Radial weights shared per irrep term or per irrep copy."""

import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

from net.common import Config
from net.component_defaults import default_graph
from net.transforms.mandala import EquivariantConvolution


def _conv(mode=None, *, nonlin=False):
    options = {} if mode is None else {"radial_weight_granularity": mode}
    with torch.random.fork_rng():
        torch.set_default_dtype(torch.float64)
        try:
            return EquivariantConvolution(
                3,
                Irreps("4x0e + 2x1e"),
                Irreps("1x0e + 1x1o"),
                Irreps("4x0e + 2x1o"),
                gate=nonlin,
                radial_layers=(5,),
                **options,
            )
        finally:
            torch.set_default_dtype(torch.float32)


def _inputs(conv):
    return (
        conv.tp.tp.irreps_in1.randn(3, -1, dtype=torch.float64),
        conv.tp.tp.irreps_in2.randn(3, -1, dtype=torch.float64),
        torch.randn(3, 3, dtype=torch.float64),
    )


def test_default_gate_map_groups_terms_and_shares_parameters_with_copy_mode():
    torch.manual_seed(14)
    default = _conv()
    explicit = _conv("term")
    explicit.load_state_dict(default.state_dict(), strict=True)
    inputs = _inputs(default)
    assert default._radial_weight_index.tolist() == [0] * 4 + [1] * 6
    assert torch.equal(default(*inputs), explicit(*inputs))
    assert default.state_dict().keys() == explicit.state_dict().keys()

    # Both modes have the same parameters; only the weight-to-copy map differs.
    per_copy = _conv("copy")
    per_copy.load_state_dict(default.state_dict(), strict=True)
    assert per_copy._radial_weight_index.tolist() == [0, 1, 2, 3, 4, 4, 4, 5, 5, 5]


@pytest.mark.parametrize("nonlin", [False, True])
def test_copy_gates_match_explicit_irrep_copy_values_and_gradients(nonlin):
    torch.manual_seed(23)
    conv = _conv("copy", nonlin=nonlin)
    inputs = tuple(value.requires_grad_() for value in _inputs(conv))
    parameters = tuple(conv.parameters())
    actual = conv(*inputs)
    actual_grads = torch.autograd.grad(actual.square().sum(), (*inputs, *parameters))

    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in inputs
    )
    left, right, radial = reference_inputs
    features = conv.tp(left, right)
    if conv.nonlin is not None:
        features = conv.nonlin(features)
    gates = conv.radial_mlp(radial)
    pieces, component_start, gate_start = [], 0, 0
    for multiplicity, irrep in conv.irreps_out:
        component_stop = component_start + multiplicity * irrep.dim
        copies = features[:, component_start:component_stop].reshape(
            features.shape[0], multiplicity, irrep.dim
        )
        copy_gates = gates[:, gate_start : gate_start + multiplicity, None]
        pieces.append((copies * copy_gates).flatten(1))
        component_start = component_stop
        gate_start += multiplicity
    expected = torch.cat(pieces, dim=-1)
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (*reference_inputs, *parameters)
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("mode,used_outputs", [("term", 2), ("copy", 6)])
def test_radial_outputs_used_by_the_operator(mode, used_outputs):
    torch.manual_seed(37)
    conv = _conv(mode)

    class IndependentGates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gates = torch.nn.Parameter(torch.ones(6, dtype=torch.float64))

        def forward(self, radial):
            return self.gates.expand(radial.shape[0], -1)

    conv.radial_mlp = IndependentGates()
    conv(*_inputs(conv)).square().sum().backward()
    gradient = conv.radial_mlp.gates.grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() == used_outputs
    assert (gradient[:used_outputs] > 0).all()


@pytest.mark.parametrize("parity", [1, -1])
def test_copy_gate_ablation_preserves_o3_covariance(parity):
    torch.manual_seed(41)
    conv = _conv("copy", nonlin=True)
    left, right, radial = _inputs(conv)
    rotation = parity * rand_matrix(dtype=torch.float64)
    left_d = conv.tp.tp.irreps_in1.D_from_matrix(rotation)
    right_d = conv.tp.tp.irreps_in2.D_from_matrix(rotation)
    output_d = conv.irreps_out.D_from_matrix(rotation)
    torch.testing.assert_close(
        conv(left @ left_d.T, right @ right_d.T, radial),
        conv(left, right, radial) @ output_d.T,
        atol=1e-10,
        rtol=1e-10,
    )


def test_gate_granularity_survives_configuration_and_common_graph_lowering():
    cfg = Config(radial_weight_granularity="copy", num_layers_gnn=1)
    restored = Config.from_mapping(cfg.to_mapping())
    assert restored.radial_weight_granularity == "copy"
    graph = default_graph(restored, num_species=2)
    convolutions = [
        stage
        for name in ("node_update", "edge_update")
        for stage in graph["blocks"][name]["stages"]
        if stage["transform"] == "equivariant_convolution"
    ]
    assert len(convolutions) == 2
    for stage in convolutions:
        assert stage["options"]["radial_weight_granularity"] == "copy"


def test_invalid_gate_granularity_fails_explicitly():
    with pytest.raises(ValueError, match="radial_weight_granularity"):
        _conv("component")
