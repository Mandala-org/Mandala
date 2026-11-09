"""Neighbor-density convolution and symmetric-contraction stages (MACE).

MACE-H's interaction, product and angular-degree expansion layers are
compositions of these stages with the generic ones (see
``configs/reference/mace_h.yaml``): a channel-wise convolution summed onto
nodes, a symmetric contraction of the resulting density, and factorized tensor
products that carry a separate angular stream to the edge updates.

The tensor algebra and initialization follow MACE-H commit
3d6248b715eef302a4fd674d52bcd69254596a51.  Its many-body implementation derives
from MACE by Ilyes Batatia and Gregor Simm.  Their per-file notices are retained
in ``coupling/``.  MACE-H adaptations are Copyright (c) 2023 Xiaoxun-Gong,
distributed under the MIT license reproduced in ``coupling/MIT.md``.
"""

from __future__ import annotations

import math

import torch
from e3nn import nn as e3nn_nn, o3
from torch import nn

from .coupling.symmetric_contraction import SymmetricContraction

from net.features import (
    FeatureSpec,
    global_irreps,
    same_angular_basis,
    same_support,
)
from net.registry import Option, Port, register_transform


def _channelwise_product_paths(irreps1, irreps2, target_irreps):
    """Source-ordered external-weight ``uvu`` tensor-product instructions."""
    outputs, instructions = [], []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(outputs)
                    outputs.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    irreps_out, permutation, _ = o3.Irreps(outputs).sort()
    instructions = [
        (i, j, permutation[k], mode, train) for i, j, k, mode, train in instructions
    ]
    instructions.sort(key=lambda instruction: instruction[2])
    return irreps_out, instructions


class ChannelIrrepReshape(nn.Module):
    """Reshape flat ``[N, sum(mul * dim)]`` into ``[N, channels, sum(dim)]``.

    All irrep blocks must have equal multiplicity.  No irrep sorting or basis
    rotation takes place.  This layout is the input to symmetric contractions.
    """

    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        multiplicities = {mul for mul, _ in self.irreps}
        if len(multiplicities) != 1:
            raise ValueError("Density layout requires equal irrep multiplicities.")
        self.channels = next(iter(multiplicities))

    def forward(self, tensor):
        if tensor.shape[-1] != self.irreps.dim:
            raise ValueError("Feature width does not match density irreps.")
        fields, offset = [], 0
        for mul, ir in self.irreps:
            width = mul * ir.dim
            fields.append(tensor[:, offset : offset + width].reshape(-1, mul, ir.dim))
            offset += width
        return torch.cat(fields, dim=-1)


class ChannelwiseTensorProduct(nn.Module):
    """Channel-wise (``uvu``) tensor product with per-row path weights.

    The weights of every path come from an e3nn fully connected network of
    the radial features, as in MACE's interaction. Output irreps are the
    requested ones reachable from the inputs, sorted and simplified.
    """

    def __init__(
        self,
        irreps_in,
        irreps_harmonics,
        radial_dim,
        target_irreps,
        radial_layers=(64, 64, 64),
    ):
        super().__init__()
        irreps_in, irreps_harmonics = o3.Irreps(irreps_in), o3.Irreps(irreps_harmonics)
        irreps_mid, instructions = _channelwise_product_paths(
            irreps_in, irreps_harmonics, o3.Irreps(target_irreps)
        )
        self.conv_tp = o3.TensorProduct(
            irreps_in,
            irreps_harmonics,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = e3nn_nn.FullyConnectedNet(
            [radial_dim, *radial_layers, self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.irreps_out = irreps_mid.simplify()

    def forward(self, features, harmonics, radial):
        return self.conv_tp(features, harmonics, self.conv_tp_weights(radial))


def angular_expansion_projection_irreps(pre_irreps, requested_irreps):
    """Source multiplicity policy for inputs of an angular-degree expansion.

    Retains input irrep types needed to reach the requested output.  This is a
    configuration-time helper, not a hidden shape-changing forward operation.
    """
    pre_irreps, requested_irreps = o3.Irreps(pre_irreps), o3.Irreps(requested_irreps)
    output = (
        o3.Irreps([(requested_irreps.count(ir), ir) for _, ir in pre_irreps])
        .sort()[0]
        .simplify()
    )
    if pre_irreps.lmax > requested_irreps.lmax:
        multiplicity = requested_irreps.count(o3.Irrep(f"{requested_irreps.lmax}y"))
        for ell in range(requested_irreps.lmax + 1, pre_irreps.lmax + 1):
            output += o3.Irreps([(multiplicity, o3.Irrep(f"{ell}y"))])
    if len(pre_irreps.sort()[0].simplify()) != len(output):
        raise ValueError("Angular expansion projection cannot drop input irrep types.")
    return output


class SineRadialBasis(nn.Module):
    """Trainable sine-over-radius basis with the source frequency convention.

    Message graphs must exclude zero-length onsite edges before evaluating
    this expression, as in the reference neighbor-density stream.
    """

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = True):
        super().__init__()
        weights = (
            math.pi
            / r_max
            * torch.linspace(1.0, num_basis, num_basis, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.bessel_weights = nn.Parameter(weights)
        else:
            self.register_buffer("bessel_weights", weights)
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(math.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, lengths):
        return self.prefactor * (torch.sin(self.bessel_weights * lengths) / lengths)


class PolynomialEnvelope(nn.Module):
    """Polynomial radial cutoff, applied after sine-basis expansion."""

    def __init__(self, r_max: float, p: int = 5):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, lengths):
        scaled = lengths / self.r_max
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * scaled.pow(self.p)
            + self.p * (self.p + 2.0) * scaled.pow(self.p + 1.0)
            - (self.p * (self.p + 1.0) / 2.0) * scaled.pow(self.p + 2.0)
        )
        return envelope * (lengths < self.r_max)


class CutoffRadialExpansion(nn.Module):
    """Sine radial basis multiplied by a polynomial envelope."""

    def __init__(self, r_max: float, num_basis: int = 8, cutoff_order: int = 5):
        super().__init__()
        self.bessel_fn = SineRadialBasis(r_max, num_basis)
        self.cutoff_fn = PolynomialEnvelope(r_max, cutoff_order)
        self.out_dim = num_basis

    def forward(self, lengths):
        if lengths.ndim == 1:
            lengths = lengths[:, None]
        return self.bessel_fn(lengths) * self.cutoff_fn(lengths)


def _even_scalars(irreps, what):
    if any(ir.l or ir.p != 1 for _, ir in irreps):
        raise ValueError(f"{what} must be even scalars.")


@register_transform(
    "channelwise_tensor_product",
    category="convolution",
    summary="MACE convolution: channel-wise product with harmonics, path weights from a radial MLP",
    inputs=[
        Port("features", None, "irreps"),
        Port("harmonics", None, "irreps"),
        Port("radial", None, "invariant"),
    ],
    options=[
        Option("irreps_out", "irreps", key=True, doc="Requested (target) irreps"),
        Option("radial_layers", "int_list", (64, 64, 64), key=True),
    ],
)
def _channelwise_tensor_product(p, inputs, mapper):
    features = global_irreps(inputs[0])
    harmonics = global_irreps(inputs[1])
    radial = global_irreps(inputs[2])
    if len({spec.domain for spec in inputs}) != 1:
        raise ValueError("Convolution inputs must share their entities")
    same_support(*inputs)
    basis = same_angular_basis(inputs[0], inputs[1])
    _even_scalars(radial, "Radial features")
    module = ChannelwiseTensorProduct(
        features, harmonics, radial.num_irreps, p["irreps_out"], p["radial_layers"]
    )
    return module, [
        FeatureSpec(
            inputs[0].domain,
            "irreps",
            str(module.irreps_out),
            support=inputs[0].support,
            angular_basis=basis,
        )
    ]


@register_transform(
    "channel_reshape",
    category="combine",
    summary="Flat irreps with equal multiplicities -> [channels, sum of irrep dims]",
    inputs=[Port("features", "node", "irreps")],
)
def _channel_reshape(p, inputs, mapper):
    irreps = global_irreps(inputs[0])
    module = ChannelIrrepReshape(irreps)
    return module, [
        FeatureSpec(
            inputs[0].domain,
            "channel_irreps",
            str(o3.Irreps([(1, ir) for _, ir in irreps])),
            width=module.channels,
            support=inputs[0].support,
            angular_basis=inputs[0].angular_basis,
        )
    ]


@register_transform(
    "symmetric_contraction",
    category="product",
    summary="MACE species-conditioned symmetric products of a channel density (body order)",
    inputs=[
        Port("density", "node", "channel"),
        Port("attributes", "node", "invariant", doc="Species one-hot"),
    ],
    options=[
        Option("irreps_out", "irreps", key=True),
        Option("correlation", "int", 3, key=True, doc="Body order minus one (1-4)"),
        Option(
            "num_elements",
            "int",
            infer=lambda inputs, mapper, options: global_irreps(inputs[1], "node").dim,
            infer_doc="width of the species attributes",
        ),
    ],
)
def _symmetric_contraction(p, inputs, mapper):
    density = inputs[0]
    if (
        density.domain != "node"
        or density.representation != "channel_irreps"
        or density.frame != "global"
        or density.irreps is None
        or density.width is None
    ):
        raise ValueError(
            "symmetric_contraction requires channel-first global node density."
        )
    attrs = global_irreps(inputs[1], "node")
    same_support(*inputs)
    _even_scalars(attrs, "Species attributes")
    irreps_in = o3.Irreps(
        [(density.width * mul, ir) for mul, ir in o3.Irreps(density.irreps)]
    )
    irreps_out = o3.Irreps(p["irreps_out"])
    if not 1 <= p["correlation"] <= 4:
        raise ValueError("Supported correlation orders are 1 through 4.")
    channels = irreps_in.count(o3.Irrep("0e"))
    if channels == 0 or any(mul != channels for mul, _ in irreps_out):
        raise ValueError("Output multiplicities must equal the scalar channel count.")
    module = SymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        correlation=p["correlation"],
        num_elements=p["num_elements"],
    )
    return module, [
        FeatureSpec(
            "node",
            "irreps",
            str(irreps_out),
            support=density.support,
            angular_basis=density.angular_basis,
        )
    ]


@register_transform(
    "cutoff_radial_expansion",
    category="radial_basis",
    summary="Sine (Bessel) radial basis times a polynomial cutoff envelope",
    inputs=[Port("lengths", "edge")],
    options=[
        Option(
            "r_max",
            "float",
            key=True,
            doc="Cutoff radius (Angstrom); the basis is zero beyond it",
        ),
        Option("num_basis", "int", 8, key=True, doc="Number of sine basis functions"),
        Option("cutoff_order", "int", 5),
    ],
)
def _cutoff_radial_expansion(p, inputs, mapper):
    if inputs[0].domain != "edge":
        raise ValueError("Radial expansion requires edge-scoped distances.")
    module = CutoffRadialExpansion(**p)
    return module, [
        FeatureSpec(
            "edge",
            irreps=f"{module.out_dim}x0e",
            width=module.out_dim,
            support=inputs[0].support,
        )
    ]
