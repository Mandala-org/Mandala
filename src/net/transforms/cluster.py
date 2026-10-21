"""Neighbor-density, symmetric-product and angular-degree transformations.

These are reusable stages, rather than a second model implementation.  A typical
ordered stage sequence is ``NeighborDensityInteraction ->
SymmetricProductContraction`` repeated at each correlation layer.  Save every
hidden output and the first interaction's ``expansion`` output; feed those to
successive ``AngularDegreeExpansion`` stages, then to independently configured
edge updates.  This is deliberately not the fixed node/edge alternation.

The tensor algebra and initialization follow MACE-H commit
3d6248b715eef302a4fd674d52bcd69254596a51.  Its many-body implementation derives
from MACE by Ilyes Batatia and Gregor Simm.  Their per-file notices are retained
in ``coupling/``.  MACE-H adaptations are Copyright (c) 2023 Xiaoxun-Gong,
distributed under the MIT license reproduced in ``coupling/MIT.md``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from e3nn import nn as e3nn_nn, o3
from torch import nn

from .coupling.symmetric_contraction import SymmetricContraction
from .equivariant import (
    FactorizedTensorProduct,
    GraphIrrepNorm,
    make_tensor_product_gate,
)


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


class NeighborDensityInteraction(nn.Module):
    """Radially weighted neighbor tensor products, summed before correlation.

    ``skip`` is a species-conditioned tensor product of the *input* node
    features.  ``density`` is channel-first.  An optional independently learned
    ``expansion`` is flat and is saved for a later angular-degree stream.

    With ``normalization='none'`` both post-sum projections are divided by
    ``avg_num_neighbors``.  With ``'graph_component'`` the summed intermediate
    is normalized *before* either projection, and no degree divisor is used.
    This distinction reproduces the two actual upstream transformations.

    Parameters retain upstream names and creation order, including the radial
    MLP's e3nn initialization and external tensor-product path weights.
    """

    def __init__(
        self,
        node_attrs_irreps,
        node_feats_irreps,
        edge_attrs_irreps,
        edge_feats_irreps,
        target_irreps,
        hidden_irreps,
        avg_num_neighbors: float,
        radial_layers: Sequence[int] = (64, 64, 64),
        *,
        emit_expansion: bool = False,
        normalization: str = "none",
    ):
        super().__init__()
        if normalization not in {"none", "graph_component"}:
            raise ValueError("Unknown neighbor-density normalization.")
        if normalization == "none" and (
            not math.isfinite(float(avg_num_neighbors)) or avg_num_neighbors <= 0
        ):
            raise ValueError("avg_num_neighbors must be finite and positive.")
        self.node_attrs_irreps = o3.Irreps(node_attrs_irreps)
        self.node_feats_irreps = o3.Irreps(node_feats_irreps)
        self.edge_attrs_irreps = o3.Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.radial_MLP = list(radial_layers)
        self.emit_expansion = bool(emit_expansion)
        self.normalization = normalization
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        irreps_mid, instructions = _channelwise_product_paths(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = e3nn_nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps]
            + self.radial_MLP
            + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )
        if normalization == "graph_component":
            self.norm = GraphIrrepNorm(irreps_mid)
        if emit_expansion:
            self.linear_up_expansion = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = ChannelIrrepReshape(self.irreps_out)

    def forward(
        self,
        node_attrs,
        node_feats,
        edge_attrs,
        edge_feats,
        edge_index,
        batch=None,
    ):
        sender, receiver = edge_index
        skip = self.skip_tp(node_feats, node_attrs)
        lifted = self.linear_up(node_feats)
        weights = self.conv_tp_weights(edge_feats)
        edge_messages = self.conv_tp(lifted[sender], edge_attrs, weights)
        message = edge_messages.new_zeros((node_feats.shape[0], edge_messages.shape[1]))
        message.scatter_add_(
            0, receiver[:, None].expand_as(edge_messages), edge_messages
        )
        if self.normalization == "graph_component":
            if batch is None:
                raise ValueError("Graph normalization requires node graph indices.")
            message = self.norm(message, batch)
            density = self.linear(message)
        else:
            density = self.linear(message) / self.avg_num_neighbors
        result = {"density": self.reshape(density), "skip": skip}
        if self.emit_expansion:
            expansion = self.linear_up_expansion(message)
            if self.normalization == "none":
                expansion = expansion / self.avg_num_neighbors
            result["expansion"] = expansion
        return result


class SymmetricProductContraction(nn.Module):
    """Learned species-conditioned symmetric products of summed densities.

    Uses the source real Clebsch--Gordan coupling basis and Horner contraction,
    including all orders from one through ``correlation``.  The source tracing
    example RNG draws are intentionally retained: removing them would change
    seeded from-scratch initialization of all later stages.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        correlation: int,
        num_elements: int,
        *,
        use_skip: bool = True,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        if correlation < 1 or correlation > 4:
            raise ValueError("Supported correlation orders are 1 through 4.")
        if len({mul for mul, _ in self.irreps_in}) != 1:
            raise ValueError("Symmetric products require equal input multiplicities.")
        channels = self.irreps_in.count(o3.Irrep("0e"))
        if channels == 0 or any(mul != channels for mul, _ in self.irreps_out):
            raise ValueError(
                "Output multiplicities must equal the scalar channel count."
            )
        self.use_sc = bool(use_skip)
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )
        self.linear = o3.Linear(
            self.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, density, node_attrs, skip=None):
        output = self.linear(self.symmetric_contractions(density, node_attrs))
        if self.use_sc and skip is not None:
            output = output + skip
        return output


class AngularDegreeExpansion(nn.Module):
    """Couple one hidden node stream to a separately carried angular stream.

    Order: independent projections -> factorized tensor product -> optional
    gate -> biased linear -> optional species self-connection -> graph norm
    -> biased projection of the original hidden input.  Its output can become
    the next layer's angular stream without changing the main hidden stream.
    """

    def __init__(
        self,
        hidden_irreps,
        hidden_post_irreps,
        up_irreps,
        up_post_irreps,
        irreps_out,
        num_species: int,
        *,
        use_self_connection: bool = True,
        nonlinearity: bool = True,
        normalization: str = "graph_component",
    ):
        super().__init__()
        hidden_irreps, up_irreps = o3.Irreps(hidden_irreps), o3.Irreps(up_irreps)
        hidden_post_irreps = o3.Irreps(hidden_post_irreps)
        up_post_irreps = o3.Irreps(up_post_irreps)
        irreps_out = o3.Irreps(irreps_out)
        if normalization not in {"none", "graph_component"}:
            raise ValueError("Unknown angular expansion normalization.")
        self.hidden_linear = o3.Linear(
            hidden_irreps,
            hidden_post_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.up_linear = o3.Linear(
            up_irreps, up_post_irreps, internal_weights=True, shared_weights=True
        )
        self.nonlin = None
        if nonlinearity:
            self.nonlin = make_tensor_product_gate(
                hidden_post_irreps, up_post_irreps, irreps_out
            )
            irreps_tp_out = self.nonlin.irreps_in
            self.irreps_out = self.nonlin.irreps_out
        else:
            reachable = {
                ir
                for _, left in hidden_post_irreps
                for _, right in up_post_irreps
                for ir in left * right
            }
            irreps_tp_out = o3.Irreps(
                [(mul, ir) for mul, ir in irreps_out if ir in reachable]
            )
            self.irreps_out = irreps_tp_out
        self.tp = FactorizedTensorProduct(
            hidden_post_irreps, up_post_irreps, irreps_tp_out
        )
        self.lin_post = o3.Linear(self.irreps_out, self.irreps_out, biases=True)
        self.norm = (
            GraphIrrepNorm(self.irreps_out)
            if normalization == "graph_component"
            else None
        )
        self.sc = None
        if use_self_connection:
            self.sc = o3.FullyConnectedTensorProduct(
                hidden_irreps, f"{num_species}x0e", self.irreps_out
            )
        self.skip_connect = o3.Linear(hidden_irreps, self.irreps_out, biases=True)

    def forward(self, hidden, expansion, node_attrs, batch=None):
        output = self.tp(self.hidden_linear(hidden), self.up_linear(expansion))
        if self.nonlin is not None:
            output = self.nonlin(output)
        output = self.lin_post(output)
        if self.sc is not None:
            output = output + self.sc(hidden, node_attrs)
        if self.norm is not None:
            if batch is None:
                raise ValueError("Graph normalization requires node graph indices.")
            output = self.norm(output, batch)
        return output + self.skip_connect(hidden)


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


def build_cluster_transform(name, options, inputs):
    """Build a typed stage for the shared configuration-driven pipeline.

    Positional input order:

    * neighbor_density: attrs, hidden, harmonics, radial, edge_index[, batch]
    * symmetric_product: density, attrs[, skip]
    * angular_degree_expansion: hidden, angular, attrs[, batch]

    Density outputs are ordered ``density, skip[, expansion]``.  Both other
    transformations return one flat global-irrep node feature tensor.
    """
    from net.pipeline import (
        FeatureSpec,
        _global_irreps,
        _same_support,
        _index_spec,
        _same_angular_basis,
    )

    parameters = dict(options)
    if name == "neighbor_density":
        if len(inputs) not in {5, 6}:
            raise ValueError(
                "neighbor_density expects five inputs plus optional batch."
            )
        attrs = _global_irreps(inputs[0], "node")
        hidden = _global_irreps(inputs[1], "node")
        angular = _global_irreps(inputs[2], "edge")
        radial = _global_irreps(inputs[3], "edge")
        basis = _same_angular_basis(inputs[1], inputs[2])
        _same_support(inputs[2], inputs[3], inputs[4])
        _same_support(inputs[0], inputs[1])
        if any(ir.l or ir.p != 1 for _, ir in attrs + radial):
            raise ValueError(
                "Species attributes and radial features must be even scalars."
            )
        if inputs[4].domain != "relation" or inputs[4].representation != "index":
            raise ValueError(
                "neighbor_density requires a directed-edge index relation."
            )
        if len(inputs) == 6 and (
            inputs[5].domain != "node" or inputs[5].representation != "index"
        ):
            raise ValueError("Density batch indices must be node-scoped integers.")
        if len(inputs) == 6:
            _same_support(inputs[1], inputs[5])
        module = NeighborDensityInteraction(
            node_attrs_irreps=attrs,
            node_feats_irreps=hidden,
            edge_attrs_irreps=angular,
            edge_feats_irreps=radial,
            **parameters,
        )
        angular_basis = o3.Irreps([(1, ir) for _, ir in module.target_irreps])
        specs = [
            FeatureSpec(
                "node",
                "channel_irreps",
                str(angular_basis),
                width=module.reshape.channels,
                support=inputs[1].support,
                angular_basis=basis,
            ),
            FeatureSpec(
                "node",
                "irreps",
                str(module.hidden_irreps),
                support=inputs[1].support,
                angular_basis=basis,
            ),
        ]
        if module.emit_expansion:
            specs.append(
                FeatureSpec(
                    "node",
                    "irreps",
                    str(module.target_irreps),
                    support=inputs[1].support,
                    angular_basis=basis,
                )
            )
        return module, specs
    if name == "symmetric_product":
        if len(inputs) not in {2, 3}:
            raise ValueError(
                "symmetric_product expects density, species and optional skip."
            )
        density = inputs[0]
        if (
            density.domain != "node"
            or density.representation != "channel_irreps"
            or density.frame != "global"
            or density.irreps is None
            or density.width is None
        ):
            raise ValueError(
                "symmetric_product requires channel-first global node density."
            )
        attrs = _global_irreps(inputs[1], "node")
        _same_support(*inputs)
        _same_angular_basis(*inputs)
        if any(ir.l or ir.p != 1 for _, ir in attrs):
            raise ValueError("Species attributes must be even scalars.")
        full_irreps = o3.Irreps(
            [(density.width * mul, ir) for mul, ir in o3.Irreps(density.irreps)]
        )
        parameters.setdefault("num_elements", attrs.dim)
        module = SymmetricProductContraction(full_irreps, **parameters)
        if len(inputs) == 3 and _global_irreps(inputs[2], "node") != module.irreps_out:
            raise ValueError("Symmetric-product skip irreps do not match the output.")
        return module, [
            FeatureSpec(
                "node",
                "irreps",
                str(module.irreps_out),
                support=density.support,
                angular_basis=density.angular_basis,
            )
        ]
    if name == "angular_degree_expansion":
        if len(inputs) not in {3, 4}:
            raise ValueError(
                "Angular expansion expects hidden, angular, species[, batch]."
            )
        hidden = _global_irreps(inputs[0], "node")
        angular = _global_irreps(inputs[1], "node")
        attrs = _global_irreps(inputs[2], "node")
        _same_support(*inputs)
        basis = _same_angular_basis(*inputs)
        if len(inputs) == 4:
            _index_spec(inputs[3], "node")
        if any(ir.l or ir.p != 1 for _, ir in attrs):
            raise ValueError("Species attributes must be even scalars.")
        requested = parameters["irreps_out"]
        parameters.setdefault("num_species", attrs.dim)
        parameters.setdefault(
            "hidden_post_irreps", angular_expansion_projection_irreps(hidden, requested)
        )
        parameters.setdefault(
            "up_post_irreps", angular_expansion_projection_irreps(angular, requested)
        )
        module = AngularDegreeExpansion(
            hidden_irreps=hidden, up_irreps=angular, **parameters
        )
        return module, [
            FeatureSpec(
                "node",
                "irreps",
                str(module.irreps_out),
                support=inputs[0].support,
                angular_basis=basis,
            )
        ]
    if name == "cutoff_radial_expansion":
        if len(inputs) != 1 or inputs[0].domain != "edge":
            raise ValueError("Radial expansion requires edge-scoped distances.")
        module = CutoffRadialExpansion(**parameters)
        return module, [
            FeatureSpec(
                "edge",
                irreps=f"{module.out_dim}x0e",
                width=module.out_dim,
                support=inputs[0].support,
            )
        ]
    raise ValueError(f"Unknown cluster transformation {name!r}")
