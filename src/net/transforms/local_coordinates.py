"""Invariant-channel operations and explicit bond-frame operations.

DeepH's crystal-graph and local-coordinate message layers are compositions of
these stages with the generic gather, gated-message and envelope stages (see
``configs/reference/deeph.yaml``).
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import LayerNorm, GraphNorm, InstanceNorm, PairNorm
from e3nn import o3
from net.features import (
    FeatureSpec,
    global_irreps,
    index_spec,
    same_support,
)
from net.registry import Option, Port, register_transform


def _normalization(width, kind):
    constructors = {
        "layer": LayerNorm,
        "batch": nn.BatchNorm1d,
        "graph": GraphNorm,
        "instance": InstanceNorm,
    }
    if kind in (None, "none"):
        return None
    if kind == "pair":
        return PairNorm()
    if kind not in constructors:
        raise ValueError(f"Unknown scalar normalization: {kind}")
    return constructors[kind](width)


def _apply_norm(norm, x, batch):
    return (
        x
        if norm is None
        else norm(x) if isinstance(norm, nn.BatchNorm1d) else norm(x, batch)
    )


class ScalarNormalization(nn.Module):
    """Normalization of invariant channels; ``segments`` equal channel groups
    of a row are normalized as separate rows sharing statistics and affine
    parameters (e.g. the two endpoint halves of a bond feature)."""

    def __init__(self, width, kind="layer", segments=1):
        super().__init__()
        if width % segments:
            raise ValueError("Feature width must be divisible by segments")
        self.segments = int(segments)
        self.norm = _normalization(width // self.segments, kind)

    def forward(self, values, batch=None):
        if self.norm is None:
            return values
        if self.segments > 1:
            rows = values.reshape(-1, values.shape[-1] // self.segments)
            if batch is not None:
                batch = batch.repeat_interleave(self.segments)
            return _apply_norm(self.norm, rows, batch).reshape(values.shape)
        return _apply_norm(self.norm, values, batch)


class BondEndpointSum(nn.Module):
    """Sum bond-neighborhood messages into the two endpoints of each bond.

    ``endpoint_index`` holds ``2 * bond + endpoint`` per neighborhood row; the
    result has one row per bond with the two endpoint sums side by side.
    """

    def forward(self, messages, endpoint_index, bonds):
        sums = messages.new_zeros((2 * len(bonds), messages.shape[-1]))
        sums = sums.index_add(0, endpoint_index, messages)
        return sums.reshape(len(bonds), 2 * messages.shape[-1])


class ChannelLinear(nn.Module):
    """Independent affine readouts per output channel, with source parameter axes."""

    def __init__(self, num_channels, in_features, out_features, *, activation=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_channels, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_channels, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        self.activation = activation

    def forward(self, x):
        if self.activation:
            x = F.silu(x)
        return torch.matmul(x, self.weight) + self.bias[:, None, :]


class FlatChannelLinear(ChannelLinear):
    """Independent affine channel outputs flattened per input entity.

    The primitive stores weights as (channels,input,output); its raw channel
    axis precedes entities. Pipeline features always retain entity as axis zero.
    """

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                "channel_linear requires a two-dimensional entity feature tensor"
            )
        values = super().forward(x)
        return values.transpose(0, 1).reshape(
            x.shape[0], self.weight.shape[0] * self.weight.shape[2]
        )


class PairDistanceAffine(nn.Module):
    def __init__(self, num_pairs):
        super().__init__()
        self.embedding = nn.Embedding(
            num_pairs,
            2,
            _weight=torch.stack([torch.ones(num_pairs), torch.zeros(num_pairs)], -1),
        )

    def forward(self, distance, pair_type):
        coefficients = self.embedding(pair_type)
        return distance * coefficients[:, 0] + coefficients[:, 1]


class IrrepToBondFrame(nn.Module):
    """Rotate gathered node irreps into each output bond frame explicitly.

    ``axis_permutation`` states which Cartesian axes define the incoming
    irrep convention. Source coupled features use (1,2,0), whereas standard
    xyz harmonics use (0,1,2). Output local components retain that convention.
    """

    def __init__(self, irreps, *, endpoint=0, axis_permutation=(0, 1, 2)):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be 0 or 1")
        if sorted(axis_permutation) != [0, 1, 2]:
            raise ValueError(
                "axis_permutation must contain each Cartesian axis exactly once"
            )
        self.endpoint = endpoint
        self.axis_permutation = tuple(axis_permutation)

    def forward(self, node, edge_index, frames):
        indices = torch.tensor(self.axis_permutation, device=frames.device)
        frame_rotation = (
            frames.transpose(-1, -2).index_select(-2, indices).index_select(-1, indices)
        )
        rotation = self.irreps.D_from_matrix(frame_rotation)
        return torch.einsum("eij,ej->ei", rotation, node[edge_index[self.endpoint]])


def _invariant_width(spec, what):
    """Width of invariant channels: even scalars or bond-frame components."""
    if spec.representation == "local_components" and spec.frame == "bond":
        if spec.dim is None:
            raise ValueError(f"{what} need a declared width")
        return spec.dim
    irreps = global_irreps(spec)
    if any(ir.l or ir.p != 1 for _, ir in irreps):
        raise ValueError(f"{what} require invariant scalar channels")
    return irreps.dim


@register_transform(
    "scalar_normalization",
    category="normalization",
    summary="Layer / batch / graph / instance / pair normalization of invariant channels",
    inputs=[
        Port("features", None, "invariant"),
        Port("batch", None, "index", optional=True, doc="Graph of each row"),
    ],
    options=[
        Option(
            "kind",
            "choice",
            "layer",
            key=True,
            choices=("layer", "batch", "graph", "instance", "pair", "none"),
            doc="layer/graph/instance/pair statistics are taken per graph",
        ),
        Option(
            "segments",
            "int",
            1,
            doc="Normalize this many equal channel groups of a row as separate rows",
        ),
    ],
)
def _scalar_normalization(p, inputs, mapper):
    width = _invariant_width(inputs[0], "Scalar normalization")
    if len(inputs) > 1:
        index_spec(inputs[1], inputs[0].domain)
        same_support(*inputs)
    elif p["kind"] not in {"batch", "none"}:
        raise ValueError(f"{p['kind']} normalization needs the graph index input")
    return ScalarNormalization(width, p["kind"], p["segments"]), [inputs[0]]


@register_transform(
    "bond_endpoint_sum",
    category="aggregation",
    summary="Sum bond-neighborhood messages into [endpoint i, endpoint j] per bond",
    inputs=[
        Port("messages", "bond_neighbor"),
        Port("endpoint_index", "bond_neighbor", "index", doc="2 * bond + endpoint"),
        Port("bonds", "edge", doc="Reference for the bond count"),
    ],
)
def _bond_endpoint_sum(p, inputs, mapper):
    messages = inputs[0]
    if messages.domain != "bond_neighbor":
        raise ValueError("bond_endpoint_sum sums bond-neighborhood messages")
    index_spec(inputs[1], "bond_neighbor")
    same_support(*inputs)
    width = _invariant_width(messages, "Bond endpoint sums")
    if messages.representation == "local_components":
        spec = FeatureSpec(
            "edge",
            "local_components",
            frame="bond",
            width=2 * width,
            support=inputs[2].support,
        )
    else:
        spec = FeatureSpec(
            "edge",
            irreps=f"{2 * width}x0e",
            width=2 * width,
            support=inputs[2].support,
        )
    return BondEndpointSum(), [spec]


@register_transform(
    "channel_linear",
    category="linear",
    summary="Independent dense map per channel (DeepH multi-channel output)",
    inputs=[Port("features", "edge")],
    options=[
        Option("num_channels", "int", key=True),
        Option(
            "in_features",
            "int",
            infer=lambda inputs, mapper, options: inputs[0].dim,
            infer_doc="input width",
        ),
        Option("out_features", "int", key=True),
        Option("activation", "bool", False),
    ],
)
def _channel_linear(p, inputs, mapper):
    source = inputs[0]
    if source.domain != "edge":
        raise ValueError("channel_linear requires edge-scoped features")
    if source.representation == "local_components" and source.frame == "bond":
        representation, frame = "local_components", "bond"
    else:
        irreps = global_irreps(source, "edge")
        if any(ir.l or ir.p != 1 for _, ir in irreps):
            raise ValueError(
                "channel_linear requires invariant scalars or explicitly rotated bond components"
            )
        representation, frame = "scalar", "global"
    if source.dim != p["in_features"]:
        raise ValueError(
            "channel_linear input width disagrees with the declared feature layout"
        )
    width = p["num_channels"] * p["out_features"]
    return FlatChannelLinear(**p), [
        FeatureSpec(
            "edge",
            representation,
            irreps=f"{width}x0e" if representation == "scalar" else None,
            frame=frame,
            width=width,
            support=source.support,
            angular_basis=source.angular_basis,
        )
    ]


@register_transform(
    "pair_distance_affine",
    category="radial_basis",
    summary="Learned per species-pair affine rescaling of distances",
    inputs=[Port("lengths", "edge"), Port("pair_type", "edge", "index")],
    options=[Option("num_pairs", "int", key=True)],
)
def _pair_distance_affine(p, inputs, mapper):
    index_spec(inputs[1], "edge")
    same_support(*inputs)
    return PairDistanceAffine(**p), [inputs[0]]


@register_transform(
    "irrep_to_bond",
    category="frame",
    summary="Rotate endpoint irreps into the local bond frame",
    inputs=[
        Port("nodes", "node", "irreps"),
        Port("edge_index", "relation", "index"),
        Port("bond_frames", "edge", "geometry"),
    ],
    options=[Option("endpoint", "choice", 0, choices=(0, 1))],
)
def _irrep_to_bond(p, inputs, mapper):
    index_spec(inputs[1], "relation")
    if inputs[2].domain != "edge" or inputs[2].frame != "bond":
        raise ValueError("Bond-frame rotation requires edge-scoped bond frames")
    support = same_support(inputs[1], inputs[2])
    basis = inputs[0].angular_basis or "xyz"
    permutation = tuple("xyz".index(axis) for axis in basis)
    irreps = global_irreps(inputs[0], "node")
    module = IrrepToBondFrame(irreps, axis_permutation=permutation, **p)
    return module, [
        FeatureSpec(
            "edge",
            "local_components",
            str(irreps),
            "bond",
            support=support,
            angular_basis=inputs[0].angular_basis,
        )
    ]
