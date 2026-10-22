"""Invariant messages and explicit bond-frame operations.

The formulas implement gated crystal messages and endpoint-local aggregation;
these modules have no dependency on an architecture implementation or dataset.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import LayerNorm, GraphNorm, InstanceNorm, PairNorm
from e3nn import o3
from net.transforms.composition import (
    activation,
    GaussianEnvelope,
    IndexedAggregation,
    ScalarActivation,
)


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


def _sum(x, index, size):
    return x.new_zeros((size, *x.shape[1:])).index_add(0, index, x)


class ScalarGatedNodeUpdate(nn.Module):
    """sigmoid-filtered softplus messages, target sum, normalization, residual.

    Directed edge_index[0] sends to edge_index[1], matching PyG's default flow.
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        *,
        normalization="layer",
        distance_envelope=True,
        envelope_sigma=3.0,
        filter_activation="sigmoid",
        message_activation="softplus",
        aggregation="sum",
        residual=True,
    ):
        super().__init__()
        self.lin_f = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.lin_s = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.norm = _normalization(node_dim, normalization)
        self.distance_envelope = distance_envelope
        self.envelope = GaussianEnvelope(envelope_sigma)
        self.filter_activation, self.message_activation = activation(
            filter_activation
        ), activation(message_activation)
        self.aggregation = IndexedAggregation(aggregation, receiver=1)
        self.residual = residual
        # This is part of the original from-scratch initialization schedule.
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if isinstance(self.norm, nn.BatchNorm1d):
            self.norm.reset_parameters()

    def forward(self, node, edge_index, edge, batch, distance):
        source, target = edge_index
        z = torch.cat([node[target], node[source], edge], -1)
        message = self.filter_activation(self.lin_f(z)) * self.message_activation(
            self.lin_s(z)
        )
        if self.distance_envelope:
            message = message * self.envelope(distance)
        update = _apply_norm(
            self.norm, self.aggregation(message, edge_index, node), batch
        )
        return node + update if self.residual else update


class EndpointScalarEdgeUpdate(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        out_dim,
        *,
        hidden_dim=128,
        final_activation=True,
        activation_name="silu",
    ):
        super().__init__()
        layers = [
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            ScalarActivation(activation_name),
            nn.Linear(hidden_dim, out_dim),
        ]
        if final_activation:
            layers.append(ScalarActivation(activation_name))
        self.e_lin = nn.Sequential(*layers)

    def forward(self, node, edge_index, edge):
        return self.e_lin(
            torch.cat([node[edge_index[0]], node[edge_index[1]], edge], -1)
        )


class LocalCoordinateMessage(nn.Module):
    """Two endpoint environments per output edge, with bond-local angles."""

    def __init__(
        self,
        node_dim,
        edge_dim,
        angular_dim,
        out_dim,
        *,
        hidden_dim=128,
        normalization="none",
        distance_envelope=True,
        envelope_sigma=3.0,
        filter_activation="sigmoid",
        message_activation="softplus",
        readout_activation="silu",
    ):
        super().__init__()
        self.lin_f = nn.Linear(2 * node_dim + edge_dim + angular_dim, node_dim)
        self.lin_s = nn.Linear(2 * node_dim + edge_dim + angular_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)
        self.e_lin = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            ScalarActivation(readout_activation),
            nn.Linear(hidden_dim, out_dim),
        )
        if normalization not in {"none", "batch"}:
            raise ValueError(
                "Local endpoint aggregation supports none or batch normalization"
            )
        self.normalization = normalization
        self.distance_envelope = distance_envelope
        self.envelope = GaussianEnvelope(envelope_sigma)
        self.filter_activation, self.message_activation = activation(
            filter_activation
        ), activation(message_activation)
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if normalization == "batch":
            self.bn.reset_parameters()

    def forward(
        self, node, edge, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance
    ):
        z = torch.cat(
            [
                node[sub_atom_idx[:, 0]],
                node[sub_atom_idx[:, 1]],
                edge[sub_edge_idx],
                sub_edge_ang,
            ],
            -1,
        )
        messages = self.filter_activation(self.lin_f(z)) * self.message_activation(
            self.lin_s(z)
        )
        if self.distance_envelope:
            messages = messages * self.envelope(distance[sub_edge_idx])
        endpoints = _sum(messages, sub_index, 2 * len(edge))
        if self.normalization == "batch":
            endpoints = self.bn(endpoints)
        endpoints = endpoints.reshape(len(edge), 2, -1)
        return self.e_lin(torch.cat([endpoints[:, 0], endpoints[:, 1], edge], -1))


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


def build_local_transform(name, p, inputs):
    from net.pipeline import FeatureSpec, _global_irreps

    if name == "irrep_to_bond":
        irreps = _global_irreps(inputs[0], "node")
        return IrrepToBondFrame(irreps, **p), [
            FeatureSpec("edge", "local_components", str(irreps), "bond")
        ]
    if name == "pair_distance_affine":
        return PairDistanceAffine(**p), [inputs[0]]
    if name == "channel_linear":
        if len(inputs) != 1:
            raise ValueError("channel_linear requires one feature input")
        source = inputs[0]
        if source.domain != "edge":
            raise ValueError("channel_linear requires edge-scoped features")
        if source.representation == "local_components" and source.frame == "bond":
            representation, frame = "local_components", "bond"
        else:
            irreps = _global_irreps(source, "edge")
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
            )
        ]
    node_ir = _global_irreps(inputs[0], "node")
    if any(ir.l or ir.p != 1 for _, ir in node_ir):
        raise ValueError(
            "Scalar message updates require invariant scalar node features"
        )
    node_dim = node_ir.dim
    if name == "scalar_gated_node":
        edge = _global_irreps(inputs[2], "edge")
        if any(ir.l or ir.p != 1 for _, ir in edge):
            raise ValueError("Scalar messages require invariant edge channels")
        return ScalarGatedNodeUpdate(node_dim, edge.dim, **p), [inputs[0]]
    if name == "endpoint_scalar_edge":
        edge = _global_irreps(inputs[2], "edge")
        if any(ir.l or ir.p != 1 for _, ir in edge):
            raise ValueError("Scalar edge updates require invariant edge channels")
        width = p["out_dim"]
        return EndpointScalarEdgeUpdate(node_dim, edge.dim, **p), [
            FeatureSpec("edge", irreps=f"{width}x0e", width=width)
        ]
    if name == "local_coordinate_message":
        edge = _global_irreps(inputs[1], "edge")
        if any(ir.l or ir.p != 1 for _, ir in edge):
            raise ValueError(
                "Local coordinate messages require invariant edge channels"
            )
        if inputs[4].frame != "bond":
            raise ValueError("Local messages require bond-frame angular features")
        return LocalCoordinateMessage(node_dim, edge.dim, **p), [
            FeatureSpec("edge", "local_components", frame="bond", width=p["out_dim"])
        ]
    raise ValueError(f"Unknown local transformation {name}")
