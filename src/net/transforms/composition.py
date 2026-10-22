"""Independent message maps, reducers, envelopes and equivariant connections."""

from dataclasses import replace
import math
import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter


def activation(name):
    choices = {
        "identity": lambda x: x,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "softplus": torch.nn.functional.softplus,
        "silu": torch.nn.functional.silu,
        "leakyrelu": torch.nn.functional.leaky_relu,
        "relu": torch.relu,
    }
    if name not in choices:
        raise ValueError(f"Unknown activation {name!r}")
    return choices[name]


class ScalarActivation(nn.Module):
    def __init__(self, name="silu"):
        super().__init__()
        self.function = activation(name)

    def forward(self, value):
        return self.function(value)


class EndpointGather(nn.Module):
    def forward(self, node, edge, index):
        return torch.cat((node[index[0]], node[index[1]], edge), dim=-1)


class IndexedAggregation(nn.Module):
    def __init__(self, reduction="sum", receiver=1):
        super().__init__()
        if reduction not in {"sum", "mean"} or receiver not in {0, 1}:
            raise ValueError("Aggregation requires sum/mean and receiver 0/1")
        self.reduction, self.receiver = reduction, receiver

    def forward(self, messages, index, node_reference):
        return scatter(
            messages,
            index[self.receiver],
            dim=0,
            dim_size=len(node_reference),
            reduce=self.reduction,
        )


class ProjectedAttention(nn.Module):
    """The original projected attention reducer, independent of its message map."""

    def __init__(
        self,
        node_irreps,
        message_irreps,
        radial_width,
        *,
        scalar_dim=64,
        heads=4,
        receiver=1,
        projection=None,
    ):
        super().__init__()
        from net.common import Config, E3MLP

        if scalar_dim < 1 or heads < 1 or scalar_dim % heads or receiver not in {0, 1}:
            raise ValueError("Attention dimensions/receiver are invalid")
        self.heads, self.head_dim, self.receiver = heads, scalar_dim // heads, receiver
        cfg = Config.from_mapping(projection or {})
        scalar = o3.Irreps(f"{scalar_dim}x0e")
        self.query_proj = E3MLP(
            node_irreps, o3.Irreps("64x0e"), scalar, 1, cfg, activate_last=False
        )
        self.key_proj = E3MLP(
            (o3.Irreps(f"{radial_width}x0e") + message_irreps).simplify(),
            o3.Irreps("64x0e"),
            scalar,
            1,
            cfg,
            activate_last=False,
        )
        self.value_projs = nn.ModuleList(
            [
                E3MLP(
                    message_irreps,
                    message_irreps,
                    message_irreps,
                    1,
                    cfg,
                    activate_last=False,
                )
                for _ in range(heads)
            ]
        )

    def forward(self, messages, nodes, radial, index):
        from net.layers import projected_attention_reduce

        return projected_attention_reduce(
            nodes,
            messages,
            radial,
            index[self.receiver],
            self.query_proj,
            self.key_proj,
            self.value_projs,
            self.heads,
            self.head_dim,
        )


class GaussianEnvelope(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Envelope sigma must be finite and positive")
        self.sigma = sigma

    def forward(self, lengths):
        return torch.exp(-lengths.reshape(-1, 1).square() / (2 * self.sigma**2))


class ScaleFeatures(nn.Module):
    def forward(self, values, weights):
        if weights.ndim == 1:
            weights = weights[:, None]
        return values * weights


class PeriodicDisplacement(nn.Module):
    def forward(self, positions, box, index, shifts, node_batch):
        from data.graph_topology import edge_displacements

        return edge_displacements(
            {"positions": positions, "box": box, "node_batch": node_batch},
            index,
            shifts,
        )


class VectorLengths(nn.Module):
    def forward(self, vectors):
        return vectors.norm(dim=-1)


class BondFrames(nn.Module):
    def forward(self, index, vectors):
        from data.bond_geometry import bond_frames

        return bond_frames(index, vectors)


class BondHarmonics(nn.Module):
    def __init__(self, num_l=5):
        super().__init__()
        if isinstance(num_l, bool) or not isinstance(num_l, int) or num_l < 1:
            raise ValueError("num_l must be a positive integer")
        self.num_l = num_l

    def forward(self, vectors, frames, edge_reference, endpoint_reference):
        from data.bond_geometry import tesseral_harmonics

        local = -torch.einsum(
            "ei,eij->ej", vectors[edge_reference], frames[endpoint_reference // 2]
        )
        return tesseral_harmonics(local, self.num_l)


class ScalarGatedMessage(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        filter_activation="sigmoid",
        message_activation="softplus",
    ):
        super().__init__()
        self.lin_f, self.lin_s = nn.Linear(in_features, out_features), nn.Linear(
            in_features, out_features
        )
        self.filter, self.message = activation(filter_activation), activation(
            message_activation
        )

    def forward(self, values):
        return self.filter(self.lin_f(values)) * self.message(self.lin_s(values))


def register_components(register):
    from net.pipeline import (
        FeatureSpec,
        _global_irreps,
        _same_support,
        _same_angular_basis,
        _index_spec,
    )

    @register("periodic_displacement")
    def displacement(p, inputs, mapper):
        if (
            p
            or len(inputs) != 5
            or inputs[0].domain != "node"
            or inputs[0].representation != "geometry"
            or inputs[1].domain != "graph"
        ):
            raise ValueError(
                "Displacement expects positions, cell, edge index, shifts and node batch"
            )
        _index_spec(inputs[2], "relation")
        _index_spec(inputs[3], "relation")
        _index_spec(inputs[4], "node")
        support = _same_support(inputs[2], inputs[3])
        return PeriodicDisplacement(), [
            FeatureSpec("edge", "geometry", width=3, support=support)
        ]

    @register("vector_lengths")
    def lengths(p, inputs, mapper):
        if (
            p
            or len(inputs) != 1
            or inputs[0].representation != "geometry"
            or inputs[0].dim != 3
        ):
            raise ValueError("Vector lengths expect Cartesian vectors")
        return VectorLengths(), [
            FeatureSpec(inputs[0].domain, width=1, support=inputs[0].support)
        ]

    @register("bond_frames")
    def frames(p, inputs, mapper):
        if (
            p
            or len(inputs) != 2
            or inputs[1].representation != "geometry"
            or inputs[1].dim != 3
        ):
            raise ValueError("Bond frames expect an index and Cartesian vectors")
        _index_spec(inputs[0], "relation")
        support = _same_support(*inputs)
        return BondFrames(), [
            FeatureSpec("edge", "geometry", frame="bond", width=9, support=support)
        ]

    @register("bond_harmonics")
    def local_angles(p, inputs, mapper):
        if (
            len(inputs) != 4
            or inputs[0].representation != "geometry"
            or inputs[1].frame != "bond"
        ):
            raise ValueError(
                "Bond harmonics expect vectors, frames and neighborhood references"
            )
        _index_spec(inputs[2], "bond_neighbor")
        _index_spec(inputs[3], "bond_neighbor")
        support = _same_support(*inputs)
        module = BondHarmonics(**p)
        return module, [
            FeatureSpec(
                "bond_neighbor",
                "local_components",
                frame="bond",
                width=module.num_l**2,
                support=support,
            )
        ]

    @register("endpoint_gather")
    def gather(p, inputs, mapper):
        if p or len(inputs) != 3:
            raise ValueError("Endpoint gathering expects nodes, edges and an index")
        node, edge = _global_irreps(inputs[0], "node"), _global_irreps(
            inputs[1], "edge"
        )
        _index_spec(inputs[2], "relation")
        _same_support(inputs[1], inputs[2])
        basis = _same_angular_basis(inputs[0], inputs[1])
        return EndpointGather(), [
            FeatureSpec(
                "edge",
                "irreps",
                str(node + node + edge),
                support=inputs[1].support,
                angular_basis=basis,
            )
        ]

    @register("indexed_aggregation")
    def aggregation(p, inputs, mapper):
        if len(inputs) != 3:
            raise ValueError("Aggregation expects messages, index and node reference")
        _global_irreps(inputs[0], "edge")
        _global_irreps(inputs[2], "node")
        _index_spec(inputs[1], "relation")
        _same_support(inputs[0], inputs[1])
        _same_angular_basis(inputs[0], inputs[2])
        return IndexedAggregation(**p), [
            replace(inputs[0], domain="node", support=inputs[2].support)
        ]

    @register("projected_attention")
    def attention(p, inputs, mapper):
        if len(inputs) != 4:
            raise ValueError("Attention expects messages, nodes, radial and index")
        messages, nodes, radial = (
            _global_irreps(inputs[0], "edge"),
            _global_irreps(inputs[1], "node"),
            _global_irreps(inputs[2], "edge"),
        )
        _index_spec(inputs[3], "relation")
        _same_support(inputs[0], inputs[2], inputs[3])
        _same_angular_basis(inputs[0], inputs[1])
        if any(ir.l or ir.p != 1 for _, ir in radial):
            raise ValueError("Attention radial features must be invariant scalars")
        return ProjectedAttention(nodes, messages, radial.dim, **p), [
            replace(inputs[0], domain="node", support=inputs[1].support)
        ]

    @register("radial_tensor_product")
    def product(p, inputs, mapper):
        from net.transforms.equivariant import GatedRadialTensorProduct

        if len(inputs) != 4:
            raise ValueError(
                "Radial tensor product expects features, harmonics, radial and edge batch"
            )
        value, angular, radial = [_global_irreps(spec, "edge") for spec in inputs[:3]]
        _index_spec(inputs[3], "edge")
        support = _same_support(*inputs)
        basis = _same_angular_basis(inputs[0], inputs[1])
        if any(ir.l or ir.p != 1 for _, ir in radial):
            raise ValueError("Radial features must be invariant scalars")
        module = GatedRadialTensorProduct(radial.dim, value, angular, **p)
        return module, [
            FeatureSpec(
                "edge",
                "irreps",
                str(module.irreps_out),
                support=support,
                angular_basis=basis,
            )
        ]

    @register("gaussian_envelope")
    def envelope(p, inputs, mapper):
        if len(inputs) != 1 or inputs[0].dim != 1 or inputs[0].domain != "edge":
            raise ValueError("Gaussian envelope requires edge distances")
        return GaussianEnvelope(**p), [
            FeatureSpec("edge", "irreps", "1x0e", support=inputs[0].support)
        ]

    @register("scale_features")
    def scale(p, inputs, mapper):
        if p or len(inputs) != 2:
            raise ValueError("Scaling expects values and one invariant weight")
        _same_support(*inputs)
        if _global_irreps(inputs[1]) != o3.Irreps("1x0e"):
            raise ValueError("Scaling weights must be invariant scalars")
        return ScaleFeatures(), [inputs[0]]

    @register("gated_scalar_message")
    def scalar_message(p, inputs, mapper):
        irreps = _global_irreps(inputs[0])
        if len(inputs) != 1 or any(ir.l or ir.p != 1 for _, ir in irreps):
            raise ValueError("Scalar messages require invariant features")
        width = p["out_features"]
        return ScalarGatedMessage(irreps.dim, **p), [
            replace(
                inputs[0], representation="irreps", irreps=f"{width}x0e", width=width
            )
        ]

    @register("irrep_residual")
    def residual(p, inputs, mapper):
        from net.transforms.equivariant import IrrepResidual

        if p or len(inputs) != 2 or inputs[0].domain != inputs[1].domain:
            raise ValueError("Residual expects old/new features on the same entities")
        _same_support(*inputs)
        _same_angular_basis(*inputs)
        return IrrepResidual(*[_global_irreps(spec) for spec in inputs]), [inputs[1]]

    @register("species_tensor_product")
    def species(p, inputs, mapper):
        if len(inputs) != 2 or inputs[0].domain != inputs[1].domain:
            raise ValueError("Species connection expects matching entities")
        _same_support(*inputs)
        value, attrs = map(_global_irreps, inputs)
        if any(ir.l or ir.p != 1 for _, ir in attrs):
            raise ValueError("Species attributes must be invariant scalars")
        module = o3.FullyConnectedTensorProduct(value, attrs, **p)
        return module, [replace(inputs[0], irreps=str(module.irreps_out), width=None)]

    @register("irrep_activation")
    def nonlinearity(p, inputs, mapper):
        from e3nn.nn import Gate, NormActivation

        if len(inputs) != 1:
            raise ValueError("Activation requires one irrep input")
        irreps = _global_irreps(inputs[0])
        p = dict(p)
        kind = p.pop("kind", "norm")
        if kind == "norm":
            module = NormActivation(
                irreps, activation(p.pop("activation", "silu")), **p
            )
        elif kind == "gate":
            scalars, gates, gated = [
                o3.Irreps(p.pop(key))
                for key in ("irreps_scalars", "irreps_gates", "irreps_gated")
            ]
            scalar_act = activation(p.pop("scalar_activation", "silu"))
            gate_act = activation(p.pop("gate_activation", "sigmoid"))
            if p:
                raise ValueError(f"Unknown gate options {sorted(p)}")
            module = Gate(
                scalars,
                [scalar_act] * len(scalars),
                gates,
                [gate_act] * len(gates),
                gated,
            )
            if module.irreps_in != irreps:
                raise ValueError(
                    "Gate input irreps must include the declared gate channels"
                )
        else:
            raise ValueError(f"Unknown irrep activation {kind}")
        return module, [replace(inputs[0], irreps=str(module.irreps_out), width=None)]
