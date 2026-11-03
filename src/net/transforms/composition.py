"""Independent message maps, reducers, envelopes and equivariant connections."""

from dataclasses import replace
import math
import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter

from net.features import (
    FeatureSpec,
    global_irreps,
    index_spec,
    same_angular_basis,
    same_support,
)
from net.registry import Option, Port, infer_from, register_transform
from net.transforms.settings import (
    ACTIVATION_OPTIONS,
    MLP_OPTIONS,
    MLP_VARIANTS,
    layer_settings,
)


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


class GatherRows(nn.Module):
    """Rows of ``values`` selected by an index (one column of a pair index)."""

    def __init__(self, endpoint=None, pair_axis=0):
        super().__init__()
        self.endpoint, self.pair_axis = endpoint, pair_axis

    def forward(self, values, index):
        if self.endpoint is not None:
            index = index.select(self.pair_axis, self.endpoint)
        return values.index_select(0, index)


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


def projected_attention_reduce(
    node,
    edge_messages,
    edge_length_emb,
    dst,
    query_proj,
    key_proj,
    value_projs,
    num_heads,
    head_dim,
):
    """Shared invariant projected multihead attention reduction."""
    query = query_proj(node).reshape(node.shape[0], num_heads, head_dim)
    key_input = torch.cat([edge_length_emb, edge_messages], dim=-1)
    key = key_proj(key_input).reshape(edge_messages.shape[0], num_heads, head_dim)
    values = torch.stack([proj(edge_messages) for proj in value_projs], dim=1)
    scores = (query[dst] * key).sum(dim=-1) / float(head_dim) ** 0.5
    max_per_dst = scatter(scores, dst, dim=0, dim_size=node.size(0), reduce="max")
    scores = scores - max_per_dst.index_select(0, dst)
    exp_scores = torch.exp(scores)
    denom = scatter(exp_scores, dst, dim=0, dim_size=node.size(0), reduce="sum")
    weights = exp_scores / denom.index_select(0, dst).clamp_min(1e-12)
    weighted_values = values * weights.unsqueeze(-1)
    head_outputs = scatter(
        weighted_values,
        dst,
        dim=0,
        dim_size=node.size(0),
        reduce="sum",
    )
    return head_outputs.mean(dim=1)


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
        settings=None,
    ):
        super().__init__()
        from net.common import E3MLP
        from net.transforms.settings import layer_settings

        if scalar_dim < 1 or heads < 1 or scalar_dim % heads or receiver not in {0, 1}:
            raise ValueError("Attention dimensions/receiver are invalid")
        self.heads, self.head_dim, self.receiver = heads, scalar_dim // heads, receiver
        cfg = settings if settings is not None else layer_settings({})
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
        weights = weights.reshape(-1, *([1] * (values.ndim - 1)))
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
        from core.norms import zero_safe_norm

        return zero_safe_norm(vectors, dim=-1)


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


_ACTIVATIONS = ("identity", "sigmoid", "tanh", "softplus", "silu", "leakyrelu", "relu")


@register_transform(
    "periodic_displacement",
    category="geometry",
    summary="Edge vectors r_j - r_i + L from positions, cell and lattice shifts",
    inputs=[
        Port("positions", "node", "geometry"),
        Port("cell", "graph", "geometry"),
        Port("edge_index", "relation", "index"),
        Port("shifts", "relation", "index"),
        Port("node_batch", "node", "index"),
    ],
)
def _periodic_displacement(p, inputs, mapper):
    if (
        inputs[0].domain != "node"
        or inputs[0].representation != "geometry"
        or inputs[1].domain != "graph"
    ):
        raise ValueError(
            "Displacement expects positions, cell, edge index, shifts and node batch"
        )
    index_spec(inputs[2], "relation")
    index_spec(inputs[3], "relation")
    index_spec(inputs[4], "node")
    support = same_support(inputs[2], inputs[3])
    return PeriodicDisplacement(), [
        FeatureSpec("edge", "geometry", width=3, support=support)
    ]


@register_transform(
    "vector_lengths",
    category="geometry",
    summary="Euclidean length of each vector",
    inputs=[Port("vectors", None, "geometry")],
)
def _vector_lengths(p, inputs, mapper):
    if inputs[0].representation != "geometry" or inputs[0].dim != 3:
        raise ValueError("Vector lengths expect Cartesian vectors")
    return VectorLengths(), [
        FeatureSpec(inputs[0].domain, width=1, support=inputs[0].support)
    ]


@register_transform(
    "bond_frames",
    category="frame",
    summary="Local orthonormal frame of each bond (DeepH)",
    inputs=[
        Port("edge_index", "relation", "index"),
        Port("vectors", "edge", "geometry"),
    ],
)
def _bond_frames(p, inputs, mapper):
    if inputs[1].representation != "geometry" or inputs[1].dim != 3:
        raise ValueError("Bond frames expect an index and Cartesian vectors")
    index_spec(inputs[0], "relation")
    support = same_support(*inputs)
    return BondFrames(), [
        FeatureSpec("edge", "geometry", frame="bond", width=9, support=support)
    ]


@register_transform(
    "bond_harmonics",
    category="angular_basis",
    summary="Tesseral harmonics of neighbor directions in each bond frame",
    inputs=[
        Port("vectors", "edge", "geometry"),
        Port("frames", "edge", "geometry"),
        Port("edge_reference", "bond_neighbor", "index"),
        Port("endpoint_reference", "bond_neighbor", "index"),
    ],
    options=[Option("num_l", "int", 5, key=True)],
)
def _bond_harmonics(p, inputs, mapper):
    if inputs[0].representation != "geometry" or inputs[1].frame != "bond":
        raise ValueError(
            "Bond harmonics expect vectors, frames and neighborhood references"
        )
    index_spec(inputs[2], "bond_neighbor")
    index_spec(inputs[3], "bond_neighbor")
    support = same_support(*inputs)
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


@register_transform(
    "gather_rows",
    category="combine",
    summary="Rows of a feature selected by an index, e.g. h[i] for every edge i -> j",
    inputs=[
        Port("values"),
        Port("index", None, "index", doc="A pair index needs the endpoint option"),
    ],
    options=[
        Option(
            "endpoint",
            "choice",
            None,
            key=True,
            choices=(None, 0, 1),
            doc="Column of a two-endpoint index (0 sender, 1 receiver)",
        ),
    ],
)
def _gather_rows(p, inputs, mapper):
    values, index = inputs
    if index.representation != "index":
        raise ValueError("gather_rows needs an integer index")
    if values.domain == "graph":
        raise ValueError("gather_rows selects rows of entity features")
    pair = index.domain == "relation" or (
        index.domain == "bond_neighbor" and p["endpoint"] is not None
    )
    if index.domain == "relation" and p["endpoint"] is None:
        raise ValueError("An edge index has two endpoints; choose endpoint 0 or 1")
    domain = "edge" if index.domain == "relation" else index.domain
    module = GatherRows(p["endpoint"], 0 if index.domain == "relation" else 1)
    if not pair and p["endpoint"] is not None:
        raise ValueError("endpoint applies to two-endpoint indices only")
    return module, [replace(values, domain=domain, support=index.support)]


@register_transform(
    "indexed_aggregation",
    category="aggregation",
    summary="Sum or mean of edge messages onto their receiving nodes",
    inputs=[
        Port("messages", "edge", "irreps"),
        Port("edge_index", "relation", "index"),
        Port("nodes", "node", doc="Reference for the node count"),
    ],
    options=[
        Option("reduction", "choice", "sum", key=True, choices=("sum", "mean")),
        Option(
            "receiver",
            "choice",
            1,
            choices=(0, 1),
            doc="Row of edge_index that receives",
        ),
    ],
)
def _indexed_aggregation(p, inputs, mapper):
    global_irreps(inputs[0], "edge")
    index_spec(inputs[1], "relation")
    same_support(inputs[0], inputs[1])
    same_angular_basis(inputs[0], inputs[2])
    return IndexedAggregation(**p), [
        replace(inputs[0], domain="node", support=inputs[2].support)
    ]


@register_transform(
    "projected_attention",
    category="aggregation",
    summary="Multi-head softmax attention over incoming messages with invariant scores",
    inputs=[
        Port("messages", "edge", "irreps"),
        Port("nodes", "node", "irreps", doc="Queries"),
        Port("radial", "edge", "invariant"),
        Port("edge_index", "relation", "index"),
    ],
    options=[
        Option("scalar_dim", "int", 64, key=True, doc="Query/key width"),
        Option("heads", "int", 4, key=True),
        Option("receiver", "choice", 1, choices=(0, 1)),
        Option(
            "variant",
            "choice",
            "film",
            choices=MLP_VARIANTS,
            doc="MLP variant of the projections",
        ),
        *ACTIVATION_OPTIONS,
        *MLP_OPTIONS,
    ],
)
def _projected_attention(p, inputs, mapper):
    messages = global_irreps(inputs[0], "edge")
    nodes = global_irreps(inputs[1], "node")
    radial = global_irreps(inputs[2], "edge")
    index_spec(inputs[3], "relation")
    same_support(inputs[0], inputs[2], inputs[3])
    same_angular_basis(inputs[0], inputs[1])
    if any(ir.l or ir.p != 1 for _, ir in radial):
        raise ValueError("Attention radial features must be invariant scalars")
    module = ProjectedAttention(
        nodes,
        messages,
        radial.dim,
        scalar_dim=p["scalar_dim"],
        heads=p["heads"],
        receiver=p["receiver"],
        settings=layer_settings(p, variant=p["variant"]),
    )
    return module, [replace(inputs[0], domain="node", support=inputs[1].support)]


@register_transform(
    "gaussian_envelope",
    category="envelope",
    summary="exp(-r^2 / 2 sigma^2) distance envelope",
    inputs=[Port("lengths")],
    options=[Option("sigma", "float", 3.0, key=True)],
)
def _gaussian_envelope(p, inputs, mapper):
    if inputs[0].dim != 1 or inputs[0].representation not in {"scalar", "irreps"}:
        raise ValueError("Gaussian envelope requires one distance per row")
    return GaussianEnvelope(**p), [
        FeatureSpec(inputs[0].domain, "irreps", "1x0e", support=inputs[0].support)
    ]


@register_transform(
    "scale_features",
    category="combine",
    summary="Multiply every channel by one invariant weight",
    inputs=[Port("values"), Port("weight", None, "invariant")],
)
def _scale_features(p, inputs, mapper):
    same_support(*inputs)
    if global_irreps(inputs[1]) != o3.Irreps("1x0e"):
        raise ValueError("Scaling weights must be one invariant scalar")
    return ScaleFeatures(), [inputs[0]]


@register_transform(
    "gated_scalar_message",
    category="message",
    summary="filter(W_f x) * message(W_s x) on invariant channels",
    inputs=[Port("features", None, "invariant")],
    options=[
        Option("out_features", "int", key=True),
        Option("filter_activation", "choice", "sigmoid", choices=_ACTIVATIONS),
        Option("message_activation", "choice", "softplus", choices=_ACTIVATIONS),
    ],
)
def _gated_scalar_message(p, inputs, mapper):
    from net.transforms.local_coordinates import _invariant_width

    width_in = _invariant_width(inputs[0], "Scalar messages")
    width = p["out_features"]
    if inputs[0].representation == "local_components":
        output = replace(inputs[0], width=width)
    else:
        output = replace(
            inputs[0], representation="irreps", irreps=f"{width}x0e", width=width
        )
    return ScalarGatedMessage(width_in, **p), [output]


@register_transform(
    "species_tensor_product",
    category="linear",
    summary="Species-dependent linear map: tensor product with species one-hot",
    inputs=[Port("features", None, "irreps"), Port("attributes", None, "invariant")],
    options=[
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=infer_from(0),
            infer_doc="input irreps",
        ),
    ],
)
def _species_tensor_product(p, inputs, mapper):
    if inputs[0].domain != inputs[1].domain:
        raise ValueError("Species connection expects matching entities")
    same_support(*inputs)
    value, attrs = map(global_irreps, inputs)
    if any(ir.l or ir.p != 1 for _, ir in attrs):
        raise ValueError("Species attributes must be invariant scalars")
    module = o3.FullyConnectedTensorProduct(value, attrs, p["irreps_out"])
    return module, [replace(inputs[0], irreps=str(module.irreps_out), width=None)]


@register_transform(
    "irrep_activation",
    category="nonlinearity",
    summary="Norm activation, or a gate whose scalars are listed explicitly",
    inputs=[Port("features", None, "irreps")],
    options=[
        Option("kind", "choice", "norm", key=True, choices=("norm", "gate")),
        Option("activation", "choice", "silu", choices=_ACTIVATIONS),
        Option("irreps_scalars", "irreps", None),
        Option("irreps_gates", "irreps", None),
        Option("irreps_gated", "irreps", None),
        Option("scalar_activation", "choice", "silu", choices=_ACTIVATIONS),
        Option("gate_activation", "choice", "sigmoid", choices=_ACTIVATIONS),
        Option("bias", "bool", False),
        Option("normalize", "bool", True),
    ],
)
def _irrep_activation(p, inputs, mapper):
    from e3nn.nn import Gate, NormActivation

    irreps = global_irreps(inputs[0])
    if p["kind"] == "norm":
        module = NormActivation(
            irreps,
            activation(p["activation"]),
            normalize=p["normalize"],
            bias=p["bias"],
        )
    else:
        if None in (p["irreps_scalars"], p["irreps_gates"], p["irreps_gated"]):
            raise ValueError(
                "Gate activation requires irreps_scalars, irreps_gates and irreps_gated"
            )
        scalars, gates, gated = (
            o3.Irreps(p[key])
            for key in ("irreps_scalars", "irreps_gates", "irreps_gated")
        )
        module = Gate(
            scalars,
            [activation(p["scalar_activation"])] * len(scalars),
            gates,
            [activation(p["gate_activation"])] * len(gates),
            gated,
        )
        if module.irreps_in != irreps:
            raise ValueError(
                "Gate input irreps must include the declared gate channels"
            )
    return module, [replace(inputs[0], irreps=str(module.irreps_out), width=None)]
