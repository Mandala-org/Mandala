"""The MANDALA layer library as independent, routable pipeline stages.

Each stage is one mathematical operation of the MANDALA E3GNN: equivariant
MLPs, the gated radial convolution, endpoint assembly, self-connections,
nonlinearities, normalization, dropout and the per species-pair readout. The
flat configuration lowers to these stages (``net.component_defaults``), and
recipes combine them freely with every other registered transformation.

Routing gives a layer separate parameters per category of its rows (edge role,
species pair, species): each copy is evaluated only on the rows of its
category.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from e3nn import o3
from e3nn.nn import Dropout, Gate
from torch import nn

from net.features import (
    FeatureSpec,
    global_irreps,
    index_spec,
    invariant_irreps,
    irrep_feature,
    require_canonical,
    same_support,
)
from net.registry import Option, Port, infer_from, register_transform
from net.transforms.settings import (
    ACTIVATION_OPTIONS,
    MLP_OPTIONS,
    MLP_VARIANTS,
    NONLINEARITIES,
    SCALAR_ACTIVATIONS,
    layer_settings,
)

ROUTE_OPTION = Option(
    "routes",
    "int",
    1,
    infer=lambda inputs, mapper, options: _route_count(inputs),
    infer_doc="number of categories of the route index",
    doc="Separate parameter copies, one per category of the route input",
)
INACTIVE_OPTION = Option(
    "inactive_routes",
    "list",
    (),
    doc="Categories (indices or labels) without parameters; their rows output zero",
)


def route_indices(values, spec, routes):
    """Category indices for ``values`` given as indices or labels of ``spec``."""
    labels = (spec.categories or ()) if spec is not None else ()
    indices = []
    for value in values:
        if isinstance(value, str):
            if value not in labels:
                raise ValueError(
                    f"Unknown route category {value!r}; categories are {list(labels)}"
                )
            value = labels.index(value)
        if not 0 <= int(value) < routes:
            raise ValueError(f"Route category {value} is outside 0..{routes - 1}")
        indices.append(int(value))
    return indices


def _route_count(inputs):
    route = next((spec for spec in inputs[1:] if spec.representation == "index"), None)
    return route.width if route is not None and route.width else None


def _sum_operations(modules):
    from net.cost_model import operations_per_entity

    return sum(operations_per_entity(module) for module in modules)


class _Inactive(nn.Module):
    """Placeholder of a category without parameters; its rows output zero."""


class Routed(nn.Module):
    """Independent parameter copies selected per row by an integer category."""

    def __init__(self, copies, out_dim):
        super().__init__()
        self.routes = nn.ModuleList(copies)
        self.out_dim = int(out_dim)

    def forward(self, values, route):
        # One host synchronization (the category sizes) per call.
        counts = torch.bincount(route, minlength=len(self.routes)).tolist()
        order = torch.argsort(route, stable=True)
        result = values.new_zeros(values.shape[0], self.out_dim)
        start = 0
        for module, count in zip(self.routes, counts):
            rows = order[start : start + count]
            start += count
            if count and not isinstance(module, _Inactive):
                result.index_copy_(0, rows, module(values.index_select(0, rows)))
        return result

    def operations_per_entity(self):
        active = [m for m in self.routes if not isinstance(m, _Inactive)]
        return _sum_operations(active) / max(1, len(self.routes))


def _route_port(doc="Category index selecting the parameter copy"):
    return Port("route", ("node", "edge"), "index", optional=True, doc=doc)


def _check_route(inputs, routes, position=1):
    if len(inputs) > position:
        index_spec(inputs[position], inputs[0].domain)
        same_support(inputs[0], inputs[position])
        width = inputs[position].width
        if width and routes != width:
            raise ValueError(
                f"routes={routes} but the route index has {width} categories; "
                "every category needs a copy (use inactive_routes to leave one empty)"
            )
    elif routes > 1:
        raise ValueError("routes > 1 requires a route index input")


def _build_routable(builder, inputs, routes, out_dim, inactive=()):
    """One module, or one copy per category when a route index is wired."""
    _check_route(inputs, routes)
    if len(inputs) == 1:
        if inactive:
            raise ValueError("inactive_routes requires a route index input")
        return builder(0)
    inactive = set(route_indices(inactive, inputs[1], routes))
    copies = [_Inactive() if c in inactive else builder(c) for c in range(routes)]
    return Routed(copies, out_dim)


@register_transform(
    "e3mlp",
    category="mlp",
    summary="Equivariant MLP: linear maps between irreps with equivariant activations",
    inputs=[Port("features", None, "irreps"), _route_port()],
    options=[
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=infer_from(0),
            infer_doc="input irreps",
        ),
        Option(
            "irreps_hidden", "irreps", infer=infer_from(0), infer_doc="input irreps"
        ),
        Option("layers", "int", 1, key=True, doc="Number of linear layers"),
        Option("variant", "choice", "film", key=True, choices=MLP_VARIANTS),
        Option("activate_last", "bool", False),
        Option("post_scale", "float", 1.0),
        ROUTE_OPTION,
        INACTIVE_OPTION,
        *ACTIVATION_OPTIONS,
        *MLP_OPTIONS,
    ],
)
def _e3mlp(p, inputs, mapper):
    from net.common import E3MLP

    irreps_in = global_irreps(inputs[0])
    hidden, out = o3.Irreps(p["irreps_hidden"]), o3.Irreps(p["irreps_out"])
    cfg = layer_settings(p, variant=p["variant"])

    def build(category):
        return E3MLP(
            irreps_in,
            hidden,
            out,
            p["layers"],
            cfg,
            activate_last=p["activate_last"],
            post_scale=p["post_scale"],
        )

    module = _build_routable(build, inputs, p["routes"], out.dim, p["inactive_routes"])
    return module, [irrep_feature(inputs[0], out)]


@register_transform(
    "tensor_square",
    category="product",
    summary="Symmetric tensor square x (x) x projected to the requested irreps",
    inputs=[Port("features", None, "irreps"), _route_port()],
    options=[
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=infer_from(0),
            infer_doc="input irreps",
        ),
        ROUTE_OPTION,
        INACTIVE_OPTION,
    ],
)
def _tensor_square(p, inputs, mapper):
    irreps_in = global_irreps(inputs[0])
    out = o3.Irreps(p["irreps_out"])

    def build(category):
        return o3.TensorSquare(irreps_in, irreps_out=out)

    module = _build_routable(build, inputs, p["routes"], out.dim, p["inactive_routes"])
    return module, [irrep_feature(inputs[0], out)]


@register_transform(
    "fc_tensor_product",
    category="product",
    summary="Fully connected tensor product of two features on the same entities",
    inputs=[Port("left", None, "irreps"), Port("right", None, "irreps")],
    options=[
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=infer_from(0),
            infer_doc="left irreps",
        )
    ],
)
def _fc_tensor_product(p, inputs, mapper):
    if inputs[0].domain != inputs[1].domain:
        raise ValueError("Tensor product operands must share their entities")
    same_support(*inputs)
    require_canonical(*inputs)
    out = o3.Irreps(p["irreps_out"])
    module = o3.FullyConnectedTensorProduct(
        global_irreps(inputs[0]),
        global_irreps(inputs[1]),
        out,
        internal_weights=True,
        shared_weights=True,
    )
    return module, [irrep_feature(inputs[0], out)]


class EndpointConcat(nn.Module):
    """Assemble ``[context(h_i, h_j), e_ij]`` for every edge ``i -> j``."""

    def __init__(self, combine, node_irreps, order="sender_receiver"):
        super().__init__()
        self.combine = combine
        self.first = 0 if order == "sender_receiver" else 1
        self.pair_tp = (
            o3.FullyConnectedTensorProduct(
                node_irreps,
                node_irreps,
                node_irreps,
                internal_weights=True,
                shared_weights=True,
            )
            if combine == "tensor_product"
            else None
        )

    def forward(self, nodes, edges, edge_index):
        sender, receiver = nodes[edge_index[0]], nodes[edge_index[1]]
        if self.combine == "concat":
            first, second = (
                (sender, receiver) if self.first == 0 else (receiver, sender)
            )
            return torch.cat([first, second, edges], dim=-1)
        if self.combine == "sum":
            context = sender + receiver
        else:
            context = self.pair_tp(sender, receiver)
        return torch.cat([context, edges], dim=-1)


@register_transform(
    "endpoint_concat",
    category="combine",
    summary="Per edge: [h_i, h_j, e_ij], or [h_i + h_j, e_ij], or [TP(h_i, h_j), e_ij]",
    inputs=[
        Port("nodes", "node", "irreps"),
        Port("edges", "edge", "irreps"),
        Port("edge_index", "relation", "index"),
    ],
    options=[
        Option(
            "combine",
            "choice",
            "concat",
            key=True,
            choices=("concat", "sum", "tensor_product"),
        ),
        Option(
            "simplify", "bool", True, doc="Merge adjacent equal irreps of the result"
        ),
        Option(
            "order",
            "choice",
            "sender_receiver",
            choices=("sender_receiver", "receiver_sender"),
            doc="Endpoint order of the concatenation",
        ),
    ],
)
def _endpoint_concat(p, inputs, mapper):
    nodes, edges = global_irreps(inputs[0], "node"), global_irreps(inputs[1], "edge")
    index_spec(inputs[2], "relation")
    same_support(inputs[1], inputs[2])
    require_canonical(inputs[0], inputs[1])
    context = nodes + nodes if p["combine"] == "concat" else nodes
    irreps = context + edges
    if p["simplify"]:
        irreps = irreps.simplify()
    module = EndpointConcat(p["combine"], nodes, p["order"])
    return module, [irrep_feature(inputs[1], irreps)]


def _reachable(irreps_in1, irreps_in2, ir_out):
    return any(ir_out in ir1 * ir2 for _, ir1 in irreps_in1 for _, ir2 in irreps_in2)


class EquivariantConvolution(nn.Module):
    """Gated tensor product with the edge harmonics, weighted by a radial MLP.

    ``z = Gate(TP(x, Y(r_hat))) * MLP(radial)``; the radial weights are shared
    by every component of an irrep term (``term``) or of an irrep copy
    (``copy``). Output irreps are the requested ones reachable by the product.
    """

    def __init__(
        self,
        radial_dim,
        irreps_in1,
        irreps_in2,
        irreps_out,
        *,
        tensor_product="separate_weight",
        gate=True,
        radial_layers=(64, 64),
        radial_weight_granularity="term",
        num_pairs=None,
        pair_embedding_dim=0,
    ):
        super().__init__()
        from net.common import RadialMLP, SeparateWeightTensorProduct

        irreps_in1, irreps_in2 = o3.Irreps(irreps_in1), o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)
        dtype = torch.get_default_dtype()
        radial_in = radial_dim
        self.pair_emb = None
        if pair_embedding_dim:
            self.pair_emb = nn.Embedding(num_pairs, pair_embedding_dim, dtype=dtype)
            nn.init.normal_(self.pair_emb.weight, std=0.2)
            radial_in += pair_embedding_dim
        self.nonlin = None
        if gate:
            scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_out
                    if ir.l == 0 and _reachable(irreps_in1, irreps_in2, ir)
                ]
            ).simplify()
            gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_out
                    if ir.l > 0 and _reachable(irreps_in1, irreps_in2, ir)
                ]
            ).simplify()
            gates = o3.Irreps([])
            if gated.dim > 0:
                for candidate in ("0e", "0o"):
                    if _reachable(irreps_in1, irreps_in2, o3.Irrep(candidate)):
                        gates = o3.Irreps(
                            [(mul, candidate) for mul, _ in gated]
                        ).simplify()
                        break
                else:
                    raise ValueError(f"Cannot produce gates for irreps_gated={gated}")
            self.nonlin = Gate(
                scalars,
                [
                    torch.nn.functional.silu if ir.p == 1 else torch.tanh
                    for _, ir in scalars
                ],
                gates,
                [torch.sigmoid if ir.p == 1 else torch.tanh for _, ir in gates],
                gated,
            )
            irreps_tp_out = self.nonlin.irreps_in
            self.irreps_out = self.nonlin.irreps_out
        else:
            irreps_tp_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_out
                    if _reachable(irreps_in1, irreps_in2, ir)
                ]
            )
            self.irreps_out = irreps_tp_out
        if tensor_product == "separate_weight":
            self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)
        elif tensor_product == "fully_connected":
            self.tp = o3.FullyConnectedTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_tp_out,
                internal_weights=True,
                shared_weights=True,
            )
        else:
            raise ValueError(f"Unknown tensor product {tensor_product!r}")
        if radial_weight_granularity not in {"term", "copy"}:
            raise ValueError("radial_weight_granularity must be 'term' or 'copy'.")
        self.radial_mlp = RadialMLP(
            in_dim=radial_in,
            out_dim=self.irreps_out.num_irreps,
            layers=tuple(radial_layers),
            act="silu",
            dtype=dtype,
        )
        index, offset = [], 0
        for term, (mul, ir) in enumerate(self.irreps_out):
            if radial_weight_granularity == "term":
                index.extend([term] * (mul * ir.dim))
            else:
                for copy in range(mul):
                    index.extend([offset + copy] * ir.dim)
            offset += mul
        self.register_buffer(
            "_radial_weight_index",
            torch.tensor(index, dtype=torch.long),
            persistent=False,
        )

    def forward(self, features, harmonics, radial, pair_type=None):
        z = self.tp(features, harmonics)
        if z.shape[-1] == 0:
            return z
        if self.nonlin is not None:
            z = self.nonlin(z)
        if self.pair_emb is not None:
            radial = torch.cat([radial, self.pair_emb(pair_type)], dim=-1)
        weights = self.radial_mlp(radial)
        return z * weights.index_select(1, self._radial_weight_index)


@register_transform(
    "equivariant_convolution",
    category="convolution",
    summary="Gated tensor product with edge harmonics, weighted by a radial MLP",
    inputs=[
        Port("features", "edge", "irreps"),
        Port("harmonics", "edge", "irreps"),
        Port("radial", "edge", "invariant"),
        Port(
            "pair_type",
            "edge",
            "index",
            optional=True,
            doc="Needed by pair-conditioned radial weights",
        ),
    ],
    options=[
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=infer_from(0),
            infer_doc="input irreps",
        ),
        Option(
            "tensor_product",
            "choice",
            "separate_weight",
            key=True,
            choices=("separate_weight", "fully_connected"),
        ),
        Option("gate", "bool", True, key=True),
        Option("radial_layers", "int_list", (64, 64), key=True),
        Option("radial_weight_granularity", "choice", "term", choices=("term", "copy")),
        Option(
            "pair_embedding_dim",
            "int",
            0,
            doc="Condition the radial MLP on a species-pair embedding",
        ),
        Option(
            "num_pairs",
            "int",
            infer=lambda inputs, mapper, options: (
                inputs[3].width if len(inputs) > 3 else None
            ),
            infer_doc="categories of the pair index",
            default=None,
        ),
    ],
)
def _equivariant_convolution(p, inputs, mapper):
    features = global_irreps(inputs[0], "edge")
    harmonics = global_irreps(inputs[1], "edge")
    radial = invariant_irreps(inputs[2], "edge", what="Radial features")
    same_support(*inputs)
    require_canonical(inputs[0], inputs[1])
    if p["pair_embedding_dim"]:
        if len(inputs) < 4:
            raise ValueError(
                "Pair-conditioned radial weights need the pair index input"
            )
        index_spec(inputs[3], "edge")
        if not p["num_pairs"]:
            raise ValueError("Pair-conditioned radial weights need num_pairs")
    module = EquivariantConvolution(
        radial.dim,
        features,
        harmonics,
        p["irreps_out"],
        tensor_product=p["tensor_product"],
        gate=p["gate"],
        radial_layers=p["radial_layers"],
        radial_weight_granularity=p["radial_weight_granularity"],
        num_pairs=p["num_pairs"],
        pair_embedding_dim=p["pair_embedding_dim"],
    )
    return module, [irrep_feature(inputs[0], module.irreps_out)]


class SelfConnection(nn.Module):
    """``update + TP(previous, attributes)``: a species-dependent skip path."""

    def __init__(self, irreps_previous, irreps_attributes, irreps_update):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_previous,
            irreps_attributes,
            irreps_update,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, previous, attributes, update):
        return update + self.tp(previous, attributes)


@register_transform(
    "self_connection",
    category="residual",
    bypass=2,
    summary="Add a species-dependent linear map of the previous features to an update",
    inputs=[
        Port("previous", None, "irreps"),
        Port("attributes", None, "invariant", doc="Species (pair) one-hot"),
        Port("update", None, "irreps"),
    ],
)
def _self_connection(p, inputs, mapper):
    if len({spec.domain for spec in inputs}) != 1:
        raise ValueError("Self-connection inputs must share their entities")
    same_support(*inputs)
    require_canonical(inputs[0], inputs[2])
    attributes = invariant_irreps(inputs[1], what="Self-connection attributes")
    module = SelfConnection(
        global_irreps(inputs[0]), attributes, global_irreps(inputs[2])
    )
    return module, [inputs[2]]


@register_transform(
    "irrep_nonlinearity",
    category="nonlinearity",
    summary="Equivariant activation: norm activation, S2 activation or gated magnitudes",
    inputs=[Port("features", None, "irreps")],
    options=[
        Option("nonlinearity", "choice", "normact", key=True, choices=NONLINEARITIES),
        Option(
            "scalar_activation",
            "choice",
            "leakyrelu",
            key=True,
            choices=SCALAR_ACTIVATIONS,
        ),
        Option("gate_activation", "choice", "softplus", choices=SCALAR_ACTIVATIONS),
        Option("norm", "choice", "component", choices=("component", "norm")),
        Option("s2_resolution", "int", 128),
    ],
)
def _irrep_nonlinearity(p, inputs, mapper):
    from net.activations import make_nonlinearity

    irreps = global_irreps(inputs[0])
    return make_nonlinearity(irreps, layer_settings(p)), [inputs[0]]


@register_transform(
    "layer_norm",
    category="normalization",
    summary="Per-entity normalization of each irrep with learned scale (and scalar bias)",
    inputs=[Port("features", None, "irreps")],
    options=[
        Option("eps", "float", 1e-5),
        Option("affine", "bool", True),
        Option("normalization", "choice", "component", choices=("component", "norm")),
    ],
)
def _layer_norm(p, inputs, mapper):
    from net.layer_norm import E3LayerNorm

    return E3LayerNorm(global_irreps(inputs[0]), **p), [inputs[0]]


@register_transform(
    "irrep_dropout",
    category="normalization",
    summary="Dropout of whole irrep copies (training only)",
    inputs=[Port("features", None, "irreps")],
    options=[Option("p", "float", 0.1, key=True)],
    pure=False,
)
def _irrep_dropout(p, inputs, mapper):
    return Dropout(global_irreps(inputs[0]), p=p["p"]), [inputs[0]]


class Residual(nn.Module):
    def __init__(self, project=None):
        super().__init__()
        self.project = project

    def forward(self, previous, update):
        if self.project is not None:
            previous = self.project(previous)
        return update + previous


class Passthrough(nn.Module):
    def forward(self, previous, update):
        return update


@register_transform(
    "residual",
    category="residual",
    summary="update + previous; projected or skipped when the layouts differ",
    inputs=[Port("previous", None, "irreps"), Port("update", None, "irreps")],
    bypass=1,
    options=[
        Option(
            "mode",
            "choice",
            "add_if_matching",
            key=True,
            choices=("add_if_matching", "add", "project", "linear"),
            doc="add_if_matching passes the update through when irreps differ; "
            "project maps previous linearly when they differ, linear always",
        ),
        Option("bias", "bool", False, doc="Bias of the linear map (even scalars)"),
    ],
)
def _residual(p, inputs, mapper):
    if inputs[0].domain != inputs[1].domain:
        raise ValueError("Residual expects old/new features on the same entities")
    same_support(*inputs)
    require_canonical(*inputs)
    previous, update = global_irreps(inputs[0]), global_irreps(inputs[1])
    if p["mode"] == "linear" or (p["mode"] == "project" and previous != update):
        return Residual(o3.Linear(previous, update, biases=p["bias"])), [inputs[1]]
    if previous == update:
        return Residual(), [inputs[1]]
    if p["mode"] == "add":
        raise ValueError(
            f"Residual addition needs equal irreps ({previous} vs {update})"
        )
    return Passthrough(), [inputs[1]]


class EdgeRoles(nn.Module):
    """0 onsite (i = j, L = 0), then shifted self-images, then off-site edges."""

    def __init__(self, separate_shifted_self=True):
        super().__init__()
        self.separate_shifted_self = bool(separate_shifted_self)

    def forward(self, edge_index, edge_shift):
        same_atom = edge_index[0] == edge_index[1]
        zero_shift = (edge_shift == 0).all(dim=0)
        onsite = same_atom & zero_shift
        if not self.separate_shifted_self:
            return torch.where(onsite, 0, 1)
        return torch.where(onsite, 0, torch.where(same_atom, 1, 2))


@register_transform(
    "edge_roles",
    category="routing",
    summary="Classify edges as onsite / shifted self-image / off-site (route index)",
    inputs=[
        Port("edge_index", "relation", "index"),
        Port("edge_shift", "relation", "index"),
    ],
    options=[Option("separate_shifted_self", "bool", True, key=True)],
)
def _edge_roles(p, inputs, mapper):
    index_spec(inputs[0], "relation")
    index_spec(inputs[1], "relation")
    support = same_support(*inputs)
    labels = (
        ("onsite", "shifted_self", "offsite")
        if p["separate_shifted_self"]
        else ("onsite", "offsite")
    )
    return EdgeRoles(**p), [
        FeatureSpec(
            "edge",
            "index",
            frame="none",
            width=len(labels),
            support=support,
            categories=labels,
        )
    ]


class OnsiteFromNodes(nn.Module):
    def __init__(self, onsite=0):
        super().__init__()
        self.onsite = int(onsite)

    def forward(self, edges, nodes, edge_index, roles):
        onsite = (roles == self.onsite).unsqueeze(-1)
        return torch.where(onsite, nodes.index_select(0, edge_index[0]), edges)


@register_transform(
    "onsite_from_nodes",
    category="routing",
    summary="Replace the features of onsite edges by the features of their atom",
    inputs=[
        Port("edges", "edge", "irreps"),
        Port("nodes", "node", "irreps"),
        Port("edge_index", "relation", "index"),
        Port("roles", "edge", "index"),
    ],
)
def _onsite_from_nodes(p, inputs, mapper):
    edges, nodes = global_irreps(inputs[0], "edge"), global_irreps(inputs[1], "node")
    if edges != nodes:
        raise ValueError(
            f"Onsite substitution needs equal node and edge irreps ({nodes} vs {edges}); "
            "insert irrep_linear"
        )
    index_spec(inputs[2], "relation")
    index_spec(inputs[3], "edge")
    same_support(inputs[0], inputs[2], inputs[3])
    require_canonical(inputs[0], inputs[1])
    labels = inputs[3].categories or ()
    return OnsiteFromNodes(labels.index("onsite") if "onsite" in labels else 0), [
        inputs[0]
    ]


def _sanitize(key):
    return key.replace("-", "_")


def shared_output_irreps(pair_irreps):
    """Smallest irreps holding every species pair's output (max multiplicity)."""
    counts, order = defaultdict(int), set()
    for irreps in pair_irreps.values():
        for mul, ir in irreps:
            counts[(ir.l, ir.p)] = max(counts[(ir.l, ir.p)], int(mul))
            order.add((ir.l, ir.p))
    parts = [f"{counts[(l, p)]}x{l}{'e' if p == 1 else 'o'}" for l, p in sorted(order)]
    return o3.Irreps("+".join(parts)).simplify() if parts else o3.Irreps("")


def pair_output_indices(shared, pair):
    """Columns of ``shared`` that hold each copy of ``pair``'s irreps."""
    ranges, cursor = {}, 0
    for mul, ir in shared:
        ranges[(ir.l, ir.p)] = list(range(cursor, cursor + mul * ir.dim))
        cursor += mul * ir.dim
    gather, offsets = [], defaultdict(int)
    for mul, ir in pair:
        start = offsets[(ir.l, ir.p)] * ir.dim
        gather.extend(ranges[(ir.l, ir.p)][start : start + mul * ir.dim])
        offsets[(ir.l, ir.p)] += mul
    return torch.tensor(gather, dtype=torch.long)


class PairProjection(nn.Module):
    """Per species-pair (and per route) projection to orbital-pair irreps.

    Returns ``{pair_key: vectors}``; the rows of each pair are the edges
    ``partitions[pair_key]["global_idx"]``, in that order.
    ``split`` builds one MLP per (route, pair); ``shared_conditioned`` one per
    route acting on ``[h, onehot(pair)]`` whose output is gathered per pair.
    Optional log-scale MLPs multiply each output by ``exp(s(h))``.
    """

    def __init__(
        self,
        irreps_in,
        pair_irreps,
        *,
        routes=1,
        mode="split",
        layers=1,
        cfg=None,
        route_scales=None,
        log_scale=False,
        log_scale_layers=1,
        log_scale_cfg=None,
    ):
        super().__init__()
        from net.common import E3MLP

        self.pair_keys = list(pair_irreps)
        self.pair_dims = {
            key: o3.Irreps(irreps).dim for key, irreps in pair_irreps.items()
        }
        self.mode, self.routes = mode, int(routes)
        route_scales = list(route_scales or [1.0] * self.routes)
        if len(route_scales) != self.routes:
            raise ValueError(f"route_scales needs one scale per route ({self.routes})")
        irreps_in = o3.Irreps(irreps_in)
        self.projections = nn.ModuleDict()
        self.shared = nn.ModuleList()
        self._indices = {}
        if mode == "split":
            for route in range(self.routes):
                for key in self.pair_keys:
                    self.projections[f"{route}_{_sanitize(key)}"] = E3MLP(
                        irreps_in,
                        irreps_in,
                        o3.Irreps(pair_irreps[key]),
                        layers,
                        cfg,
                        activate_last=False,
                        post_scale=route_scales[route],
                    )
        elif mode == "shared_conditioned":
            shared = shared_output_irreps(
                {k: o3.Irreps(v) for k, v in pair_irreps.items()}
            )
            conditioned = (
                irreps_in + o3.Irreps(f"{len(self.pair_keys)}x0e")
            ).simplify()
            for key in self.pair_keys:
                indices = pair_output_indices(shared, o3.Irreps(pair_irreps[key]))
                self.register_buffer(
                    f"index_{_sanitize(key)}", indices, persistent=False
                )
            for route in range(self.routes):
                self.shared.append(
                    E3MLP(
                        conditioned,
                        conditioned,
                        shared,
                        layers,
                        cfg,
                        activate_last=False,
                        post_scale=route_scales[route],
                    )
                )
        else:
            raise ValueError("mode must be 'split' or 'shared_conditioned'")
        self.log_scales = nn.ModuleDict()
        if log_scale:
            for route in range(self.routes):
                for key in self.pair_keys:
                    self.log_scales[f"{route}_{_sanitize(key)}"] = E3MLP(
                        irreps_in,
                        o3.Irreps("64x0e"),
                        o3.Irreps("1x0e"),
                        log_scale_layers,
                        log_scale_cfg or cfg,
                    )

    def _project(self, key, route, hidden):
        name = f"{route}_{_sanitize(key)}"
        if self.mode == "split":
            vectors = self.projections[name](hidden)
        else:
            condition = hidden.new_zeros(hidden.shape[0], len(self.pair_keys))
            condition[:, self.pair_keys.index(key)] = 1.0
            shared = self.shared[route](torch.cat([hidden, condition], dim=-1))
            vectors = shared.index_select(-1, getattr(self, f"index_{_sanitize(key)}"))
        if self.log_scales:
            vectors = torch.exp(self.log_scales[name](hidden)) * vectors
        return vectors

    def forward(self, hidden, partitions, route=None):
        present = [
            key
            for key in self.pair_keys
            if key in partitions and partitions[key]["global_idx"].numel()
        ]
        if self.routes == 1:
            return {
                key: self._project(
                    key, 0, hidden.index_select(0, partitions[key]["global_idx"])
                )
                for key in present
            }
        # Sort the rows once by (pair, route): one host synchronization per call.
        edges = hidden.shape[0]
        pair = torch.full(
            (edges,), len(present), dtype=torch.long, device=hidden.device
        )
        local = torch.zeros(edges, dtype=torch.long, device=hidden.device)
        for index, key in enumerate(present):
            rows = partitions[key]["global_idx"]
            pair[rows] = index
            local[rows] = torch.arange(rows.numel(), device=hidden.device)
        category = pair * self.routes + route
        counts = torch.bincount(
            category, minlength=(len(present) + 1) * self.routes
        ).tolist()
        order = torch.argsort(category, stable=True)
        result, start = {}, 0
        for index, key in enumerate(present):
            vectors = hidden.new_zeros(
                partitions[key]["global_idx"].numel(), self.pair_dims[key]
            )
            for category_index in range(self.routes):
                count = counts[index * self.routes + category_index]
                rows = order[start : start + count]
                start += count
                if count:
                    projected = self._project(
                        key, category_index, hidden.index_select(0, rows)
                    )
                    vectors.index_copy_(
                        0, local.index_select(0, rows), projected.to(vectors.dtype)
                    )
            result[key] = vectors
        return result

    def operations_per_entity(self):
        modules = [*self.projections.values(), *self.shared, *self.log_scales.values()]
        return _sum_operations(modules) / max(1, len(self.pair_keys) * self.routes)


def _route_scales(scales, spec, routes):
    if scales is None or isinstance(scales, (list, tuple)):
        return scales
    labels = (spec.categories or ()) if spec is not None else ()
    unknown = set(scales) - set(labels) - {str(i) for i in range(routes)}
    if unknown and not set(scales) & set(labels):
        raise ValueError(f"route_scales names unknown categories {sorted(unknown)}")
    return [
        (
            float(scales.get(labels[i], scales.get(str(i), 1.0)))
            if i < len(labels)
            else float(scales.get(str(i), 1.0))
        )
        for i in range(routes)
    ]


@register_transform(
    "pair_projection",
    category="readout",
    summary="Project edge features to the orbital-pair irreps of each species pair",
    inputs=[
        Port("features", "edge", "irreps"),
        Port(
            "partitions",
            "relation",
            "object",
            doc="Edges of each species pair (edge_partitions)",
        ),
        Port(
            "route",
            "edge",
            "index",
            optional=True,
            doc="Edge roles: separate weights per role",
        ),
    ],
    outputs=("block_vectors",),
    options=[
        Option(
            "mode", "choice", "split", key=True, choices=("split", "shared_conditioned")
        ),
        Option("layers", "int", 1, key=True),
        Option("variant", "choice", "normact", key=True, choices=MLP_VARIANTS),
        Option(
            "routes",
            "int",
            1,
            infer=lambda inputs, mapper, options: (
                inputs[2].width if len(inputs) > 2 else None
            ),
            infer_doc="categories of the route index",
        ),
        Option(
            "route_scales",
            "mapping",
            None,
            doc="Output scale per route: a list, or {category label: scale} (default 1)",
        ),
        Option("log_scale", "bool", False, doc="Multiply by exp(MLP(h)) per pair"),
        Option("log_scale_layers", "int", 1),
        Option("log_scale_variant", "choice", "film", choices=MLP_VARIANTS),
        *ACTIVATION_OPTIONS,
        *MLP_OPTIONS,
    ],
)
def _pair_projection(p, inputs, mapper):
    if mapper is None:
        raise ValueError("pair_projection requires the orbital mapper")
    irreps_in = global_irreps(inputs[0], "edge")
    require_canonical(inputs[0])
    if inputs[1].representation != "object":
        raise ValueError("pair_projection reads the species-pair edge partitions")
    same_support(*inputs)
    _check_route(inputs, p["routes"], position=2)
    pair_irreps = {key: mapper.get_pair_irreps(key) for key in mapper.edge_types}
    module = PairProjection(
        irreps_in,
        pair_irreps,
        routes=p["routes"],
        mode=p["mode"],
        layers=p["layers"],
        cfg=layer_settings(p, variant=p["variant"]),
        route_scales=_route_scales(
            p["route_scales"], inputs[2] if len(inputs) > 2 else None, p["routes"]
        ),
        log_scale=p["log_scale"],
        log_scale_layers=p["log_scale_layers"],
        log_scale_cfg=layer_settings(p, variant=p["log_scale_variant"]),
    )
    return module, [FeatureSpec("edge", "block_vectors", support=inputs[0].support)]
