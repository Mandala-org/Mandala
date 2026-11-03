"""Elementary transformations: embeddings, linear maps, bases and decoders."""

from __future__ import annotations

from dataclasses import replace

import torch
from e3nn import o3
from torch import nn

from net.features import (
    FeatureSpec,
    global_irreps,
    index_spec,
    invariant_irreps,
    same_angular_basis,
    same_support,
)
from net.registry import Option, Port, TransformResult, infer_from, register_transform


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, indices):
        return torch.nn.functional.one_hot(indices, self.num_classes).to(
            torch.get_default_dtype()
        )


class Add(nn.Module):
    def forward(self, *values):
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result


class Concat(nn.Module):
    def forward(self, *values):
        return torch.cat(values, dim=-1)


class Harmonics(nn.Module):
    """Real spherical harmonics of the direction of each vector.

    Vectors of zero length (onsite edges) have zero non-scalar harmonics.
    """

    def __init__(self, irreps, normalization="component", permutation=(0, 1, 2)):
        super().__init__()
        self.permutation = tuple(permutation)
        self.sh = o3.SphericalHarmonics(
            irreps, normalize=False, normalization=normalization
        )

    def forward(self, vectors):
        from core.norms import zero_safe_norm

        if self.permutation != (0, 1, 2):
            vectors = vectors[:, self.permutation]
        directions = vectors / zero_safe_norm(vectors, keepdim=True).clamp_min(1e-12)
        return self.sh(directions)


class SoftOneHotBasis(nn.Module):
    """e3nn ``soft_one_hot_linspace`` radial basis on ``[start, end]``."""

    def __init__(
        self, number, end, start=0.0, basis="gaussian", cutoff=False, scale=1.0
    ):
        super().__init__()
        self.number, self.start, self.end = int(number), float(start), float(end)
        self.basis, self.cutoff, self.scale = basis, bool(cutoff), float(scale)

    def forward(self, lengths):
        from e3nn.math import soft_one_hot_linspace

        values = soft_one_hot_linspace(
            lengths.reshape(-1),
            start=self.start,
            end=self.end,
            number=self.number,
            basis=self.basis,
            cutoff=self.cutoff,
        )
        return values * self.scale if self.scale != 1.0 else values

    def operations_per_entity(self):
        return 4 * self.number


class LengthRatio(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, lengths, reference):
        return lengths.reshape(-1) / reference.reshape(-1).clamp_min(self.eps)


class CodecDecode(nn.Module):
    def __init__(self, codec, basis="yzx"):
        super().__init__()
        self.codec = codec
        from net.transforms.orbital import IrrepCartesianBasis

        self.coordinates = IrrepCartesianBasis(codec.irreps_out, basis, "yzx")

    def forward(self, x):
        return self.codec.decode(self.coordinates(x))


class LocalOrbitalOutput(nn.Module):
    """Decode scalar orbital entries and explicitly ordered SOC components.

    The original local-coordinate reader interleaves real/imaginary values per
    orbital in order uu, dd, ud, du. The common codec uses uu, ud, du, dd.
    """

    def __init__(
        self, packed_dim, spinful=False, component_order=("uu", "dd", "ud", "du")
    ):
        super().__init__()
        self.packed_dim = int(packed_dim)
        self.spinful = bool(spinful)
        if self.packed_dim <= 0:
            raise ValueError("packed_dim must be positive")
        if len(component_order) != 4 or set(component_order) != {
            "uu",
            "ud",
            "du",
            "dd",
        }:
            raise ValueError(
                "SOC component_order must be a permutation of uu, ud, du, dd"
            )
        self.component_indices = tuple(
            component_order.index(c) for c in ("uu", "ud", "du", "dd")
        )

    def forward(self, value):
        expected = self.packed_dim * (8 if self.spinful else 1)
        if value.ndim != 2 or value.shape[1] != expected:
            raise ValueError(f"Local orbital output expects shape (edges,{expected})")
        if not self.spinful:
            return value
        pairs = value.reshape(value.shape[0], self.packed_dim, 4, 2)
        complex_values = torch.complex(pairs[..., 0], pairs[..., 1])
        return complex_values[:, :, self.component_indices].transpose(1, 2)


def _scalar(spec, width):
    return FeatureSpec(
        spec.domain, irreps=f"{width}x0e", width=width, support=spec.support
    )


@register_transform(
    "embedding",
    category="embedding",
    summary="Learned vector per integer class (species or species pair)",
    inputs=[Port("index", ("node", "edge"), "index")],
    options=[
        Option(
            "num_embeddings",
            "int",
            key=True,
            infer=infer_from(0, "width"),
            infer_doc="categories of the index",
            doc="Number of classes",
        ),
        Option("width", "int", 32, key=True),
        Option("extra_embeddings", "int", 0, doc="Unused extra rows"),
        Option("padding_idx", "int", None),
    ],
)
def _embedding(p, inputs, mapper):
    if inputs[0].representation != "index":
        raise ValueError("embedding requires integer class indices")
    module = nn.Embedding(
        p["num_embeddings"] + p["extra_embeddings"],
        p["width"],
        padding_idx=p["padding_idx"],
    )
    return module, [_scalar(inputs[0], p["width"])]


@register_transform(
    "one_hot",
    category="embedding",
    summary="One-hot encoding of integer classes",
    inputs=[Port("index", ("node", "edge"), "index")],
    options=[
        Option(
            "num_classes",
            "int",
            key=True,
            infer=infer_from(0, "width"),
            infer_doc="categories of the index",
        )
    ],
)
def _one_hot(p, inputs, mapper):
    index_spec(inputs[0], inputs[0].domain)
    return OneHot(p["num_classes"]), [_scalar(inputs[0], p["num_classes"])]


def _linear_output(inputs, mapper, options):
    if options.get("degrees_of") is None:
        return inputs[0].irreps if inputs else None
    from net.transforms.cluster import angular_expansion_projection_irreps

    return str(
        angular_expansion_projection_irreps(
            global_irreps(inputs[0]), options["degrees_of"]
        )
    )


@register_transform(
    "irrep_linear",
    category="linear",
    summary="Equivariant linear map between irreps (mixes copies of the same irrep)",
    inputs=[Port("features", None, "irreps")],
    options=[
        Option(
            "degrees_of",
            "irreps",
            None,
            doc="Infer irreps_out as the input's degrees with the multiplicities "
            "they have in these irreps (projection before an angular expansion)",
        ),
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=_linear_output,
            infer_doc="input irreps, or the input degrees of degrees_of",
        ),
        Option("biases", "bool", False, doc="Learned bias on even scalars"),
    ],
)
def _irrep_linear(p, inputs, mapper):
    irreps = global_irreps(inputs[0])
    out = o3.Irreps(p["irreps_out"])
    if p["degrees_of"] is not None and str(out) != _linear_output(inputs, mapper, p):
        raise ValueError(
            "Give irreps_out or degrees_of, not conflicting values of both"
        )
    module = o3.Linear(irreps, out, biases=p["biases"])
    return module, [
        FeatureSpec(
            inputs[0].domain,
            "irreps",
            str(out),
            support=inputs[0].support,
            angular_basis=inputs[0].angular_basis,
        )
    ]


@register_transform(
    "bond_local_linear",
    category="linear",
    summary="Linear map of bond-frame local components",
    inputs=[Port("features", "edge", "local")],
    options=[Option("out_features", "int", key=True), Option("bias", "bool", True)],
)
def _bond_local_linear(p, inputs, mapper):
    if inputs[0].representation != "local_components" or inputs[0].frame != "bond":
        raise ValueError(
            "bond_local_linear requires explicitly rotated bond-frame components"
        )
    width = p["out_features"]
    return nn.Linear(inputs[0].dim, width, bias=p["bias"]), [
        FeatureSpec(
            inputs[0].domain,
            "local_components",
            frame="bond",
            width=width,
            support=inputs[0].support,
        )
    ]


@register_transform(
    "scalar_linear",
    category="linear",
    summary="Dense linear map of invariant channels",
    inputs=[Port("features", None, "invariant")],
    options=[Option("out_features", "int", key=True), Option("bias", "bool", True)],
)
def _scalar_linear(p, inputs, mapper):
    if inputs[0].frame != "global" or (
        inputs[0].irreps
        and any(ir.l or ir.p != 1 for _, ir in o3.Irreps(inputs[0].irreps))
    ):
        raise ValueError("scalar_linear accepts invariant scalar channels")
    width = p["out_features"]
    return nn.Linear(inputs[0].dim, width, bias=p["bias"]), [_scalar(inputs[0], width)]


class ScalarMLP(nn.Module):
    """Dense layers with a pointwise activation between them (and optionally after)."""

    def __init__(
        self, in_features, hidden, out_features, activation, activate_last, bias
    ):
        super().__init__()
        from net.transforms.composition import ScalarActivation

        widths = [in_features, *hidden, out_features]
        layers = []
        for index, (width_in, width_out) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(width_in, width_out, bias=bias))
            if index < len(widths) - 2 or activate_last:
                layers.append(ScalarActivation(activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, values):
        return self.layers(values)


class ConstantScale(nn.Module):
    def __init__(self, factor=1.0, divisor=1.0):
        super().__init__()
        if not divisor:
            raise ValueError("constant_scale divisor must be nonzero")
        self.factor, self.divisor = float(factor), float(divisor)

    def forward(self, values):
        if self.factor != 1.0:
            values = values * self.factor
        return values / self.divisor if self.divisor != 1.0 else values


@register_transform(
    "scalar_mlp",
    category="mlp",
    summary="Dense MLP of invariant channels (Linear, activation, ..., Linear)",
    inputs=[Port("features", None, "invariant")],
    options=[
        Option("out_features", "int", key=True),
        Option("hidden", "int_list", (128,), key=True, doc="Hidden layer widths"),
        Option(
            "activation",
            "choice",
            "silu",
            choices=(
                "identity",
                "sigmoid",
                "tanh",
                "softplus",
                "silu",
                "leakyrelu",
                "relu",
            ),
        ),
        Option("activate_last", "bool", False, doc="Activation after the last layer"),
        Option("bias", "bool", True),
    ],
)
def _scalar_mlp(p, inputs, mapper):
    from net.transforms.local_coordinates import _invariant_width

    width = _invariant_width(inputs[0], "scalar_mlp")
    module = ScalarMLP(
        width, list(p["hidden"]), **{k: p[k] for k in p if k != "hidden"}
    )
    out = p["out_features"]
    if inputs[0].representation == "local_components":
        return module, [replace(inputs[0], width=out)]
    return module, [_scalar(inputs[0], out)]


@register_transform(
    "constant_scale",
    category="combine",
    summary="values * factor / divisor (e.g. divide by the average neighbor count)",
    inputs=[Port("features")],
    options=[
        Option("factor", "float", 1.0, key=True),
        Option("divisor", "float", 1.0, key=True),
    ],
)
def _constant_scale(p, inputs, mapper):
    if inputs[0].representation in {"index", "object", "geometry"}:
        raise ValueError("constant_scale multiplies feature values")
    return ConstantScale(p["factor"], p["divisor"]), [inputs[0]]


def _family(spec):
    # Scalar features with irreps are irrep features with only 0e channels.
    if spec.representation == "scalar" and spec.irreps is not None:
        return "irreps"
    return spec.representation


def _bond_local(spec):
    return spec.representation == "local_components" and spec.frame == "bond"


def _combinable(inputs):
    if any(map(_bond_local, inputs)) and all(
        _bond_local(s) or s.invariant for s in inputs
    ):
        # Invariant scalars are the same numbers in every frame.
        if len({s.domain for s in inputs}) > 1:
            raise ValueError("Combined features must share their entities")
        same_support(*inputs)
        return None
    if not inputs or any(
        (s.domain, s.frame, _family(s))
        != (inputs[0].domain, inputs[0].frame, _family(inputs[0]))
        for s in inputs
    ):
        raise ValueError(
            "Combined features must share entity, frame and representation"
        )
    same_support(*inputs)
    return same_angular_basis(*inputs)


def _sum_layout(spec):
    family = _family(spec)
    irreps = str(o3.Irreps(spec.irreps)) if spec.irreps is not None else None
    return (spec.domain, spec.support, spec.frame, family, irreps, spec.dim)


@register_transform(
    "add",
    category="combine",
    summary="Sum of features with identical layouts",
    inputs=[Port("first"), Port("others", variadic=True)],
)
def _add(p, inputs, mapper):
    _combinable(inputs)
    if any(_sum_layout(s) != _sum_layout(inputs[0]) for s in inputs):
        raise ValueError("Addition requires identical feature layouts")
    return Add(), [inputs[0]]


@register_transform(
    "concat",
    category="combine",
    summary="Concatenate feature channels (irreps are appended in order)",
    inputs=[Port("first"), Port("others", variadic=True)],
    options=[
        Option(
            "simplify", "bool", False, doc="Merge adjacent equal irreps of the result"
        )
    ],
)
def _concat(p, inputs, mapper):
    basis = _combinable(inputs)
    if any(map(_bond_local, inputs)):
        if any(s.dim is None for s in inputs):
            raise ValueError("Concatenated bond-frame components need declared widths")
        return Concat(), [
            FeatureSpec(
                inputs[0].domain,
                "local_components",
                frame="bond",
                width=sum(s.dim for s in inputs),
                support=inputs[0].support,
            )
        ]
    irreps = None
    if all(s.irreps for s in inputs):
        irreps = sum((o3.Irreps(s.irreps) for s in inputs), o3.Irreps())
        irreps = str(irreps.simplify() if p["simplify"] else irreps)
    width = (
        sum(s.dim for s in inputs) if all(s.dim is not None for s in inputs) else None
    )
    representation = (
        "irreps"
        if any(spec.representation == "irreps" for spec in inputs)
        else inputs[0].representation
    )
    return Concat(), [
        FeatureSpec(
            inputs[0].domain,
            representation,
            irreps,
            inputs[0].frame,
            width,
            inputs[0].support,
            basis,
        )
    ]


def _distance_input(spec, name):
    if spec.domain != "edge" or spec.representation not in {"scalar", "geometry"}:
        raise ValueError(f"{name} requires edge-scoped distances")


@register_transform(
    "gaussian_basis",
    category="radial_basis",
    summary="Gaussians on a uniform grid with one-spacing width (SchNet form)",
    inputs=[Port("lengths", "edge", doc="Edge lengths (Angstrom)")],
    options=[
        Option("n_gaussians", "int", 50, key=True),
        Option("start", "float", 0.0),
        Option("stop", "float", 5.0, key=True),
        Option("centered", "bool", False),
        Option("trainable", "bool", False),
    ],
)
def _gaussian_basis(p, inputs, mapper):
    from net.transforms.equivariant import GaussianBasis

    _distance_input(inputs[0], "gaussian_basis")
    return GaussianBasis(**p), [_scalar(inputs[0], p["n_gaussians"])]


@register_transform(
    "length_ratio",
    category="geometry",
    summary="Edge lengths divided by a per-edge reference length (e.g. a pair radius r0)",
    inputs=[
        Port("lengths", "edge"),
        Port("reference", "edge", doc="Positive reference length"),
    ],
    options=[Option("eps", "float", 1e-12)],
)
def _length_ratio(p, inputs, mapper):
    for spec in inputs:
        _distance_input(spec, "length_ratio")
    same_support(*inputs)
    return LengthRatio(p["eps"]), [
        FeatureSpec("edge", width=1, support=inputs[0].support)
    ]


@register_transform(
    "soft_one_hot_basis",
    category="radial_basis",
    summary="e3nn soft one-hot radial basis (gaussian, cosine, smooth_finite, fourier, bessel)",
    inputs=[Port("lengths", "edge", doc="Edge lengths (Angstrom)")],
    options=[
        Option("number", "int", 16, key=True, doc="Basis size"),
        Option("end", "float", key=True, doc="Upper end of the grid (Angstrom)"),
        Option("start", "float", 0.0),
        Option(
            "basis",
            "choice",
            "gaussian",
            key=True,
            choices=("gaussian", "cosine", "smooth_finite", "fourier", "bessel"),
        ),
        Option("cutoff", "bool", False, doc="Vanish smoothly at both ends"),
        Option("scale", "float", 1.0, doc="Constant factor (sqrt(number) normalizes)"),
    ],
)
def _soft_one_hot_basis(p, inputs, mapper):
    _distance_input(inputs[0], "soft_one_hot_basis")
    return SoftOneHotBasis(**p), [_scalar(inputs[0], p["number"])]


@register_transform(
    "spherical_harmonics",
    category="angular_basis",
    summary="Real spherical harmonics of edge directions",
    inputs=[Port("vectors", "edge", "geometry")],
    options=[
        Option("irreps", "irreps", key=True, doc="e.g. 1x0e+1x1o+1x2e"),
        Option(
            "normalization",
            "choice",
            "component",
            choices=("component", "norm", "integral"),
        ),
        Option(
            "permutation",
            "int_list",
            (0, 1, 2),
            doc="Evaluate in permuted Cartesian axes; the output is labelled with that basis",
        ),
    ],
)
def _spherical_harmonics(p, inputs, mapper):
    if inputs[0].representation != "geometry" or inputs[0].frame != "global":
        raise ValueError("spherical_harmonics requires global Cartesian vectors")
    if sorted(p["permutation"]) != [0, 1, 2]:
        raise ValueError(
            "Harmonic permutation must contain each Cartesian axis exactly once"
        )
    irreps = o3.Irreps(p["irreps"])
    source = inputs[0].angular_basis or "xyz"
    basis = "".join(source[index] for index in p["permutation"])
    return Harmonics(irreps, p["normalization"], p["permutation"]), [
        FeatureSpec(
            inputs[0].domain,
            "irreps",
            str(irreps),
            support=inputs[0].support,
            angular_basis=basis,
        )
    ]


@register_transform(
    "graph_irrep_norm",
    category="normalization",
    summary="Per-graph normalization of each irrep (DeepH-E3 / MACE style)",
    inputs=[Port("features", None, "irreps"), Port("batch", None, "index")],
    options=[
        Option("eps", "float", 1e-5),
        Option("affine", "bool", True),
        Option("normalization", "choice", "component", choices=("component", "norm")),
        Option("subtract_mean", "bool", True),
        Option("divide_norm", "bool", False),
    ],
)
def _graph_irrep_norm(p, inputs, mapper):
    from net.transforms.equivariant import GraphIrrepNorm

    index_spec(inputs[1], inputs[0].domain)
    same_support(*inputs)
    return GraphIrrepNorm(global_irreps(inputs[0]), **p), [inputs[0]]


def _codec_irreps(inputs, mapper, options):
    from core.operator_layout import ShellTensorCodec

    if "shell_pairs" not in options:
        return None
    codec = ShellTensorCodec(
        options["shell_pairs"],
        spinful=options.get("spinful", False),
        ignore_parity=options.get("ignore_parity", False),
        sort_irreps=options.get("sort_irreps", False),
    )
    return str(codec.irreps_out)


@register_transform(
    "shell_tensor_decode",
    category="output",
    summary="Decode coupled irreps into packed shell-pair orbital blocks",
    inputs=[Port("coupled", "edge", "irreps")],
    options=[
        Option("shell_pairs", "list", key=True, doc="Usually ${layout.shell_pairs}"),
        Option("spinful", "bool", False),
        Option("ignore_parity", "bool", False),
        Option("sort_irreps", "bool", False),
    ],
)
def _shell_tensor_decode(p, inputs, mapper):
    from core.operator_layout import ShellTensorCodec

    codec = ShellTensorCodec(**p, dtype=torch.get_default_dtype())
    received = global_irreps(inputs[0], "edge")
    if received != codec.irreps_out:
        raise ValueError(
            f"Readout irreps must exactly match the shell-channel codec: "
            f"expected {codec.irreps_out}, received {received}"
        )
    basis = same_angular_basis(inputs[0])
    return CodecDecode(codec, basis or "xyz"), [
        FeatureSpec(
            "edge",
            "matrix",
            frame="global",
            width=codec.packed_dim,
            support=inputs[0].support,
        )
    ]


@register_transform(
    "local_orbital_output",
    category="output",
    summary="Decode bond-frame local components into orbital blocks",
    inputs=[Port("components", "edge", "local")],
    options=[
        Option("packed_dim", "int", key=True),
        Option("spinful", "bool", False),
        Option("component_order", "list", ("uu", "dd", "ud", "du")),
    ],
)
def _local_orbital_output(p, inputs, mapper):
    if (
        inputs[0].domain != "edge"
        or inputs[0].representation != "local_components"
        or inputs[0].frame != "bond"
    ):
        raise ValueError(
            "local_orbital_output requires bond-frame local edge components"
        )
    module = LocalOrbitalOutput(**p)
    expected = module.packed_dim * (8 if module.spinful else 1)
    if inputs[0].dim is not None and inputs[0].dim != expected:
        raise ValueError("Local orbital channel width disagrees with packed output")
    return module, [
        FeatureSpec(
            "edge",
            "matrix",
            frame="bond",
            width=module.packed_dim,
            support=inputs[0].support,
        )
    ]


@register_transform(
    "species_irrep_affine",
    category="statistics",
    summary="Per species-pair irrep mean/scale fitted on the training set",
    inputs=[Port("coupled", "edge", "irreps"), Port("pair_type", "edge", "index")],
    options=[
        Option("num_species", "int", key=True, doc="Number of species pairs"),
        Option("gradient_mode", "choice", "physical", choices=("physical", "identity")),
        Option("eps", "float", 1e-12),
        Option("target", "str", "hamiltonian"),
        Option("statistics_basis", "str", "yzx"),
    ],
)
def _species_irrep_affine(p, inputs, mapper):
    from net.training_policies import SpeciesIrrepAffine

    index_spec(inputs[1], "edge")
    same_support(*inputs)
    module = SpeciesIrrepAffine(
        irreps=global_irreps(inputs[0], "edge"),
        feature_basis=inputs[0].angular_basis,
        **p,
    )
    return module, [inputs[0]]


__all__ = [
    "Add",
    "CodecDecode",
    "Concat",
    "Harmonics",
    "LocalOrbitalOutput",
    "OneHot",
    "SoftOneHotBasis",
    "TransformResult",
    "invariant_irreps",
]
