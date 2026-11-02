"""A cookbook of new E3GNN layers, one for every kind of building block.

Importing this module registers twelve transformations with the MANDALA
pipeline compiler. Each registration consists of two parts:

* an ordinary ``torch.nn.Module`` holding the mathematics, and
* a *factory* ``(options, input_specs, mapper) -> (module, output_specs)``
  that validates the typed input contract and declares the typed output.

The factory is where a layer states which entities it reads (nodes, edges,
relations), which irreducible representations it accepts and produces, and on
which support (index set) its outputs live. The pipeline compiler refuses to
connect layers whose contracts disagree, so every model assembled from these
layers is checked before a single tensor is allocated.

Layer kinds and the transformation names registered here:

==========================  ================================
Kind                        Transformation
==========================  ================================
Radial basis                ``bessel_basis``
Node encoder                ``periodic_table_embedding``
Edge encoder                ``radial_angular_encoder``
Tensor-product convolution  ``depthwise_convolution``
Message aggregation         ``tempered_softmax_aggregation``
Equivariant nonlinearity    ``squash_activation``
Equivariant normalization   ``degree_rms_norm``
Equivariant MLP block       ``equivariant_glu``
Residual update             ``rezero_residual``
Edge update                 ``endpoint_bilinear_edge_update``
Operator readout            ``pair_mixture_readout``
Output envelope             ``learned_pair_envelope``
==========================  ================================
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter

from core.norms import zero_safe_norm
from net.pipeline import FeatureSpec, register_transform

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def global_irreps(spec: FeatureSpec, domain: str | None = None) -> o3.Irreps:
    """Return the irreps of a global-frame feature, enforcing its domain."""
    if spec.frame != "global" or spec.irreps is None:
        raise ValueError("Expected global-frame irrep features")
    if domain is not None and spec.domain != domain:
        raise ValueError(f"Expected {domain} features, received {spec.domain}")
    return o3.Irreps(spec.irreps)


def invariant_scalars(irreps: o3.Irreps) -> bool:
    return all(ir.l == 0 and ir.p == 1 for _, ir in irreps)


def require_index(spec: FeatureSpec, domain: str) -> None:
    if spec.domain != domain or spec.representation != "index":
        raise ValueError(f"Expected a {domain}-scoped integer index")


def require_same_support(*specs: FeatureSpec) -> str | None:
    supports = {spec.support for spec in specs}
    if len(supports) != 1:
        raise ValueError(
            f"Inputs live on different supports: {sorted(map(str, supports))}"
        )
    return supports.pop()


def copy_invariants(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """One rotation- and inversion-invariant number per irrep copy.

    Even scalars are invariant as they are. Odd scalars change sign under
    inversion, so their square is used. Every other copy contributes its norm.
    """
    values = []
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        block = x[..., sl].reshape(*x.shape[:-1], mul, ir.dim)
        if ir.l == 0 and ir.p == 1:
            values.append(block[..., 0])
        elif ir.l == 0:
            values.append(block[..., 0].square())
        else:
            values.append(zero_safe_norm(block, dim=-1))
    return torch.cat(values, dim=-1)


def expand_per_copy(values: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Broadcast one value per irrep copy to every magnetic component."""
    repeats = torch.tensor(
        [ir.dim for mul, ir in irreps for _ in range(mul)], device=values.device
    )
    return values.repeat_interleave(repeats, dim=-1)


# --------------------------------------------------------------------------
# 1. Radial basis: smooth Bessel functions with a polynomial cutoff
# --------------------------------------------------------------------------


class BesselBasis(nn.Module):
    r"""``phi_n(r) = sqrt(2/r_c) * sin(n pi r / r_c) / r * u(r / r_c)``.

    ``u`` is the C^2 polynomial envelope of order ``p``. The ``sin(x)/r`` factor
    is evaluated through ``sinc`` so that onsite edges (``r = 0``) are finite.
    """

    def __init__(self, cutoff, n_basis=8, envelope_exponent=6, trainable=False):
        super().__init__()
        self.cutoff = float(cutoff)
        self.p = int(envelope_exponent)
        frequencies = torch.arange(1, n_basis + 1, dtype=torch.get_default_dtype())
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)

    def envelope(self, x):
        p = self.p
        u = (
            1
            - (p + 1) * (p + 2) / 2 * x**p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2 * x ** (p + 2)
        )
        return torch.where(x < 1, u, torch.zeros_like(u))

    def forward(self, lengths):
        r = lengths.reshape(-1, 1)
        x = r / self.cutoff
        # sin(n pi x) / r == (n pi / r_c) * sinc(n x)
        radial = (
            self.frequencies * math.pi / self.cutoff * torch.sinc(self.frequencies * x)
        )
        return math.sqrt(2 / self.cutoff) * radial * self.envelope(x)


@register_transform("bessel_basis")
def _bessel_basis(options, inputs, mapper):
    (lengths,) = inputs
    if lengths.domain != "edge" or lengths.dim != 1:
        raise ValueError("bessel_basis expects one edge length per edge")
    module = BesselBasis(**options)
    width = len(module.frequencies)
    return module, [
        FeatureSpec("edge", irreps=f"{width}x0e", width=width, support=lengths.support)
    ]


# --------------------------------------------------------------------------
# 2. Node encoder: chemistry-aware embedding factorized over the periodic table
# --------------------------------------------------------------------------


def period_and_group(atomic_number: int) -> tuple[int, int]:
    """Period (1-7) and IUPAC group (1-18) of an element.

    Lanthanides and actinides are assigned to group 3.
    """
    period_ends = (2, 10, 18, 36, 54, 86, 118)
    period = next(i + 1 for i, end in enumerate(period_ends) if atomic_number <= end)
    position = atomic_number - (period_ends[period - 2] if period > 1 else 0)
    if period == 1:
        return period, 1 if position == 1 else 18
    if period in (2, 3):
        return period, position if position <= 2 else position + 10
    if period in (4, 5):
        return period, position
    if position <= 2:
        return period, position
    if position <= 17:
        return period, 3
    return period, position - 14


class PeriodicTableEmbedding(nn.Module):
    r"""``h_i = E_species[z_i] + E_period[period(z_i)] + E_group[group(z_i)]``.

    Elements sharing a row or column of the periodic table share part of their
    embedding, so chemically related species start from related features.
    """

    def __init__(self, elements, width):
        super().__init__()
        from ase.data import atomic_numbers

        rows = [period_and_group(atomic_numbers[symbol]) for symbol in elements]
        self.register_buffer("period", torch.tensor([row[0] - 1 for row in rows]))
        self.register_buffer("group", torch.tensor([row[1] - 1 for row in rows]))
        self.species = nn.Embedding(len(elements), width)
        self.periods = nn.Embedding(7, width)
        self.groups = nn.Embedding(18, width)
        for table in (self.species, self.periods, self.groups):
            nn.init.normal_(table.weight, std=0.2)

    def forward(self, species_index):
        return (
            self.species(species_index)
            + self.periods(self.period[species_index])
            + self.groups(self.group[species_index])
        )


@register_transform("periodic_table_embedding")
def _periodic_table_embedding(options, inputs, mapper):
    (species,) = inputs
    require_index(species, "node")
    module = PeriodicTableEmbedding(**options)
    width = options["width"]
    return module, [FeatureSpec("node", irreps=f"{width}x0e", width=width)]


# --------------------------------------------------------------------------
# 3. Edge encoder: radially modulated spherical harmonics
# --------------------------------------------------------------------------


class RadialAngularEncoder(nn.Module):
    r"""``e_ij^{(l,k)} = R_{l,k}(phi(r_ij)) Y_l(r_hat_ij)``.

    A radial MLP produces one invariant amplitude per output copy ``(l, k)``;
    each copy carries the spherical harmonic of its degree.
    """

    def __init__(self, radial_dim, angular_irreps, irreps_out, hidden=32):
        super().__init__()
        self.angular_irreps = o3.Irreps(angular_irreps)
        self.irreps_out = o3.Irreps(irreps_out)
        angular = {
            ir: sl
            for (_, ir), sl in zip(self.angular_irreps, self.angular_irreps.slices())
        }
        missing = [str(ir) for _, ir in self.irreps_out if ir not in angular]
        if missing:
            raise ValueError(f"Angular basis lacks output irreps {missing}")
        self.slices = [angular[ir] for _, ir in self.irreps_out]
        self.radial = nn.Sequential(
            nn.Linear(radial_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.irreps_out.num_irreps),
        )

    def forward(self, radial, angular):
        amplitudes = self.radial(radial)
        blocks, cursor = [], 0
        for (mul, ir), sl in zip(self.irreps_out, self.slices):
            weight = amplitudes[:, cursor : cursor + mul, None]
            blocks.append(
                (weight * angular[:, None, sl]).reshape(len(radial), mul * ir.dim)
            )
            cursor += mul
        return torch.cat(blocks, dim=-1)


@register_transform("radial_angular_encoder")
def _radial_angular_encoder(options, inputs, mapper):
    radial, angular = inputs
    radial_irreps = global_irreps(radial, "edge")
    if not invariant_scalars(radial_irreps):
        raise ValueError("Radial features must be invariant scalars")
    support = require_same_support(radial, angular)
    module = RadialAngularEncoder(
        radial_irreps.dim, global_irreps(angular, "edge"), **options
    )
    return module, [
        FeatureSpec("edge", "irreps", str(module.irreps_out), support=support)
    ]


# --------------------------------------------------------------------------
# 4. Tensor-product convolution: depthwise ("uvu") with radial path weights
# --------------------------------------------------------------------------


class DepthwiseConvolution(nn.Module):
    r"""``m_ij = sum_paths w_path(phi(r_ij)) * (h_j (x) Y(r_hat_ij))_path``.

    Every sender channel couples only with itself ("uvu" instructions), so the
    cost grows linearly with the multiplicity. The radial network emits one
    weight per channel and coupling path, i.e. a distance-dependent filter.
    """

    def __init__(self, node_irreps, angular_irreps, radial_dim, lmax=2, hidden=32):
        super().__init__()
        node_irreps, angular_irreps = o3.Irreps(node_irreps), o3.Irreps(angular_irreps)
        outputs, instructions = [], []
        for i, (mul, ir_in) in enumerate(node_irreps):
            for j, (_, ir_sh) in enumerate(angular_irreps):
                for ir_out in ir_in * ir_sh:
                    if ir_out.l <= lmax:
                        instructions.append((i, j, len(outputs), "uvu", True))
                        outputs.append((mul, ir_out))
        irreps_mid = o3.Irreps(outputs)
        irreps_out, permutation, _ = irreps_mid.sort()
        instructions = [
            (i, j, permutation[k], mode, train) for i, j, k, mode, train in instructions
        ]
        self.tp = o3.TensorProduct(
            node_irreps,
            angular_irreps,
            irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial = nn.Sequential(
            nn.Linear(radial_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.tp.weight_numel),
        )
        self.irreps_out = irreps_out

    def forward(self, node, angular, radial, edge_index):
        sender = edge_index[0]
        return self.tp(node[sender], angular, self.radial(radial))


@register_transform("depthwise_convolution")
def _depthwise_convolution(options, inputs, mapper):
    node, angular, radial, edge_index = inputs
    require_index(edge_index, "relation")
    support = require_same_support(angular, radial, edge_index)
    radial_irreps = global_irreps(radial, "edge")
    if not invariant_scalars(radial_irreps):
        raise ValueError("Radial features must be invariant scalars")
    module = DepthwiseConvolution(
        global_irreps(node, "node"),
        global_irreps(angular, "edge"),
        radial_irreps.dim,
        **options,
    )
    return module, [
        FeatureSpec("edge", "irreps", str(module.irreps_out), support=support)
    ]


# --------------------------------------------------------------------------
# 5. Aggregation: softmax attention with a learned temperature
# --------------------------------------------------------------------------


class TemperedSoftmaxAggregation(nn.Module):
    r"""``h_i = sum_j alpha_ij m_ij``, ``alpha_ij = softmax_j(s_ij / tau)``.

    Scores ``s_ij`` are computed from invariants only (radial features and the
    norm of each message copy), so the weights are rotation invariant and the
    weighted sum stays equivariant. ``tau`` is learned; ``tau -> infinity``
    recovers mean aggregation, ``tau -> 0`` selects the strongest neighbor.
    """

    def __init__(self, message_irreps, radial_dim, hidden=32, temperature=1.0):
        super().__init__()
        self.irreps = o3.Irreps(message_irreps)
        self.score = nn.Sequential(
            nn.Linear(radial_dim + self.irreps.num_irreps, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    def forward(self, messages, radial, edge_index, node_reference):
        receiver, count = edge_index[1], len(node_reference)
        features = torch.cat([radial, copy_invariants(messages, self.irreps)], dim=-1)
        scores = self.score(features).squeeze(-1) / self.log_temperature.exp()
        scores = (
            scores - scatter(scores, receiver, dim_size=count, reduce="max")[receiver]
        )
        weights = scores.exp()
        weights = weights / scatter(weights, receiver, dim_size=count)[
            receiver
        ].clamp_min(1e-12)
        return scatter(weights[:, None] * messages, receiver, dim=0, dim_size=count)


@register_transform("tempered_softmax_aggregation")
def _tempered_softmax_aggregation(options, inputs, mapper):
    messages, radial, edge_index, node_reference = inputs
    require_index(edge_index, "relation")
    require_same_support(messages, radial, edge_index)
    radial_irreps = global_irreps(radial, "edge")
    if not invariant_scalars(radial_irreps):
        raise ValueError("Attention scores must use invariant radial features")
    global_irreps(node_reference, "node")
    module = TemperedSoftmaxAggregation(
        global_irreps(messages, "edge"), radial_irreps.dim, **options
    )
    return module, [replace(messages, domain="node", support=node_reference.support)]


# --------------------------------------------------------------------------
# 6. Nonlinearity: capsule-style squash with a learned knee per copy
# --------------------------------------------------------------------------


class SquashActivation(nn.Module):
    r"""``v -> v |v| / (kappa^2 + |v|^2)`` for every copy except even scalars.

    The map rescales each copy by a function of its norm, so it commutes with
    rotations; it is odd in ``v``, so it also commutes with inversion for odd
    parity copies. Even scalars use SiLU. ``kappa`` is learned per copy.
    """

    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.raw_knee = nn.Parameter(torch.zeros(self.irreps.num_irreps))
        self.register_buffer(
            "even_scalar",
            torch.tensor(
                [ir.l == 0 and ir.p == 1 for mul, ir in self.irreps for _ in range(mul)]
            ),
        )

    def forward(self, x):
        norms = []
        for (mul, ir), sl in zip(self.irreps, self.irreps.slices()):
            block = x[..., sl].reshape(*x.shape[:-1], mul, ir.dim)
            norms.append(zero_safe_norm(block, dim=-1))
        norm = torch.cat(norms, dim=-1)
        knee = nn.functional.softplus(self.raw_knee) + 1e-3
        squash = norm / (knee.square() + norm.square())
        scale = expand_per_copy(squash, self.irreps)
        even = expand_per_copy(self.even_scalar.to(x.dtype), self.irreps).bool()
        return torch.where(even, nn.functional.silu(x), x * scale)


@register_transform("squash_activation")
def _squash_activation(options, inputs, mapper):
    (features,) = inputs
    module = SquashActivation(global_irreps(features), **options)
    return module, [features]


# --------------------------------------------------------------------------
# 7. Normalization: per-entity RMS normalization of each irrep term
# --------------------------------------------------------------------------


class DegreeRMSNorm(nn.Module):
    r"""``y = gamma_c x_c / sqrt(mean_{c in term} |x_c|^2 / d + eps) (+ beta_c)``.

    Each irrep term (all copies of one ``(l, p)``) of every entity is divided by
    its own root-mean-square magnitude. The statistic is invariant, so the
    normalization is equivariant; a bias is only added to even scalars.
    """

    def __init__(self, irreps, eps=1e-6):
        super().__init__()
        self.irreps, self.eps = o3.Irreps(irreps), float(eps)
        self.gain = nn.Parameter(torch.ones(self.irreps.num_irreps))
        scalars = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        self.bias = nn.Parameter(torch.zeros(scalars))

    def forward(self, x):
        blocks, bias_cursor = [], 0
        for (mul, ir), sl in zip(self.irreps, self.irreps.slices()):
            block = x[..., sl].reshape(*x.shape[:-1], mul, ir.dim)
            rms = block.square().mean(dim=(-1, -2), keepdim=True).add(self.eps).sqrt()
            blocks.append((block / rms).reshape(*x.shape[:-1], mul * ir.dim))
            if ir.l == 0 and ir.p == 1:
                blocks[-1] = blocks[-1] + self.bias[bias_cursor : bias_cursor + mul]
                bias_cursor += mul
        return torch.cat(blocks, dim=-1) * expand_per_copy(self.gain, self.irreps)


@register_transform("degree_rms_norm")
def _degree_rms_norm(options, inputs, mapper):
    (features,) = inputs
    return DegreeRMSNorm(global_irreps(features), **options), [features]


# --------------------------------------------------------------------------
# 8. Equivariant MLP block: gated linear unit with a residual connection
# --------------------------------------------------------------------------


class EquivariantGLU(nn.Module):
    r"""``y = x + Linear_v(x) * sigma(MLP(inv(x)))`` (residual when shapes match).

    The value path is an equivariant linear map. The gate path sees only
    invariants of ``x`` and outputs one gate per output copy, which multiplies
    whole copies and therefore preserves equivariance.
    """

    def __init__(self, irreps_in, irreps_out=None, hidden=32):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out or irreps_in)
        self.value = o3.Linear(self.irreps_in, self.irreps_out)
        self.gate = nn.Sequential(
            nn.Linear(self.irreps_in.num_irreps, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.irreps_out.num_irreps),
        )
        self.residual = self.irreps_in == self.irreps_out

    def forward(self, x):
        gates = torch.sigmoid(self.gate(copy_invariants(x, self.irreps_in)))
        y = self.value(x) * expand_per_copy(gates, self.irreps_out)
        return x + y if self.residual else y


@register_transform("equivariant_glu")
def _equivariant_glu(options, inputs, mapper):
    (features,) = inputs
    module = EquivariantGLU(global_irreps(features), **options)
    return module, [replace(features, irreps=str(module.irreps_out), width=None)]


# --------------------------------------------------------------------------
# 9. Residual update: ReZero-style learned skip per irrep copy
# --------------------------------------------------------------------------


class ReZeroResidual(nn.Module):
    r"""``h' = h + alpha_c * Delta h`` with ``alpha_c`` initialized at zero.

    At initialization the block is the identity map, so arbitrarily deep stacks
    start out well conditioned; each copy learns how much update to admit.
    """

    def __init__(self, irreps, initial=0.0):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.alpha = nn.Parameter(torch.full((self.irreps.num_irreps,), float(initial)))

    def forward(self, old, update):
        return old + expand_per_copy(self.alpha, self.irreps) * update


@register_transform("rezero_residual")
def _rezero_residual(options, inputs, mapper):
    old, update = inputs
    if old.domain != update.domain or global_irreps(old) != global_irreps(update):
        raise ValueError("rezero_residual requires identical feature layouts")
    require_same_support(old, update)
    return ReZeroResidual(global_irreps(old), **options), [old]


# --------------------------------------------------------------------------
# 10. Edge update: bilinear coupling of the two endpoint states
# --------------------------------------------------------------------------


class EndpointBilinearEdgeUpdate(nn.Module):
    r"""``e'_ij = W_skip e_ij + TP(h_i, h_j)``.

    The fully connected tensor product of both endpoint states provides
    angular momentum combinations (for example ``1o (x) 1o -> 1e``) that the
    edge geometry alone cannot express, so the output irreps may be richer
    than the input edge irreps.
    """

    def __init__(self, node_irreps, edge_irreps, irreps_out=None):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out or edge_irreps)
        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps, node_irreps, self.irreps_out
        )
        self.skip = o3.Linear(edge_irreps, self.irreps_out)

    def forward(self, node, edge, edge_index):
        return self.skip(edge) + self.tp(node[edge_index[0]], node[edge_index[1]])


@register_transform("endpoint_bilinear_edge_update")
def _endpoint_bilinear_edge_update(options, inputs, mapper):
    node, edge, edge_index = inputs
    require_index(edge_index, "relation")
    require_same_support(edge, edge_index)
    module = EndpointBilinearEdgeUpdate(
        global_irreps(node, "node"), global_irreps(edge, "edge"), **options
    )
    return module, [replace(edge, irreps=str(module.irreps_out), width=None)]


# --------------------------------------------------------------------------
# 11. Readout: low-rank mixture of shared maps, conditioned on the atom pair
# --------------------------------------------------------------------------


class PairMixtureReadout(nn.Module):
    r"""``x_ij = sum_k c_k(a_i, a_j) W_k e_ij``.

    ``K`` shared equivariant linear maps are blended by invariant coefficients
    that depend only on the ordered species pair. ``K = 1`` is a single shared
    readout; ``K = number of pairs`` recovers one readout per pair.
    """

    def __init__(self, edge_irreps, irreps_out, num_pairs, rank=4):
        super().__init__()
        self.maps = nn.ModuleList(
            o3.Linear(edge_irreps, irreps_out) for _ in range(rank)
        )
        self.coefficients = nn.Embedding(num_pairs, rank)
        nn.init.constant_(self.coefficients.weight, 1.0 / rank)

    def forward(self, edge, pair_index):
        stacked = torch.stack([linear(edge) for linear in self.maps], dim=1)
        return (self.coefficients(pair_index)[..., None] * stacked).sum(dim=1)


@register_transform("pair_mixture_readout")
def _pair_mixture_readout(options, inputs, mapper):
    edge, pair_index = inputs
    require_index(pair_index, "edge")
    support = require_same_support(edge, pair_index)
    options = dict(options)
    irreps_out = o3.Irreps(options.pop("irreps_out"))
    module = PairMixtureReadout(global_irreps(edge, "edge"), irreps_out, **options)
    return module, [FeatureSpec("edge", "irreps", str(irreps_out), support=support)]


# --------------------------------------------------------------------------
# 12. Output envelope: learned Slater-type radial prior per atom pair
# --------------------------------------------------------------------------


class LearnedPairEnvelope(nn.Module):
    r"""``x_ij <- A_p (1 + r_ij)^nu_p exp(-beta_p r_ij) x_ij`` with ``p = (a_i, a_j)``.

    One invariant factor scales every component of an edge, so equivariance is
    untouched. The network then predicts a slowly varying correction to a
    physically motivated exponential decay. ``A = 1, nu = beta = 0`` at start.
    """

    def __init__(self, num_pairs):
        super().__init__()
        self.log_amplitude = nn.Parameter(torch.zeros(num_pairs))
        self.power = nn.Parameter(torch.zeros(num_pairs))
        self.raw_decay = nn.Parameter(torch.full((num_pairs,), -6.0))

    def forward(self, values, lengths, pair_index):
        r = lengths.reshape(-1)
        decay = nn.functional.softplus(self.raw_decay)[pair_index]
        log_factor = (
            self.log_amplitude[pair_index]
            + self.power[pair_index] * torch.log1p(r)
            - decay * r
        )
        return values * log_factor.exp()[:, None]


@register_transform("learned_pair_envelope")
def _learned_pair_envelope(options, inputs, mapper):
    values, lengths, pair_index = inputs
    require_index(pair_index, "edge")
    if lengths.dim != 1:
        raise ValueError("learned_pair_envelope expects one length per edge")
    require_same_support(values, lengths, pair_index)
    global_irreps(values, "edge")
    return LearnedPairEnvelope(**options), [values]
