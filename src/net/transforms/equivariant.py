"""Composable equivariant transformations with reference parameter coordinates.

DeepH-E3's endpoint node and edge updates are compositions of these stages
with the generic ones (see ``configs/reference/deeph_e3.yaml``). The tensor
products, normalization and gate construction below are adapted from
DeepH-E3, commit ``036bf95b996052ef778c5c38e3c54c2ac0e2b591``:
https://github.com/Xiaoxun-Gong/DeepH-E3
The Gaussian basis originates in SchNetPack.  The equations, parameter creation
order, tensor-product instructions and default initialization are retained so
that mapped weights and training gradients can be compared directly.  These
are transformations, not independent model architectures.

Copyright (c) 2023 Xiaoxun-Gong
Copyright (c) 2018 Kristof Schütt, Michael Gastegger, Pan Kessel, Kim Nicoli
Copyright (c) 2018, the respective SchNetPack contributors

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from e3nn.nn import Extract, Gate
from e3nn.o3 import Irrep, Irreps, TensorProduct

from net.registry import Option, Port, infer_from, register_transform


def _tensor_product_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    target = Irrep(ir_out)
    return any(
        target in ir1 * ir2
        for _, ir1 in Irreps(irreps_in1).simplify()
        for _, ir2 in Irreps(irreps_in2).simplify()
    )


def make_tensor_product_gate(
    irreps_in1,
    irreps_in2,
    irreps_out,
    act: Mapping[int, Callable] | None = None,
    act_gates: Mapping[int, Callable] | None = None,
) -> Gate:
    """Select reachable scalars/tensors and construct parity-preserving gates.

    The returned gate's ``irreps_in`` describes the tensor-product output,
    including extra scalar gate channels.  Its ``irreps_out`` can be narrower
    than the requested irreps when a tensor-product path does not exist.
    """
    if act is None:
        act = {1: F.silu, -1: torch.tanh}
    if act_gates is None:
        act_gates = {1: torch.sigmoid, -1: torch.tanh}
    requested = Irreps(irreps_out)
    scalars = Irreps(
        [
            (mul, ir)
            for mul, ir in requested
            if ir.l == 0 and _tensor_product_path_exists(irreps_in1, irreps_in2, ir)
        ]
    ).simplify()
    tensors = Irreps(
        [
            (mul, ir)
            for mul, ir in requested
            if ir.l > 0 and _tensor_product_path_exists(irreps_in1, irreps_in2, ir)
        ]
    ).simplify()
    gate_irrep = None
    if tensors.dim:
        if _tensor_product_path_exists(irreps_in1, irreps_in2, "0e"):
            gate_irrep = "0e"
        elif _tensor_product_path_exists(irreps_in1, irreps_in2, "0o"):
            gate_irrep = "0o"
            warnings.warn("Using odd representations as gates", stacklevel=2)
        else:
            raise ValueError(
                f"{irreps_in1} times {irreps_in2} cannot produce scalar gates "
                f"for {tensors}."
            )
    gates = Irreps([(mul, gate_irrep) for mul, _ in tensors]).simplify()
    return Gate(
        scalars,
        [act[ir.p] for _, ir in scalars],
        gates,
        [act_gates[ir.p] for _, ir in gates],
        tensors,
    )


class GaussianBasis(nn.Module):
    """Unnormalized Gaussian features with one-grid-spacing standard deviation.

    ``distances`` has arbitrary leading shape; the feature axis is appended.
    Parameters use the current default dtype, as in the reference.  There is
    no cutoff envelope or implicit ``sqrt(n_gaussians)`` normalization.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        n_gaussians: int = 50,
        centered: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        if n_gaussians < 2:
            raise ValueError("GaussianBasis requires at least two centers.")
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.Tensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered
        self.out_dim = n_gaussians

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if self.centered:
            coefficient = -0.5 / self.offsets.pow(2)
            difference = distances[..., None]
        else:
            coefficient = -0.5 / self.width.pow(2)
            difference = distances[..., None] - self.offsets
        return torch.exp(coefficient * difference.pow(2))


class FactorizedTensorProduct(nn.Module):
    """CG tensor product with two learned channel factors per allowed path.

    Each path has weights ``W1[u,w] * W2[v,w]`` rather than an independent
    dense ``W[u,v,w]``.  Keeping both factors is essential for optimizer and
    initialization equivalence.  All allowed paths follow input1, input2,
    output order and use e3nn's ``uvw`` connection convention.
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        super().__init__()
        if kwargs.pop("internal_weights", False):
            raise ValueError("FactorizedTensorProduct supplies its own weights.")
        if not kwargs.pop("shared_weights", True):
            raise ValueError("FactorizedTensorProduct requires shared weights.")
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        instructions, weights1, weights2 = [], [], []
        for i1, (mul1, ir1) in enumerate(self.irreps_in1):
            for i2, (mul2, ir2) in enumerate(self.irreps_in2):
                for iout, (mulout, irout) in enumerate(self.irreps_out):
                    if irout in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mulout)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mulout)))
                        instructions.append((i1, i2, iout, "uvw", True, 1.0))
        self.tp = TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        factors = [
            (w1[:, None, :] * w2[None, :, :]).reshape(-1)
            for w1, w2 in zip(self.weights1, self.weights2)
        ]
        weights = torch.cat(factors) if factors else x1.new_empty(0)
        return self.tp(x1, x2, weights)


class SymmetricFactorizedTensorProduct(nn.Module):
    """Self tensor product using only input-irrep pairs ``i1 <= i2``."""

    def __init__(self, irreps_in, irreps_out, **kwargs):
        super().__init__()
        if kwargs.pop("internal_weights", False):
            raise ValueError("SymmetricFactorizedTensorProduct supplies weights.")
        if not kwargs.pop("shared_weights", True):
            raise ValueError(
                "SymmetricFactorizedTensorProduct requires shared weights."
            )
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        instructions, weights1, weights2 = [], [], []
        for i1, (mul1, ir1) in enumerate(self.irreps_in):
            for i2 in range(i1, len(self.irreps_in)):
                mul2, ir2 = self.irreps_in[i2]
                for iout, (mulout, irout) in enumerate(self.irreps_out):
                    if irout in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mulout)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mulout)))
                        instructions.append((i1, i2, iout, "uvw", True, 1.0))
        self.tp = TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factors = [
            (w1[:, None, :] * w2[None, :, :]).reshape(-1)
            for w1, w2 in zip(self.weights1, self.weights2)
        ]
        weights = torch.cat(factors) if factors else x.new_empty(0)
        return self.tp(x, x, weights)


class GraphIrrepNorm(nn.Module):
    """Normalize each irrep using rows and channels within each graph.

    ``x`` is ``[rows, irreps_in.dim]`` and ``batch`` assigns every row to a
    graph.  With the reference defaults, means are subtracted for all irreps,
    but variance normalization is applied only to ``l=0``.  The denominator
    is ``sqrt(variance) + eps``.  Only even scalars receive learned biases.
    This is graph normalization, not independent normalization of each atom.
    """

    def __init__(
        self,
        irreps_in,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        subtract_mean: bool = True,
        divide_norm: bool = False,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = self.irreps_in
        self.eps = eps
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        if normalization not in {"component", "norm"}:
            raise ValueError("normalization must be 'component' or 'norm'.")
        self.normalization = normalization
        if affine:
            ib, iw = 0, 0
            self.bias_slices, self.weight_slices = [], []
            for mul, ir in self.irreps_in:
                if ir.is_scalar():
                    self.bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    self.bias_slices.append(None)
                self.weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones(iw))
            self.bias = nn.Parameter(torch.zeros(ib))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1)
            if self.bias is not None:
                self.bias.fill_(0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if not batch.numel():
            return x
        batch_size = int(batch.max()) + 1
        counts = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch))
        counts = counts.clamp_min(1).to(dtype=x.dtype)
        result = []
        offset = 0
        for index, (mul, ir) in enumerate(self.irreps_in):
            field = x[:, offset : offset + mul * ir.dim].reshape(-1, mul, ir.dim)
            if self.subtract_mean or ir.l == 0:
                mean = (
                    scatter(
                        field, batch, dim=0, dim_size=batch_size, reduce="add"
                    ).mean(dim=1, keepdim=True)
                    / counts[:, None, None]
                )
                field = field - mean[batch]
            if self.divide_norm or ir.l == 0:
                variance = scatter(
                    field.abs().pow(2),
                    batch,
                    dim=0,
                    dim_size=batch_size,
                    reduce="mean",
                ).mean(dim=[1, 2], keepdim=True)
                if self.normalization == "norm":
                    variance = variance * ir.dim
                # Singleton graphs and absent irrep channels have exact zero
                # variance. Evaluate that inactive square-root branch at one:
                # masking sqrt(0) afterwards still produces NaN derivatives.
                zero = variance == 0
                std = torch.where(zero, torch.ones_like(variance), variance).sqrt()
                std = torch.where(zero, torch.zeros_like(std), std)
                field = field / (std[batch] + self.eps)
            if self.weight is not None:
                field = field * self.weight[self.weight_slices[index]][None, :, None]
            if self.bias is not None and ir.is_scalar():
                field = field + self.bias[self.bias_slices[index]][None, :, None]
            result.append(field.reshape(-1, mul * ir.dim))
            offset += mul * ir.dim
        return torch.cat(result, dim=-1)


class RadialChannelScale(nn.Module):
    """Multiply every irrep copy by an invariant weight from a radial MLP.

    The MLP maps the radial features of each row to one weight per irrep copy
    (``copy``) or per irrep term (``term``); the weights scale whole irreps,
    so the operation is equivariant.
    """

    def __init__(
        self,
        irreps,
        radial_dim,
        hidden: Sequence[int] = (64, 64),
        granularity: str = "copy",
    ):
        super().__init__()
        self.irreps = Irreps(irreps)
        if granularity not in {"copy", "term"}:
            raise ValueError("granularity must be 'copy' or 'term'")
        outputs = self.irreps.num_irreps if granularity == "copy" else len(self.irreps)
        layers = []
        widths = [radial_dim, *hidden, outputs]
        for index, (width_in, width_out) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(width_in, width_out))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers)
        index, offset = [], 0
        for term, (mul, ir) in enumerate(self.irreps):
            for copy in range(mul):
                index.extend(
                    [offset + copy if granularity == "copy" else term] * ir.dim
                )
            offset += mul
        self.register_buffer(
            "_weight_index", torch.tensor(index, dtype=torch.long), persistent=False
        )

    def forward(self, values, radial):
        return values * self.fc(radial).index_select(1, self._weight_index)


class GatedFactorizedTensorProduct(nn.Module):
    """Factorized tensor product, optionally followed by its e3nn gate.

    With ``gate`` the product also emits the scalar gates of the requested
    non-scalar irreps; only irreps reachable by the product are produced.
    With a single input the product is the symmetric square ``x (x) x``.
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, gate=False):
        super().__init__()
        irreps_in1 = Irreps(irreps_in1)
        right = irreps_in1 if irreps_in2 is None else Irreps(irreps_in2)
        requested = Irreps(irreps_out)
        self.nonlin = None
        if gate:
            self.nonlin = make_tensor_product_gate(irreps_in1, right, requested)
            product_out = self.nonlin.irreps_in
            self.irreps_out = self.nonlin.irreps_out
        else:
            product_out = Irreps(
                [
                    (mul, ir)
                    for mul, ir in requested
                    if _tensor_product_path_exists(irreps_in1, right, ir)
                ]
            )
            self.irreps_out = product_out
        if irreps_in2 is None:
            self.tp = SymmetricFactorizedTensorProduct(irreps_in1, product_out)
        else:
            self.tp = FactorizedTensorProduct(irreps_in1, right, product_out)

    def forward(self, left, right=None):
        value = self.tp(left) if right is None else self.tp(left, right)
        return value if self.nonlin is None else self.nonlin(value)


class IrrepPermutation(nn.Module):
    """Sort irrep fields without mixing or rescaling their coordinates."""

    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        sorted_irreps = self.irreps_in.sort()
        self.extr = Extract(
            self.irreps_in,
            [((mul, ir),) for mul, ir in sorted_irreps.irreps],
            [(i,) for i in sorted_irreps.inv],
        )
        self.extr_inv = Extract(
            sorted_irreps.irreps,
            [((mul, ir),) for mul, ir in self.irreps_in],
            [(i,) for i in sorted_irreps.p],
        )
        self.irreps_out = sorted_irreps.irreps.simplify()

    def forward(self, x):
        return torch.cat(self.extr(x), dim=-1)

    def inverse(self, x):
        return torch.cat(self.extr_inv(x), dim=-1)


def _irrep_spec(source, irreps, basis=None):
    from net.features import FeatureSpec

    irreps = Irreps(irreps)
    return FeatureSpec(
        source.domain,
        "irreps",
        str(irreps),
        support=source.support,
        angular_basis=(basis or source.angular_basis) if irreps.lmax > 0 else None,
    )


@register_transform(
    "factorized_tensor_product",
    category="product",
    summary="Tensor product with factorized path weights W1[u,w] W2[v,w], optionally gated",
    inputs=[
        Port("left", None, "irreps"),
        Port("right", None, "irreps", optional=True, doc="Omit for x (x) x"),
    ],
    options=[
        Option(
            "irreps_out",
            "irreps",
            key=True,
            infer=infer_from(0),
            infer_doc="left irreps",
        ),
        Option(
            "gate",
            "bool",
            False,
            key=True,
            doc="Emit gate scalars and apply the gate (silu/tanh, sigmoid/tanh)",
        ),
    ],
)
def _factorized_tensor_product(p, inputs, mapper):
    from net.features import global_irreps, same_angular_basis, same_support

    if len(inputs) == 2 and inputs[0].domain != inputs[1].domain:
        raise ValueError("Tensor product operands must share their entities")
    same_support(*inputs)
    basis = same_angular_basis(*inputs)
    right = global_irreps(inputs[1]) if len(inputs) == 2 else None
    module = GatedFactorizedTensorProduct(
        global_irreps(inputs[0]), right, p["irreps_out"], p["gate"]
    )
    return module, [_irrep_spec(inputs[0], module.irreps_out, basis)]


@register_transform(
    "radial_channel_scale",
    category="convolution",
    summary="Scale every irrep copy by a weight from an MLP of radial features",
    inputs=[Port("features", None, "irreps"), Port("radial", None, "invariant")],
    options=[
        Option(
            "hidden", "int_list", (64, 64), key=True, doc="Radial MLP hidden widths"
        ),
        Option("granularity", "choice", "copy", choices=("copy", "term")),
    ],
)
def _radial_channel_scale(p, inputs, mapper):
    from net.features import global_irreps, invariant_irreps, same_support

    if inputs[0].domain != inputs[1].domain:
        raise ValueError("Radial weights must belong to the scaled entities")
    same_support(*inputs)
    radial = invariant_irreps(inputs[1], what="Radial features")
    module = RadialChannelScale(
        global_irreps(inputs[0]), radial.dim, p["hidden"], p["granularity"]
    )
    return module, [inputs[0]]


@register_transform(
    "irrep_sort",
    category="combine",
    summary="Sort the irrep terms by degree and parity (no mixing or rescaling)",
    inputs=[Port("features", None, "irreps")],
)
def _irrep_sort(p, inputs, mapper):
    from net.features import global_irreps

    module = IrrepPermutation(global_irreps(inputs[0]))
    return module, [_irrep_spec(inputs[0], module.irreps_out)]
