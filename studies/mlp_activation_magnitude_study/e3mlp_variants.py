"""
Equivariant MLP variants for irreps-to-irreps mappings.

This module implements a set of E3MLP families discussed in:
`docs/irreps_mlp_generalization_ideas.md`.

Implemented variants:
- 2: NormActE3MLP
- 3: GateE3MLP
- 4: GateScalarsMLPE3MLP
- 5: GateMagnitudesE3MLP
- 7: ResidualE3MLP
- 9: BilinearSelfTPE3MLP
- 11: InvariantSelfAttentionE3MLP
- 13: InvariantMoEE3MLP
- custom: ScalarMagnitudeSelfTPGatedE3MLP

All variants support `num_layers=1` as an exact `e3nn.o3.Linear`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from e3nn.o3 import Irrep, Irreps, Linear, FullyConnectedTensorProduct
from e3nn.nn import Gate, NormActivation

from net.activations import GateMagnitudes, GateScalarsMLP, scalar_activation


ActivationKind = Literal["normact", "gate", "gate_scalars_mlp", "gate_magnitudes"]


def _as_irreps(irreps: Irreps | str) -> Irreps:
    return irreps if isinstance(irreps, Irreps) else Irreps(irreps)


def _callable_activation(name: str, negative_slope: float = 0.01) -> Callable:
    name = name.lower()
    if name == "silu":
        return F.silu
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "softplus":
        return F.softplus
    if name == "leakyrelu":
        return lambda x: F.leaky_relu(x, negative_slope=negative_slope)
    raise ValueError(f"Unsupported callable activation '{name}'")


def _build_scalar_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int],
    act_name: str,
) -> nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(scalar_activation(act_name))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class _Chunk:
    mul: int
    ir: Irrep
    start: int
    size: int


def _chunks(irreps: Irreps) -> list[_Chunk]:
    out: list[_Chunk] = []
    start = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        out.append(_Chunk(mul=mul, ir=ir, start=start, size=size))
        start += size
    return out


def _split_scalar_nonscalar_irreps(irreps: Irreps) -> tuple[Irreps, Irreps]:
    scalar = [(mul, ir) for mul, ir in irreps if ir.l == 0]
    nonscalar = [(mul, ir) for mul, ir in irreps if ir.l > 0]
    return Irreps(scalar), Irreps(nonscalar)


def _split_scalar_nonscalar_features(
    x: torch.Tensor,
    irreps: Irreps,
) -> tuple[torch.Tensor, Irreps, torch.Tensor, Irreps]:
    scalar_chunks: list[torch.Tensor] = []
    nonscalar_chunks: list[torch.Tensor] = []
    scalar_irreps: list[tuple[int, Irrep]] = []
    nonscalar_irreps: list[tuple[int, Irrep]] = []

    for c in _chunks(irreps):
        chunk = x[..., c.start : c.start + c.size]
        if c.ir.l == 0:
            scalar_chunks.append(chunk)
            scalar_irreps.append((c.mul, c.ir))
        else:
            nonscalar_chunks.append(chunk)
            nonscalar_irreps.append((c.mul, c.ir))

    x_scalar = (
        torch.cat(scalar_chunks, dim=-1)
        if scalar_chunks
        else x.new_zeros(*x.shape[:-1], 0)
    )
    x_nonscalar = (
        torch.cat(nonscalar_chunks, dim=-1)
        if nonscalar_chunks
        else x.new_zeros(*x.shape[:-1], 0)
    )
    return x_scalar, Irreps(scalar_irreps), x_nonscalar, Irreps(nonscalar_irreps)


def _filter_features_by_predicate(
    x: torch.Tensor,
    irreps: Irreps,
    pred: Callable[[Irrep], bool],
) -> tuple[torch.Tensor, Irreps, torch.Tensor, Irreps]:
    keep_chunks: list[torch.Tensor] = []
    drop_chunks: list[torch.Tensor] = []
    keep_irreps: list[tuple[int, Irrep]] = []
    drop_irreps: list[tuple[int, Irrep]] = []

    for c in _chunks(irreps):
        chunk = x[..., c.start : c.start + c.size]
        if pred(c.ir):
            keep_chunks.append(chunk)
            keep_irreps.append((c.mul, c.ir))
        else:
            drop_chunks.append(chunk)
            drop_irreps.append((c.mul, c.ir))

    keep = (
        torch.cat(keep_chunks, dim=-1) if keep_chunks else x.new_zeros(*x.shape[:-1], 0)
    )
    drop = (
        torch.cat(drop_chunks, dim=-1) if drop_chunks else x.new_zeros(*x.shape[:-1], 0)
    )
    return keep, Irreps(keep_irreps), drop, Irreps(drop_irreps)


def _non_scalar_copy_magnitudes(x_nonscalar: torch.Tensor, irreps_nonscalar: Irreps):
    if x_nonscalar.shape[-1] == 0:
        return x_nonscalar.new_zeros(*x_nonscalar.shape[:-1], 0)

    mags: list[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps_nonscalar:
        size = mul * ir.dim
        chunk = x_nonscalar[..., offset : offset + size].reshape(
            *x_nonscalar.shape[:-1], mul, ir.dim
        )
        mags.append(torch.linalg.norm(chunk, dim=-1))
        offset += size
    return (
        torch.cat(mags, dim=-1)
        if mags
        else x_nonscalar.new_zeros(*x_nonscalar.shape[:-1], 0)
    )


def _scale_nonscalar_by_copy(
    x_nonscalar: torch.Tensor,
    irreps_nonscalar: Irreps,
    gates: torch.Tensor,
) -> torch.Tensor:
    if x_nonscalar.shape[-1] == 0:
        return x_nonscalar

    out: list[torch.Tensor] = []
    feat_offset = 0
    gate_offset = 0
    for mul, ir in irreps_nonscalar:
        size = mul * ir.dim
        chunk = x_nonscalar[..., feat_offset : feat_offset + size].reshape(
            *x_nonscalar.shape[:-1], mul, ir.dim
        )
        g = gates[..., gate_offset : gate_offset + mul].unsqueeze(-1)
        out.append((chunk * g).reshape(*x_nonscalar.shape[:-1], size))
        feat_offset += size
        gate_offset += mul

    return (
        torch.cat(out, dim=-1)
        if out
        else x_nonscalar.new_zeros(*x_nonscalar.shape[:-1], 0)
    )


def _num_nonscalar_copies(irreps: Irreps) -> int:
    return sum(mul for mul, ir in irreps if ir.l > 0)


def _all_product_irreps(irreps: Irreps, mul_per_irrep: int = 1) -> Irreps:
    reachable: set[Irrep] = set()
    for _, ir1 in irreps:
        for _, ir2 in irreps:
            for ir_out in ir1 * ir2:
                reachable.add(ir_out)
    return Irreps(
        [(mul_per_irrep, ir) for ir in sorted(reachable, key=lambda x: (x.l, x.p))]
    )


def _make_gate_irreps(
    target_irreps: Irreps,
    gate_irrep: str = "0e",
) -> tuple[Irreps, Irreps, Irreps, Irreps]:
    scalars = Irreps([(mul, ir) for mul, ir in target_irreps if ir.l == 0])
    gated = Irreps([(mul, ir) for mul, ir in target_irreps if ir.l > 0])
    if gated.dim > 0:
        g_ir = Irrep(gate_irrep)
        gates = Irreps([(mul, g_ir) for mul, _ in gated])
    else:
        gates = Irreps([])
    gate_input = (scalars + gates + gated).simplify()
    return scalars, gates, gated, gate_input


class _LinearThenActivation(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        act: nn.Module,
        *,
        linear_biases: bool = False,
    ):
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out, biases=linear_biases)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class _GateActivationBlock(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        linear_biases: bool = False,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        irreps_out = irreps_out.simplify()
        scalars, gates, gated, gate_input = _make_gate_irreps(irreps_out, gate_irrep)

        self.linear = Linear(irreps_in, gate_input, biases=linear_biases)
        scalar_fn = _callable_activation(
            scalar_activation_name, negative_slope=leakyrelu_slope
        )
        gate_fn = _callable_activation(
            gate_activation_name, negative_slope=leakyrelu_slope
        )
        self.gate = Gate(
            scalars,
            [scalar_fn for _ in scalars],
            gates,
            [gate_fn for _ in gates],
            gated,
        )
        gate_out = self.gate.irreps_out.simplify()
        self.post = (
            Linear(gate_out, irreps_out, biases=linear_biases)
            if gate_out != irreps_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.gate(x)
        x = self.post(x)
        return x


def _make_hidden_map(
    *,
    kind: ActivationKind,
    irreps_in: Irreps,
    irreps_out: Irreps,
    scalar_activation_name: str = "silu",
    gate_activation_name: str = "sigmoid",
    gate_irrep: str = "0e",
    linear_biases: bool = False,
    norm_bias: bool = False,
    norm_epsilon: float | None = None,
    leakyrelu_slope: float = 0.01,
) -> nn.Module:
    if kind == "normact":
        return _LinearThenActivation(
            irreps_in,
            irreps_out,
            NormActivation(
                irreps_out,
                scalar_nonlinearity=_callable_activation(
                    scalar_activation_name, leakyrelu_slope
                ),
                normalize=True,
                epsilon=norm_epsilon,
                bias=norm_bias,
            ),
            linear_biases=linear_biases,
        )
    if kind == "gate":
        return _GateActivationBlock(
            irreps_in,
            irreps_out,
            scalar_activation_name=scalar_activation_name,
            gate_activation_name=gate_activation_name,
            gate_irrep=gate_irrep,
            linear_biases=linear_biases,
            leakyrelu_slope=leakyrelu_slope,
        )
    if kind == "gate_scalars_mlp":
        return _LinearThenActivation(
            irreps_in,
            irreps_out,
            GateScalarsMLP(
                irreps_out,
                nonlin_scalars=scalar_activation(scalar_activation_name),
                nonlin_gate=scalar_activation(gate_activation_name),
            ),
            linear_biases=linear_biases,
        )
    if kind == "gate_magnitudes":
        return _LinearThenActivation(
            irreps_in,
            irreps_out,
            GateMagnitudes(
                irreps_out,
                nonlin_scalars=scalar_activation(scalar_activation_name),
                nonlin_gate=scalar_activation(gate_activation_name),
            ),
            linear_biases=linear_biases,
        )
    raise ValueError(f"Unsupported activation kind '{kind}'")


class NormActE3MLP(nn.Module):
    """Idea 2: Linear + NormActivation."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 2,
        scalar_activation_name: str = "silu",
        linear_biases: bool = False,
        norm_bias: bool = False,
        norm_epsilon: float | None = None,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        self.hidden_blocks = nn.ModuleList(
            [
                _make_hidden_map(
                    kind="normact",
                    irreps_in=(self.irreps_in if i == 0 else self.irreps_hidden),
                    irreps_out=self.irreps_hidden,
                    scalar_activation_name=scalar_activation_name,
                    linear_biases=linear_biases,
                    norm_bias=norm_bias,
                    norm_epsilon=norm_epsilon,
                    leakyrelu_slope=leakyrelu_slope,
                )
                for i in range(num_layers - 1)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        for block in self.hidden_blocks:
            x = block(x)
        return self.output_linear(x)


class GateE3MLP(nn.Module):
    """Idea 3: Linear + canonical e3nn Gate."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 2,
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        linear_biases: bool = False,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        self.hidden_blocks = nn.ModuleList(
            [
                _make_hidden_map(
                    kind="gate",
                    irreps_in=(self.irreps_in if i == 0 else self.irreps_hidden),
                    irreps_out=self.irreps_hidden,
                    scalar_activation_name=scalar_activation_name,
                    gate_activation_name=gate_activation_name,
                    gate_irrep=gate_irrep,
                    linear_biases=linear_biases,
                    leakyrelu_slope=leakyrelu_slope,
                )
                for i in range(num_layers - 1)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        for block in self.hidden_blocks:
            x = block(x)
        return self.output_linear(x)


class GateScalarsMLPE3MLP(nn.Module):
    """Idea 4: Linear + GateScalarsMLP."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 2,
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        linear_biases: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        self.hidden_blocks = nn.ModuleList(
            [
                _make_hidden_map(
                    kind="gate_scalars_mlp",
                    irreps_in=(self.irreps_in if i == 0 else self.irreps_hidden),
                    irreps_out=self.irreps_hidden,
                    scalar_activation_name=scalar_activation_name,
                    gate_activation_name=gate_activation_name,
                    linear_biases=linear_biases,
                )
                for i in range(num_layers - 1)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        for block in self.hidden_blocks:
            x = block(x)
        return self.output_linear(x)


class GateMagnitudesE3MLP(nn.Module):
    """Idea 5: Linear + GateMagnitudes."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 2,
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        linear_biases: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        self.hidden_blocks = nn.ModuleList(
            [
                _make_hidden_map(
                    kind="gate_magnitudes",
                    irreps_in=(self.irreps_in if i == 0 else self.irreps_hidden),
                    irreps_out=self.irreps_hidden,
                    scalar_activation_name=scalar_activation_name,
                    gate_activation_name=gate_activation_name,
                    linear_biases=linear_biases,
                )
                for i in range(num_layers - 1)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        for block in self.hidden_blocks:
            x = block(x)
        return self.output_linear(x)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        irreps_hidden: Irreps,
        *,
        activation_kind: ActivationKind = "normact",
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        residual_scale: float = 1.0,
        linear_biases: bool = False,
        norm_bias: bool = False,
        norm_epsilon: float | None = None,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        self.f = _make_hidden_map(
            kind=activation_kind,
            irreps_in=irreps_hidden,
            irreps_out=irreps_hidden,
            scalar_activation_name=scalar_activation_name,
            gate_activation_name=gate_activation_name,
            gate_irrep=gate_irrep,
            linear_biases=linear_biases,
            norm_bias=norm_bias,
            norm_epsilon=norm_epsilon,
            leakyrelu_slope=leakyrelu_slope,
        )
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.f(x)


class ResidualE3MLP(nn.Module):
    """Idea 7: Residual equivariant MLP blocks."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 3,
        num_residual_blocks: int | None = None,
        activation_kind: ActivationKind = "normact",
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        residual_scale: float = 1.0,
        linear_biases: bool = False,
        norm_bias: bool = False,
        norm_epsilon: float | None = None,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        if num_residual_blocks is None:
            num_residual_blocks = max(num_layers - 1, 1)

        self.input_linear = Linear(
            self.irreps_in, self.irreps_hidden, biases=linear_biases
        )
        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(
                    self.irreps_hidden,
                    activation_kind=activation_kind,
                    scalar_activation_name=scalar_activation_name,
                    gate_activation_name=gate_activation_name,
                    gate_irrep=gate_irrep,
                    residual_scale=residual_scale,
                    linear_biases=linear_biases,
                    norm_bias=norm_bias,
                    norm_epsilon=norm_epsilon,
                    leakyrelu_slope=leakyrelu_slope,
                )
                for _ in range(num_residual_blocks)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        x = self.input_linear(x)
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)


class _BilinearSelfTPBlock(nn.Module):
    def __init__(
        self,
        irreps_hidden: Irreps,
        *,
        tp_irreps: Irreps | None = None,
        tp_mul_per_irrep: int = 1,
        activation_kind: ActivationKind = "normact",
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        residual_scale: float = 1.0,
        linear_biases: bool = False,
        norm_bias: bool = False,
        norm_epsilon: float | None = None,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        self.lin_a = Linear(irreps_hidden, irreps_hidden, biases=linear_biases)
        self.lin_b = Linear(irreps_hidden, irreps_hidden, biases=linear_biases)
        self.lin_skip = Linear(irreps_hidden, irreps_hidden, biases=linear_biases)

        self.tp_irreps = (
            tp_irreps
            if tp_irreps is not None
            else _all_product_irreps(irreps_hidden, tp_mul_per_irrep)
        )
        if self.tp_irreps.dim == 0:
            raise ValueError("Bilinear block tp_irreps is empty.")

        self.tp = FullyConnectedTensorProduct(
            irreps_hidden,
            irreps_hidden,
            self.tp_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.tp_act = _make_hidden_map(
            kind=activation_kind,
            irreps_in=self.tp_irreps,
            irreps_out=self.tp_irreps,
            scalar_activation_name=scalar_activation_name,
            gate_activation_name=gate_activation_name,
            gate_irrep=gate_irrep,
            linear_biases=linear_biases,
            norm_bias=norm_bias,
            norm_epsilon=norm_epsilon,
            leakyrelu_slope=leakyrelu_slope,
        )
        self.lin_out = Linear(self.tp_irreps, irreps_hidden, biases=linear_biases)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lin_a(x)
        b = self.lin_b(x)
        q = self.tp(a, b)
        q = self.tp_act(q)
        return self.lin_skip(x) + self.residual_scale * self.lin_out(q)


class BilinearSelfTPE3MLP(nn.Module):
    """Idea 9: Bilinear irreps MLP via self tensor products."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 3,
        num_bilinear_blocks: int | None = None,
        tp_irreps: Irreps | str | None = None,
        tp_mul_per_irrep: int = 1,
        activation_kind: ActivationKind = "normact",
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        residual_scale: float = 1.0,
        linear_biases: bool = False,
        norm_bias: bool = False,
        norm_epsilon: float | None = None,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        if num_bilinear_blocks is None:
            num_bilinear_blocks = max(num_layers - 1, 1)

        tp_ir = _as_irreps(tp_irreps) if tp_irreps is not None else None

        self.input_linear = Linear(
            self.irreps_in, self.irreps_hidden, biases=linear_biases
        )
        self.blocks = nn.ModuleList(
            [
                _BilinearSelfTPBlock(
                    self.irreps_hidden,
                    tp_irreps=tp_ir,
                    tp_mul_per_irrep=tp_mul_per_irrep,
                    activation_kind=activation_kind,
                    scalar_activation_name=scalar_activation_name,
                    gate_activation_name=gate_activation_name,
                    gate_irrep=gate_irrep,
                    residual_scale=residual_scale,
                    linear_biases=linear_biases,
                    norm_bias=norm_bias,
                    norm_epsilon=norm_epsilon,
                    leakyrelu_slope=leakyrelu_slope,
                )
                for _ in range(num_bilinear_blocks)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        x = self.input_linear(x)
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)


class _InvariantSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        irreps_hidden: Irreps,
        *,
        attention_dim: int = 32,
        context_dim: int = 16,
        context_hidden_dims: Sequence[int] = (64,),
        token_hidden_dims: Sequence[int] = (64,),
        scalar_update_hidden_dims: Sequence[int] = (64,),
        scalar_activation_name: str = "silu",
        attention_temperature: float = 1.0,
        residual_tokens: bool = True,
        use_post_linear: bool = True,
        linear_biases: bool = False,
    ):
        super().__init__()
        self.irreps_hidden = irreps_hidden
        self.scalar_irreps, self.nonscalar_irreps = _split_scalar_nonscalar_irreps(
            irreps_hidden
        )
        self.scalar_dim = self.scalar_irreps.dim
        self.nonscalar_copy_count = _num_nonscalar_copies(self.nonscalar_irreps)
        self.attention_temperature = attention_temperature
        self.residual_tokens = residual_tokens

        self.context_dim = context_dim if self.scalar_dim > 0 else 0
        self.context_mlp = (
            _build_scalar_mlp(
                self.scalar_dim,
                self.context_dim,
                context_hidden_dims,
                scalar_activation_name,
            )
            if self.context_dim > 0
            else None
        )
        token_in_dim = 1 + self.context_dim
        self.q_proj = _build_scalar_mlp(
            token_in_dim, attention_dim, token_hidden_dims, scalar_activation_name
        )
        self.k_proj = _build_scalar_mlp(
            token_in_dim, attention_dim, token_hidden_dims, scalar_activation_name
        )

        if self.scalar_dim > 0:
            scalar_in_dim = self.scalar_dim + self.nonscalar_copy_count
            self.scalar_update = _build_scalar_mlp(
                scalar_in_dim,
                self.scalar_dim,
                scalar_update_hidden_dims,
                scalar_activation_name,
            )
        else:
            self.scalar_update = None

        self.post_linear = (
            Linear(irreps_hidden, irreps_hidden, biases=linear_biases)
            if use_post_linear
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build scalar context and per-copy non-scalar magnitudes (all invariant).
        x_scalar, _, x_ns, ns_ir = _split_scalar_nonscalar_features(
            x, self.irreps_hidden
        )
        ns_mags = _non_scalar_copy_magnitudes(x_ns, ns_ir)

        if self.context_mlp is not None:
            context = self.context_mlp(x_scalar)
        else:
            context = x.new_zeros(x.shape[0], 0)

        if self.scalar_update is not None:
            scalar_input = (
                torch.cat([x_scalar, ns_mags], dim=-1)
                if ns_mags.shape[-1] > 0
                else x_scalar
            )
            x_scalar_new = self.scalar_update(scalar_input)
        else:
            x_scalar_new = x_scalar

        scalar_cursor = 0
        out_chunks: list[torch.Tensor] = []
        for c in _chunks(self.irreps_hidden):
            chunk = x[:, c.start : c.start + c.size]
            if c.ir.l == 0:
                out_chunks.append(
                    x_scalar_new[:, scalar_cursor : scalar_cursor + c.size]
                )
                scalar_cursor += c.size
                continue

            # For each (mul x irrep), attention is over multiplicity tokens only.
            chunk_reshaped = chunk.reshape(x.shape[0], c.mul, c.ir.dim)  # [B, M, d_l]
            norms = torch.linalg.norm(chunk_reshaped, dim=-1, keepdim=True)  # [B, M, 1]
            if self.context_dim > 0:
                inv_tokens = torch.cat(
                    [norms, context.unsqueeze(1).expand(-1, c.mul, -1)], dim=-1
                )
            else:
                inv_tokens = norms

            q = self.q_proj(inv_tokens)  # [B, M, A]
            k = self.k_proj(inv_tokens)  # [B, M, A]
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            logits = logits / self.attention_temperature
            attn = torch.softmax(logits, dim=-1)  # invariant weights
            mixed = torch.einsum("bij,bjd->bid", attn, chunk_reshaped)
            if self.residual_tokens:
                mixed = mixed + chunk_reshaped
            out_chunks.append(mixed.reshape(x.shape[0], c.size))

        y = torch.cat(out_chunks, dim=-1)
        y = self.post_linear(y)
        return y


class InvariantSelfAttentionE3MLP(nn.Module):
    """Idea 11: invariant self-attention over multiplicity channels."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 3,
        num_attention_blocks: int | None = None,
        attention_dim: int = 32,
        context_dim: int = 16,
        context_hidden_dims: Sequence[int] = (64,),
        token_hidden_dims: Sequence[int] = (64,),
        scalar_update_hidden_dims: Sequence[int] = (64,),
        scalar_activation_name: str = "silu",
        attention_temperature: float = 1.0,
        residual_tokens: bool = True,
        use_post_linear: bool = True,
        linear_biases: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        if num_attention_blocks is None:
            num_attention_blocks = max(num_layers - 1, 1)

        self.input_linear = Linear(
            self.irreps_in, self.irreps_hidden, biases=linear_biases
        )
        self.blocks = nn.ModuleList(
            [
                _InvariantSelfAttentionBlock(
                    self.irreps_hidden,
                    attention_dim=attention_dim,
                    context_dim=context_dim,
                    context_hidden_dims=context_hidden_dims,
                    token_hidden_dims=token_hidden_dims,
                    scalar_update_hidden_dims=scalar_update_hidden_dims,
                    scalar_activation_name=scalar_activation_name,
                    attention_temperature=attention_temperature,
                    residual_tokens=residual_tokens,
                    use_post_linear=use_post_linear,
                    linear_biases=linear_biases,
                )
                for _ in range(num_attention_blocks)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        x = self.input_linear(x)
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)


class InvariantMoEE3MLP(nn.Module):
    """Idea 13: invariant-routed mixture-of-experts."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 2,
        num_experts: int = 4,
        expert_kind: ActivationKind = "normact",
        expert_num_layers: int | None = None,
        router_hidden_dims: Sequence[int] = (64, 64),
        router_activation_name: str = "silu",
        scalar_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        gate_irrep: str = "0e",
        router_temperature: float = 1.0,
        linear_biases: bool = False,
        norm_bias: bool = False,
        norm_epsilon: float | None = None,
        leakyrelu_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.router_temperature = router_temperature

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        if expert_num_layers is None:
            expert_num_layers = max(num_layers, 2)

        self.experts = nn.ModuleList(
            [
                _make_moe_expert(
                    kind=expert_kind,
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_out,
                    irreps_hidden=self.irreps_hidden,
                    num_layers=expert_num_layers,
                    scalar_activation_name=scalar_activation_name,
                    gate_activation_name=gate_activation_name,
                    gate_irrep=gate_irrep,
                    linear_biases=linear_biases,
                    norm_bias=norm_bias,
                    norm_epsilon=norm_epsilon,
                    leakyrelu_slope=leakyrelu_slope,
                )
                for _ in range(num_experts)
            ]
        )

        scalar_ir, nonscalar_ir = _split_scalar_nonscalar_irreps(self.irreps_in)
        inv_dim = scalar_ir.dim + _num_nonscalar_copies(nonscalar_ir)
        if inv_dim == 0:
            inv_dim = 1
        self.router = _build_scalar_mlp(
            inv_dim, num_experts, router_hidden_dims, router_activation_name
        )

    def _invariants(self, x: torch.Tensor) -> torch.Tensor:
        x_scalar, _, x_ns, ns_ir = _split_scalar_nonscalar_features(x, self.irreps_in)
        ns_mags = _non_scalar_copy_magnitudes(x_ns, ns_ir)
        inv = torch.cat([x_scalar, ns_mags], dim=-1)
        if inv.shape[-1] == 0:
            inv = x.new_zeros(x.shape[0], 1)
        return inv

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_router_weights: bool = False,
    ):
        if self.num_layers == 1:
            y = self.single(x)
            return (y, None) if return_router_weights else y

        inv = self._invariants(x)
        logits = self.router(inv) / self.router_temperature
        weights = torch.softmax(logits, dim=-1)  # [B, E]

        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # [B,E,D]
        y = torch.sum(weights.unsqueeze(-1) * expert_outs, dim=1)
        if return_router_weights:
            return y, weights
        return y


def _make_moe_expert(
    *,
    kind: ActivationKind,
    irreps_in: Irreps,
    irreps_out: Irreps,
    irreps_hidden: Irreps,
    num_layers: int,
    scalar_activation_name: str,
    gate_activation_name: str,
    gate_irrep: str,
    linear_biases: bool,
    norm_bias: bool,
    norm_epsilon: float | None,
    leakyrelu_slope: float,
) -> nn.Module:
    if kind == "normact":
        return NormActE3MLP(
            irreps_in,
            irreps_out,
            irreps_hidden=irreps_hidden,
            num_layers=num_layers,
            scalar_activation_name=scalar_activation_name,
            linear_biases=linear_biases,
            norm_bias=norm_bias,
            norm_epsilon=norm_epsilon,
            leakyrelu_slope=leakyrelu_slope,
        )
    if kind == "gate":
        return GateE3MLP(
            irreps_in,
            irreps_out,
            irreps_hidden=irreps_hidden,
            num_layers=num_layers,
            scalar_activation_name=scalar_activation_name,
            gate_activation_name=gate_activation_name,
            gate_irrep=gate_irrep,
            linear_biases=linear_biases,
            leakyrelu_slope=leakyrelu_slope,
        )
    if kind == "gate_scalars_mlp":
        return GateScalarsMLPE3MLP(
            irreps_in,
            irreps_out,
            irreps_hidden=irreps_hidden,
            num_layers=num_layers,
            scalar_activation_name=scalar_activation_name,
            gate_activation_name=gate_activation_name,
            linear_biases=linear_biases,
        )
    if kind == "gate_magnitudes":
        return GateMagnitudesE3MLP(
            irreps_in,
            irreps_out,
            irreps_hidden=irreps_hidden,
            num_layers=num_layers,
            scalar_activation_name=scalar_activation_name,
            gate_activation_name=gate_activation_name,
            linear_biases=linear_biases,
        )
    raise ValueError(f"Unsupported expert kind '{kind}'")


class _ScalarMagnitudeSelfTPGatedBlock(nn.Module):
    """
    Custom variant requested by user:
    1) scalars + non-scalar magnitudes -> scalar MLP
    2) self-TP of non-scalars
    3) concat(original non-scalars, TP non-scalars)
    4) gate by MLP outputs
    5) final e3nn Linear
    """

    def __init__(
        self,
        irreps_hidden: Irreps,
        *,
        mlp_hidden_dims: Sequence[int] = (128, 128),
        mlp_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        tp_irreps: Irreps | None = None,
        tp_mul_per_irrep: int = 1,
        residual: bool = True,
        linear_biases: bool = False,
    ):
        super().__init__()
        self.irreps_hidden = irreps_hidden
        self.residual = residual

        self.scalar_irreps, self.nonscalar_irreps = _split_scalar_nonscalar_irreps(
            irreps_hidden
        )
        self.scalar_dim = self.scalar_irreps.dim
        self.nonscalar_copy_count = _num_nonscalar_copies(self.nonscalar_irreps)

        if self.nonscalar_irreps.dim > 0:
            self.tp_irreps_all = (
                tp_irreps
                if tp_irreps is not None
                else _all_product_irreps(self.nonscalar_irreps, tp_mul_per_irrep)
            )
            self.self_tp = FullyConnectedTensorProduct(
                self.nonscalar_irreps,
                self.nonscalar_irreps,
                self.tp_irreps_all,
                internal_weights=True,
                shared_weights=True,
            )
            _, self.tp_nonscalar_irreps = _split_scalar_nonscalar_irreps(
                self.tp_irreps_all
            )
        else:
            self.tp_irreps_all = Irreps([])
            self.tp_nonscalar_irreps = Irreps([])
            self.self_tp = None

        self.ns_concat_irreps = (
            self.nonscalar_irreps + self.tp_nonscalar_irreps
        ).simplify()
        self.ns_concat_copy_count = _num_nonscalar_copies(self.ns_concat_irreps)

        mlp_in_dim = self.scalar_dim + self.nonscalar_copy_count
        if mlp_in_dim == 0:
            mlp_in_dim = 1
        mlp_out_dim = self.scalar_dim + self.ns_concat_copy_count
        self.mlp = _build_scalar_mlp(
            mlp_in_dim, mlp_out_dim, mlp_hidden_dims, mlp_activation_name
        )
        self.gate_act = scalar_activation(gate_activation_name)

        self.combined_irreps = (self.scalar_irreps + self.ns_concat_irreps).simplify()
        self.final_linear = Linear(
            self.combined_irreps, self.irreps_hidden, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scalar, _, x_ns, ns_ir = _split_scalar_nonscalar_features(
            x, self.irreps_hidden
        )
        ns_mags = _non_scalar_copy_magnitudes(x_ns, ns_ir)

        inv = torch.cat([x_scalar, ns_mags], dim=-1)
        if inv.shape[-1] == 0:
            inv = x.new_zeros(x.shape[0], 1)
        mlp_out = self.mlp(inv)

        scalar_new = mlp_out[:, : self.scalar_dim]
        gate_logits = mlp_out[:, self.scalar_dim :]
        gates = self.gate_act(gate_logits) if gate_logits.shape[-1] > 0 else gate_logits

        tp_ns = x.new_zeros(x.shape[0], 0)
        if self.self_tp is not None and x_ns.shape[-1] > 0:
            tp_full = self.self_tp(x_ns, x_ns)
            tp_ns, _, _, _ = _filter_features_by_predicate(
                tp_full, self.tp_irreps_all, lambda ir: ir.l > 0
            )

        ns_concat = torch.cat([x_ns, tp_ns], dim=-1)
        ns_scaled = _scale_nonscalar_by_copy(ns_concat, self.ns_concat_irreps, gates)

        combined = torch.cat([scalar_new, ns_scaled], dim=-1)
        y = self.final_linear(combined)
        if self.residual and y.shape[-1] == x.shape[-1]:
            y = y + x
        return y


class ScalarMagnitudeSelfTPGatedE3MLP(nn.Module):
    """Custom variant requested in the user prompt."""

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        *,
        irreps_hidden: Irreps | str | None = None,
        num_layers: int = 3,
        num_custom_blocks: int | None = None,
        mlp_hidden_dims: Sequence[int] = (128, 128),
        mlp_activation_name: str = "silu",
        gate_activation_name: str = "sigmoid",
        tp_irreps: Irreps | str | None = None,
        tp_mul_per_irrep: int = 1,
        residual: bool = True,
        linear_biases: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.irreps_in = _as_irreps(irreps_in)
        self.irreps_out = (
            _as_irreps(irreps_out) if irreps_out is not None else self.irreps_in
        )
        self.irreps_hidden = (
            _as_irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_in
        )
        self.num_layers = num_layers

        if num_layers == 1:
            self.single = Linear(self.irreps_in, self.irreps_out, biases=linear_biases)
            return

        if num_custom_blocks is None:
            num_custom_blocks = max(num_layers - 1, 1)
        tp_ir = _as_irreps(tp_irreps) if tp_irreps is not None else None

        self.input_linear = Linear(
            self.irreps_in, self.irreps_hidden, biases=linear_biases
        )
        self.blocks = nn.ModuleList(
            [
                _ScalarMagnitudeSelfTPGatedBlock(
                    self.irreps_hidden,
                    mlp_hidden_dims=mlp_hidden_dims,
                    mlp_activation_name=mlp_activation_name,
                    gate_activation_name=gate_activation_name,
                    tp_irreps=tp_ir,
                    tp_mul_per_irrep=tp_mul_per_irrep,
                    residual=residual,
                    linear_biases=linear_biases,
                )
                for _ in range(num_custom_blocks)
            ]
        )
        self.output_linear = Linear(
            self.irreps_hidden, self.irreps_out, biases=linear_biases
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.single(x)
        x = self.input_linear(x)
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)
