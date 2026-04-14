from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import FullyConnectedTensorProduct, Irrep, Irreps, Linear


def scalar_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "softplus":
        return nn.Softplus()
    if name == "softsign":
        return nn.Softsign()
    raise ValueError(f"Unsupported activation '{name}'")


def scalar_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
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
    if name == "leakyrelu":
        return F.leaky_relu
    if name == "softplus":
        return F.softplus
    if name == "softsign":
        return F.softsign
    raise ValueError(f"Unsupported activation '{name}'")


def odd_safe_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    if name == "identity":
        return lambda x: x
    return torch.tanh


def _copy_slices(irreps: Irreps) -> list[tuple[int, int, int, Irrep]]:
    out: list[tuple[int, int, int, Irrep]] = []
    start = 0
    for mul, ir in Irreps(irreps):
        for _ in range(mul):
            out.append((start, ir.dim, mul, ir))
            start += ir.dim
    return out


class ScaledLinear(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        output_scale: float = 1.0,
        weight_init_scale: float = 1.0,
        biases: bool = False,
    ) -> None:
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out, biases=biases)
        self.output_scale = float(output_scale)
        self.weight_init_scale = float(weight_init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.weight.data.mul_(self.weight_init_scale)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_scale * self.linear(x)


class GateMagnitudesActivation(nn.Module):
    def __init__(
        self,
        irreps: Irreps,
        *,
        scalar_activation: str = "silu",
        odd_scalar_activation: str = "tanh",
        gate_activation: str = "sigmoid",
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        self.scalar_act = scalar_activation_fn(scalar_activation)
        self.odd_scalar_act = odd_safe_activation(odd_scalar_activation)
        self.gate_act = scalar_activation_fn(gate_activation)
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: list[torch.Tensor] = []
        cursor = 0
        for mul, ir in self.irreps:
            size = mul * ir.dim
            block = x[..., cursor : cursor + size]
            cursor += size
            if ir.l == 0:
                block = block.reshape(*block.shape[:-1], mul, ir.dim)
                block = (
                    self.scalar_act(block) if ir.p == 1 else self.odd_scalar_act(block)
                )
                out.append(block.reshape(*block.shape[:-2], size))
                continue

            block = block.reshape(*block.shape[:-1], mul, ir.dim)
            mags = torch.linalg.norm(block, dim=-1, keepdim=True).clamp_min(
                self.epsilon
            )
            gains = self.gate_act(mags)
            out.append((block * (gains / mags)).reshape(*block.shape[:-2], size))
        return torch.cat(out, dim=-1)


class InvariantFiLMActivation(nn.Module):
    def __init__(
        self,
        irreps: Irreps,
        *,
        hidden_dim: int = 128,
        scalar_activation: str = "silu",
        odd_scalar_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.scalar_activation = scalar_activation_fn(scalar_activation)
        self.odd_scalar_activation = odd_safe_activation(odd_scalar_activation)
        self.copy_specs = _copy_slices(Irreps(irreps))
        self.scalar_copy_indices = [
            idx for idx, (_, _, _, ir) in enumerate(self.copy_specs) if ir.l == 0
        ]
        inv_dim = len(self.copy_specs)
        out_dim = len(self.copy_specs) + len(self.scalar_copy_indices)
        self.mlp = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            scalar_activation_module(scalar_activation),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invariants: list[torch.Tensor] = []
        blocks: list[torch.Tensor] = []
        for start, dim, _, ir in self.copy_specs:
            block = x[..., start : start + dim]
            blocks.append(block)
            if ir.l == 0:
                invariants.append(block.reshape(*block.shape[:-1], -1))
            else:
                invariants.append(torch.linalg.norm(block, dim=-1, keepdim=True))

        inv = torch.cat(invariants, dim=-1)
        params = self.mlp(inv)
        gains = torch.sigmoid(params[..., : len(self.copy_specs)]) * 2.0
        scalar_biases = params[..., len(self.copy_specs) :]

        out: list[torch.Tensor] = []
        bias_cursor = 0
        for idx, ((_, _, _, ir), block) in enumerate(zip(self.copy_specs, blocks)):
            gain = gains[..., idx : idx + 1]
            y = gain * block
            if ir.l == 0:
                bias = scalar_biases[..., bias_cursor : bias_cursor + 1]
                y = y + bias
                y = (
                    self.scalar_activation(y)
                    if ir.p == 1
                    else self.odd_scalar_activation(y)
                )
                bias_cursor += 1
            out.append(y)
        return torch.cat(out, dim=-1)


class EquivariantRMSNorm(nn.Module):
    def __init__(self, irreps: Irreps, eps: float = 1e-8) -> None:
        super().__init__()
        self.copy_specs = _copy_slices(Irreps(irreps))
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(len(self.copy_specs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for idx, (start, dim, _, _) in enumerate(self.copy_specs):
            block = x[..., start : start + dim]
            norm = torch.linalg.norm(block, dim=-1, keepdim=True).clamp_min(self.eps)
            out.append(block * (self.weight[idx] / norm))
        return torch.cat(out, dim=-1)


def make_gate_activation(
    irreps: Irreps,
    *,
    scalar_activation: str = "silu",
    odd_scalar_activation: str = "tanh",
    gate_activation: str = "sigmoid",
    odd_gate_activation: str = "tanh",
) -> Gate:
    irreps = Irreps(irreps).simplify()
    irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0])
    irreps_gated = Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
    if irreps_gated.dim > 0:
        has_even_scalar = any(ir.l == 0 and ir.p == 1 for _, ir in irreps)
        gate_irrep = Irrep("0e" if has_even_scalar else "0o")
        irreps_gates = Irreps([(mul, gate_irrep) for mul, _ in irreps_gated]).simplify()
    else:
        irreps_gates = Irreps([])

    scalar_acts = [
        (
            scalar_activation_fn(scalar_activation)
            if ir.p == 1
            else odd_safe_activation(odd_scalar_activation)
        )
        for _, ir in irreps_scalars
    ]
    gate_acts = [
        (
            scalar_activation_fn(gate_activation)
            if ir.p == 1
            else odd_safe_activation(odd_gate_activation)
        )
        for _, ir in irreps_gates
    ]

    return Gate(
        irreps_scalars,
        scalar_acts,
        irreps_gates,
        gate_acts,
        irreps_gated,
    )


class LinearThenAct(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        act: nn.Module,
        *,
        output_scale: float,
        weight_init_scale: float,
    ) -> None:
        super().__init__()
        self.linear = ScaledLinear(
            irreps_in,
            irreps_out,
            output_scale=output_scale,
            weight_init_scale=weight_init_scale,
        )
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class GateBlock(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_hidden: Irreps,
        *,
        output_scale: float,
        weight_init_scale: float,
        scalar_activation: str,
        odd_scalar_activation: str,
        gate_activation: str,
        odd_gate_activation: str,
    ) -> None:
        super().__init__()
        gate = make_gate_activation(
            irreps_hidden,
            scalar_activation=scalar_activation,
            odd_scalar_activation=odd_scalar_activation,
            gate_activation=gate_activation,
            odd_gate_activation=odd_gate_activation,
        )
        self.linear = ScaledLinear(
            irreps_in,
            gate.irreps_in,
            output_scale=output_scale,
            weight_init_scale=weight_init_scale,
        )
        self.gate = gate
        self.post = (
            ScaledLinear(
                gate.irreps_out,
                irreps_hidden,
                output_scale=1.0,
                weight_init_scale=1.0,
            )
            if gate.irreps_out != Irreps(irreps_hidden)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.gate(self.linear(x)))


class ResidualBlock(nn.Module):
    def __init__(self, block: nn.Module, width: int, residual_scale: float) -> None:
        super().__init__()
        self.block = block
        self.layerscale = nn.Parameter(torch.full((width,), residual_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layerscale * self.block(x)


class BilinearSelfTPBlock(nn.Module):
    def __init__(
        self,
        irreps_hidden: Irreps,
        *,
        output_scale: float = 1.0,
        weight_init_scale: float = 1.0,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        irreps_hidden = Irreps(irreps_hidden)
        self.lin_a = ScaledLinear(
            irreps_hidden,
            irreps_hidden,
            output_scale=output_scale,
            weight_init_scale=weight_init_scale,
        )
        self.lin_b = ScaledLinear(
            irreps_hidden,
            irreps_hidden,
            output_scale=output_scale,
            weight_init_scale=weight_init_scale,
        )

        reachable: set[Irrep] = set()
        for _, ir1 in irreps_hidden:
            for _, ir2 in irreps_hidden:
                for ir_out in ir1 * ir2:
                    reachable.add(ir_out)
        interaction_irreps = Irreps(
            [(1, ir) for ir in sorted(reachable, key=lambda ir: (ir.l, ir.p))]
        )

        self.tp = FullyConnectedTensorProduct(
            irreps_hidden,
            irreps_hidden,
            interaction_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.norm = NormActivation(
            interaction_irreps,
            scalar_nonlinearity=torch.sigmoid,
            normalize=True,
        )
        self.proj = ScaledLinear(
            interaction_irreps,
            irreps_hidden,
            output_scale=1.0,
            weight_init_scale=1.0,
        )
        self.skip = nn.Parameter(torch.full((irreps_hidden.dim,), residual_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.tp(self.lin_a(x), self.lin_b(x))
        q = self.norm(q)
        return x + self.skip * self.proj(q)


class InvariantFiLME3MLPBlock(nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_hidden: Irreps,
        *,
        cfg: "VariantConfig",
    ) -> None:
        super().__init__()
        self.pre = ScaledLinear(
            irreps_in,
            irreps_hidden,
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
        )
        self.film = InvariantFiLMActivation(
            irreps_hidden,
            hidden_dim=cfg.film_hidden_dim,
            scalar_activation=cfg.scalar_activation,
            odd_scalar_activation=cfg.odd_scalar_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.film(self.pre(x))


@dataclass
class VariantConfig:
    output_scale: float = 1.0
    weight_init_scale: float = 1.0
    residual_scale: float = 0.25
    scalar_activation: str = "silu"
    odd_scalar_activation: str = "tanh"
    gate_activation: str = "sigmoid"
    odd_gate_activation: str = "tanh"
    film_hidden_dim: int = 128
    pre_norm: bool = False
    norm_eps: float = 1e-8


def make_variant_block(
    variant: str,
    irreps_in: Irreps,
    irreps_hidden: Irreps,
    cfg: VariantConfig,
) -> nn.Module:
    variant = variant.lower()
    if variant == "normact":
        return LinearThenAct(
            irreps_in,
            irreps_hidden,
            NormActivation(
                irreps_hidden,
                scalar_nonlinearity=torch.sigmoid,
                normalize=True,
            ),
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
        )
    if variant == "gate":
        return GateBlock(
            irreps_in,
            irreps_hidden,
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
            scalar_activation=cfg.scalar_activation,
            odd_scalar_activation=cfg.odd_scalar_activation,
            gate_activation=cfg.gate_activation,
            odd_gate_activation=cfg.odd_gate_activation,
        )
    if variant == "gatemagnitudes":
        return LinearThenAct(
            irreps_in,
            irreps_hidden,
            GateMagnitudesActivation(
                irreps_hidden,
                scalar_activation=cfg.scalar_activation,
                odd_scalar_activation=cfg.odd_scalar_activation,
                gate_activation=cfg.gate_activation,
            ),
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
        )
    if variant == "film":
        return InvariantFiLME3MLPBlock(irreps_in, irreps_hidden, cfg=cfg)
    if variant == "resnormact":
        inner = LinearThenAct(
            irreps_hidden,
            irreps_hidden,
            NormActivation(
                irreps_hidden,
                scalar_nonlinearity=torch.sigmoid,
                normalize=True,
            ),
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
        )
        if Irreps(irreps_in) != Irreps(irreps_hidden):
            return nn.Sequential(
                ScaledLinear(
                    irreps_in, irreps_hidden, output_scale=1.0, weight_init_scale=1.0
                ),
                ResidualBlock(inner, Irreps(irreps_hidden).dim, cfg.residual_scale),
            )
        return ResidualBlock(inner, Irreps(irreps_hidden).dim, cfg.residual_scale)
    if variant == "resgatemagnitudes":
        inner = LinearThenAct(
            irreps_hidden,
            irreps_hidden,
            GateMagnitudesActivation(
                irreps_hidden,
                scalar_activation=cfg.scalar_activation,
                odd_scalar_activation=cfg.odd_scalar_activation,
                gate_activation=cfg.gate_activation,
            ),
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
        )
        if Irreps(irreps_in) != Irreps(irreps_hidden):
            return nn.Sequential(
                ScaledLinear(
                    irreps_in, irreps_hidden, output_scale=1.0, weight_init_scale=1.0
                ),
                ResidualBlock(inner, Irreps(irreps_hidden).dim, cfg.residual_scale),
            )
        return ResidualBlock(inner, Irreps(irreps_hidden).dim, cfg.residual_scale)
    if variant == "bilinear":
        if Irreps(irreps_in) != Irreps(irreps_hidden):
            return nn.Sequential(
                ScaledLinear(
                    irreps_in, irreps_hidden, output_scale=1.0, weight_init_scale=1.0
                ),
                BilinearSelfTPBlock(
                    irreps_hidden,
                    output_scale=cfg.output_scale,
                    weight_init_scale=cfg.weight_init_scale,
                    residual_scale=cfg.residual_scale,
                ),
            )
        return BilinearSelfTPBlock(
            irreps_hidden,
            output_scale=cfg.output_scale,
            weight_init_scale=cfg.weight_init_scale,
            residual_scale=cfg.residual_scale,
        )
    raise ValueError(f"Unsupported E3MLP variant '{variant}'")
