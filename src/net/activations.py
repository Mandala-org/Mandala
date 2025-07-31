"""
activations.py
==============

Factory helpers that return *equivariant* non-linearities and normalisation
layers for arbitrary Irreps.

Supported kinds
---------------
• "gate"      →  e3nn.nn.Gate
• "normact"   →  e3nn.nn.NormActivation
• "s2act"     →  e3nn.nn.S2Activation
• "id"        →  identity (no non-linearity)

The choice is controlled by ``cfg.nonlin_kind`` (see Config).
Batch-Norm (equivariant) can be toggled independently through
``cfg.batch_norm`` – if enabled we apply it **before** the non-linearity.

Scalar activation names ("silu", "relu", …) are mapped to the
corresponding `torch.nn.Module` instances via :func:`scalar_activation`.
"""

from __future__ import annotations
from typing import Dict, Callable

import torch
from torch import nn
from e3nn.o3 import Irreps
from e3nn.nn import NormActivation, S2Activation

from net.common import Config

# Public ------------------------------------------------------------------- #
__all__ = [
    "scalar_activation",
    "make_nonlinearity",
]

# -------------------------------------------------------------------------- #
_SCALAR_ACTS: Dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
}


def scalar_activation(name: str) -> nn.Module:
    """Return *instance* of scalar activation by name (case-insensitive)."""
    try:
        return _SCALAR_ACTS[name.lower()]()
    except KeyError as exc:  # pragma: no cover
        raise ValueError(
            f"Unknown scalar activation '{name}'. " f"Supported: {list(_SCALAR_ACTS)}"
        ) from exc


class GateScalarsMLP(nn.Module):
    """
    A module that
    1. Takes scalars from inputs
    2. Applies an MLP to them
    3. Returns a concatenation of the original scalars (after application of a non-linearity)
        and the non-scalars multiplied by the output of the MLP (per irrep).
    """

    def __init__(self, irreps: Irreps, nonlin: nn.Module):
        super().__init__()
        self.irreps = irreps
        self.nonlin = nonlin
        self.n_scalars = sum(mul for mul, ir in irreps if ir.l == 0 and ir.p == 1)
        self.n_non_scalars = sum(mul for mul, ir in irreps if ir.l > 0 or ir.p == -1)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_scalars, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.n_non_scalars),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the gate to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., irreps.dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., irreps.dim).
        """
        # Split scalars and non-scalars
        scalars = x[..., : self.n_scalars]

        # Apply MLP to scalars and multiply with non-scalars
        gate_output = self.mlp(scalars)

        # Apply non-linearity to scalars
        scalars = self.nonlin(scalars)

        output = [scalars]
        start_g = 0
        start_x = self.n_scalars
        for mul, ir in self.irreps:
            if not (ir.l == 0 and ir.p == 1):
                # Non-scalars
                end_x = start_x + mul * ir.dim
                non_scalars = x[..., start_x:end_x]
                # Get the gate values for this irrep
                end_g = start_g + mul
                gate = gate_output[..., start_g:end_g].unsqueeze(-1)
                # Reshape non_scalars to be able to multiply by gate
                non_scalars = non_scalars.reshape(*non_scalars.shape[:-1], mul, ir.dim)
                output.append((non_scalars * gate).reshape(*non_scalars.shape[:-2], -1))
                start_g = end_g
                start_x = end_x
        return torch.cat(output, dim=-1)


class GateMagnitudes(nn.Module):
    """
    A module that
    1. Applies a non-linearity to the scalars in the input
    2. Computes the magnitudes of the non-scalars
    3. Applies an activation function to the magnitudes
    4. Multiplies the non-scalars by the activation output
    """

    def __init__(
        self, irreps: Irreps, nonlin_scalars: nn.Module, nonlin_magnitudes: nn.Module
    ):
        super().__init__()
        self.irreps = irreps
        self.nonlin_scalars = nonlin_scalars
        self.nonlin_magnitudes = nonlin_magnitudes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the gate to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., irreps.dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., irreps.dim).
        """
        output = []
        start = 0
        n_scalars = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        output.append(self.nonlin_scalars(x[..., :n_scalars]))
        start = n_scalars
        for mul, ir in self.irreps:
            if not (ir.l == 0 and ir.p == 1):
                # Non-scalars
                end = start + mul * ir.dim
                non_scalars = x[..., start:end].reshape(*x.shape[:-1], mul, ir.dim)
                # Compute magnitudes
                magnitudes = torch.linalg.norm(non_scalars, dim=-1)
                # Apply non-linearity to magnitudes
                magnitudes = self.nonlin_magnitudes(magnitudes)
                # Multiply non-scalars by magnitudes
                non_scalars = non_scalars * magnitudes.unsqueeze(-1)
                # Reshape back to original shape
                output.append(non_scalars.reshape(*x.shape[:-1], -1))
                start = end
        return torch.cat(output, dim=-1)


# -------------------------------------------------------------------------- #
def make_nonlinearity(
    irreps: Irreps,
    cfg: Config,
) -> nn.Module:
    """
    Build an equivariant **normalisation + activation** module.

    Parameters
    ----------
    irreps
        Input/output representation (unchanged by the non-linearity).
    cfg
        Any object that exposes the attributes used below
        (`nonlin_kind`, `activation_scalar`, `batch_norm`, `norm_kind`).

    Returns
    -------
    torch.nn.Module
        Callable that maps (..., irreps.dim) → (..., irreps.dim)
    """
    kind = cfg.nonlin_kind.lower()

    if kind == "normact":
        # Norm kind: "component" (default) or "norm"
        normalise_over = "component" if cfg.norm_kind == "component" else "norm"
        return NormActivation(
            irreps,
            scalar_activation(cfg.activation_scalar),
            normalize=normalise_over,
        )
    elif kind == "s2act":
        return S2Activation(
            irreps,
            scalar_activation(cfg.activation_scalar),
            res=cfg.s2act_res,
        )
    elif kind == "gate_scalars_mlp":
        return GateScalarsMLP(
            irreps,
            scalar_activation(cfg.activation_scalar),
        )
    elif kind == "gate_magnitudes":
        return GateMagnitudes(
            irreps,
            scalar_activation(cfg.activation_scalar),
            scalar_activation(cfg.activation_magnitude),
        )
    else:
        raise ValueError(
            f"Unknown nonlin_kind '{cfg.nonlin_kind}'. "
            f"Expected one of: 'normact', 's2act', 'gate_scalars_mlp', 'gate_magnitudes'."
        )
