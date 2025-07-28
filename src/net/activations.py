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

from torch import nn
from e3nn.o3 import Irreps
from e3nn.nn import Gate, NormActivation, BatchNorm

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
}


def scalar_activation(name: str) -> nn.Module:
    """Return *instance* of scalar activation by name (case-insensitive)."""
    try:
        return _SCALAR_ACTS[name.lower()]()
    except KeyError as exc:  # pragma: no cover
        raise ValueError(
            f"Unknown scalar activation '{name}'. " f"Supported: {list(_SCALAR_ACTS)}"
        ) from exc


def _split_scalars_and_rest(irreps: Irreps):
    """Return (Irreps[0e scalars], Irreps[others])."""
    scalars, nonscalars = [], []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            scalars.append((mul, ir))
        else:
            nonscalars.append((mul, ir))
    return Irreps(scalars).simplify(), Irreps(nonscalars).simplify()


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

    # optional BatchNorm (equivariant)
    pre_norm: nn.Module
    if getattr(cfg, "batch_norm", False):
        pre_norm = BatchNorm(irreps)
    else:
        pre_norm = nn.Identity()

    if kind == "id":
        return nn.Sequential(pre_norm)  # nothing else

    if kind == "gate":
        scalars_ir, rest_ir = _split_scalars_and_rest(irreps)

        # Gate requires one scalar *per* gated irrep
        if (
            scalars_ir.dim == 0
            or rest_ir.dim == 0
            or scalars_ir.num_irreps != rest_ir.num_irreps
        ):
            raise ValueError(
                "Cannot create Gate nonlinearity: "
                "expected at least one scalar and one non-scalar irrep, "
                "and the same number of irreps in both."
            )
        else:
            g_act = scalar_activation("sigmoid")
            s_act = scalar_activation(cfg.activation_scalar)
            num = scalars_ir.num_irreps
            return nn.Sequential(
                pre_norm,
                Gate(
                    irreps_scalars=scalars_ir,
                    act_scalars=[s_act] * num,
                    irreps_gates=scalars_ir,
                    act_gates=[g_act] * num,
                    irreps_gated=rest_ir,
                ),
            )

    if kind == "normact":
        # Norm kind: "component" (default) or "norm"
        normalise_over = "component" if cfg.norm_kind == "component" else "norm"
        return nn.Sequential(
            pre_norm,
            NormActivation(
                irreps,
                scalar_activation(cfg.activation_scalar),
                normalize=normalise_over,
            ),
        )

    if kind == "s2act":
        layer = NormActivation(
            irreps,
            scalar_activation(cfg.activation_scalar),
            normalize="component",
        )
        return nn.Sequential(pre_norm, layer)

    raise ValueError(
        f"Unknown nonlin_kind '{cfg.nonlin_kind}'. "
        f"Expected 'gate', 'normact', 's2act', or 'id'."
    )
