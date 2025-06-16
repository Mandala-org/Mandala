"""
common.py
=========

Light-weight utilities that are shared by *all* network sub-modules:

* Hyper-parameter dataclass : :class:`HyperParams`
* Hidden-width Irreps constructor : :func:`build_hidden_irreps`
* A flexible N-layer radial MLP   : :class:`RadialMLP`

**No activation / norm helpers live here anymore** – those have been moved
to :pymod:`net.activations` so we avoid a circular import between files.
"""

from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Sequence

import torch
from torch import nn
from e3nn.o3 import Irreps


# ════════════════════════════════════════════════════════════════════════
# 1.  Hyper-parameters
# ════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class HyperParams:
    # -------------- representation shape --------------------------------
    l_max: int = 3
    hidden_base_dim: int = 64  # multiplicity at ℓ = 0

    # -------------- depth / topology ------------------------------------
    num_layers_gnn: int = 4
    num_layers_matrix: int = 2
    use_edge_updates: bool = True
    use_self_update: bool = True

    # -------------- non-linearity & norm --------------------------------
    nonlin_kind: str = "gate"  # "gate" | "normact" | "s2act" | "id"
    activation_scalar: str = "silu"  # SiLU by default
    batch_norm: bool = False
    norm_kind: str = "component"  # for NormActivation: "component" | "norm"

    # -------------- regularisation --------------------------------------
    dropout: float = 0.0  # dropout on *all* irrep coefficients
    residual_connections: bool = True

    # -------------- radial basis ----------------------------------------
    n_radial: int = 64
    radial_layers: Sequence[int] = field(
        default_factory=lambda: (128,)
    )  # e.g. (128,) → 2-layer MLP
    share_radial: bool = True

    # -------------- output head -----------------------------------------
    head_depth: int = 3
    head_hidden_mul: float = 1.0  # can be <1 or >1
    hidden_mul_clip: float = 4.0  # safety cap to avoid huge widths

    # -------------- loss weighting --------------------------------------
    energy_loss_coef: float = 0.1
    electron_loss_coef: float = 0.05


# ════════════════════════════════════════════════════════════════════════
# 2.  Hidden irreps auto-builder  (cached – deterministic)
# ════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def build_hidden_irreps(l_max: int, base_dim: int) -> Irreps:
    """
    Create `Irreps` with multiplicity halved for every ℓ > 0
    and *both* parity channels present.

    Example  (base_dim=32, l_max=2) ::

        32x0e + 32x0o + 16x1e + 16x1o + 8x2e + 8x2o
    """
    parts: List[str] = []
    for ell in range(l_max + 1):
        mul = max(base_dim // (2**ell), 1)
        parts.append(f"{mul}x{ell}e")
        parts.append(f"{mul}x{ell}o")
    return Irreps("+".join(parts)).simplify()


# ════════════════════════════════════════════════════════════════════════
# 3.  Flexible radial MLP
# ════════════════════════════════════════════════════════════════════════
class RadialMLP(nn.Module):
    """
    Generic N-layer feed-forward network applied to the radial
    distance embedding (soft one-hot, Bessel basis, …).

    Parameters
    ----------
    in_dim
        Input size (n_radial from dataset).
    layers
        Sequence of **hidden layer widths**.  The final width
        *out_dim* is appended automatically if not already present.
    out_dim
        Output size produced per edge (number of scalar coefficients).
    act
        Name of scalar activation ("silu", "relu", ...).

    Example
    -------
    >>> mlp = RadialMLP(64, out_dim=32, layers=(128, 64))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        layers: Sequence[int] = (128,),
        act: str = "silu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from net.activations import scalar_activation  # local import

        layer_dims = list(layers)
        if not layer_dims or layer_dims[-1] != out_dim:
            layer_dims.append(out_dim)

        seq: List[nn.Module] = []
        prev = in_dim
        for hid in layer_dims:
            seq.append(nn.Linear(prev, hid, dtype=dtype))
            if hid != out_dim:  # no activation after last layer
                seq.append(scalar_activation(act))
            prev = hid
        self.net = nn.Sequential(*seq)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
