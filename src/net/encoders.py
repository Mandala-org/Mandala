"""
Encoders for node / edge raw features → hidden irreps.

Changes in v2
-------------
* NodeEncoder drops overlap-diag input (constant per species).
* Non-linearity is selected via `net.activations.make_nonlinearity`.
* Dropout uses `e3nn.nn.Dropout` so it masks every coefficient.
* Full device awareness: pass `device` to ctor; tensors & Linear weights
  are allocated on that device.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import nn
from e3nn.o3 import Irreps, Linear
from e3nn.nn import Dropout

from net.common import HyperParams, RadialMLP
from net.activations import make_nonlinearity


# --------------------------------------------------------------------------- #
class NodeEncoder(nn.Module):
    """
    Node features = *only* element one-hot → learned embedding (scalars).

    The embedding is mapped (Linear) to `out_irreps` (scalars only in our
    current setup, but we keep it general).
    """

    def __init__(
        self,
        node_one_hot_dim: int,
        out_irreps: Irreps,
        hp: HyperParams,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hp = hp
        self.out_irreps = out_irreps
        self.device = torch.device(device)

        scalar_width = hp.hidden_base_dim
        self.elem_emb = nn.Embedding(
            node_one_hot_dim,
            scalar_width,
            device=self.device,
            dtype=dtype,
        )
        nn.init.normal_(self.elem_emb.weight, std=0.2)

        self.lin = Linear(
            Irreps(f"{scalar_width}x0e"),
            out_irreps,
            internal_weights=True,
        )

        self.nl = make_nonlinearity(out_irreps, hp)
        self.dropout = (
            Dropout(out_irreps, p=hp.dropout) if hp.dropout > 0.0 else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        one_hot: torch.Tensor,  # (N, n_elem)
    ) -> torch.Tensor:
        emb = self.elem_emb(one_hot.argmax(dim=-1))
        h = self.lin(emb)
        h = self.nl(h)
        h = self.dropout(h)
        return h


# --------------------------------------------------------------------------- #
class EdgeEncoder(nn.Module):
    """
    Args
    ----
    n_edge_types   : total number of ordered element-pairs
    n_radial       : length of radial distance basis
    sh_irreps      : Irreps of provided spherical harmonic vector
    offdiag_irrep_dim : size of overlap_offdiag vector (may be 0 / None)
    """

    def __init__(
        self,
        n_edge_types: int,
        n_radial: int,
        sh_irreps: Irreps,
        offdiag_irrep_dim: Optional[int],
        out_irreps: Irreps,
        hp: HyperParams,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hp = hp
        self.out_irreps = out_irreps
        self.device = torch.device(device)

        # 1) scalar embeddings ------------------------------------------------
        sc_width = hp.hidden_base_dim
        self.edge_emb = nn.Embedding(
            n_edge_types,
            sc_width,
            device=self.device,
            dtype=dtype,
        )
        nn.init.normal_(self.edge_emb.weight, std=0.2)

        self.radial_net = RadialMLP(
            in_dim=n_radial,
            out_dim=sc_width,
            layers=hp.radial_layers,
            act=hp.activation_scalar,
            dtype=dtype,
        ).to(self.device)

        if offdiag_irrep_dim:
            self.proj_off = nn.Linear(
                offdiag_irrep_dim,
                sc_width,
            )
        else:
            self.proj_off = None

        scalar_input_ir = Irreps(f"{sc_width * (2 + bool(self.proj_off))}x0e")
        self.lin_scalar = Linear(
            scalar_input_ir,
            out_irreps,
            internal_weights=True,
        )

        # 2) spherical harmonics projector ------------------------------------
        self.sh_irreps = sh_irreps
        self.sh_proj = Linear(
            sh_irreps,
            out_irreps,
            internal_weights=True,
        )

        # 3) non-linearity + dropout
        self.nl = make_nonlinearity(out_irreps, hp)
        self.dropout = (
            Dropout(out_irreps, p=hp.dropout) if hp.dropout > 0.0 else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        one_hot: torch.Tensor,
        length_emb: torch.Tensor,
        sh: torch.Tensor,
        overlap_off: Optional[torch.Tensor] = None,
        *,
        select_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if select_indices is not None:
            one_hot = one_hot[select_indices]
            length_emb = length_emb[select_indices]
            sh = sh[select_indices]
            if overlap_off is not None:
                overlap_off = overlap_off[select_indices]

        scalars = [
            self.edge_emb(one_hot.argmax(dim=-1)),
            self.radial_net(length_emb),
        ]
        if self.proj_off is not None and overlap_off is not None:
            scalars.append(self.proj_off(overlap_off))

        h_scalar = self.lin_scalar(torch.cat(scalars, dim=-1))
        h = h_scalar + self.sh_proj(sh)
        h = self.nl(h)
        h = self.dropout(h)
        return h
