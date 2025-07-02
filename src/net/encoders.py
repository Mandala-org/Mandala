"""
Encoders for node and edge raw features → hidden irreps.

Features
--------
* NodeEncoder drops overlap-diag input (constant per species).
* Non-linearity is selected via `net.activations.make_nonlinearity`.
* Dropout uses `e3nn.nn.Dropout` to mask every irreducible representation coefficient.
* Device-aware: pass `device` to the constructor; tensors and Linear weights
  are allocated accordingly.
"""

from __future__ import annotations

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
        node_type_idx: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        emb = self.elem_emb(node_type_idx)
        h = self.lin(emb)
        h = self.nl(h)
        h = self.dropout(h)
        return h


# --------------------------------------------------------------------------- #
class EdgeEncoder(nn.Module):
    """
    Encode per-edge raw features into hidden irreps.

    Inputs:
      - edge_type_idx: LongTensor[E] of edge-type indices (n_edge_types).
      - length_emb: Tensor[E, n_radial] radial distance embeddings.
      - sh: Tensor[E, sh_irreps.dim] spherical harmonics coefficients.

    Combines:
      1. Learned edge-type embedding.
      2. MLP over radial embeddings.
      3. Projection of spherical harmonics.

    Followed by equivariant nonlinearity and dropout to produce
    Tensor[E, out_irreps.dim].
    """

    def __init__(
        self,
        n_edge_types: int,
        n_radial: int,
        sh_irreps: Irreps,
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

        scalar_input_ir = Irreps(f"{sc_width * 2}x0e")
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
        edge_type_idx: torch.Tensor,
        length_emb: torch.Tensor,
        sh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return hidden edge features: Tensor[E, out_irreps.dim].
        """
        scalars = [
            self.edge_emb(edge_type_idx),
            self.radial_net(length_emb),
        ]

        h_scalar = self.lin_scalar(torch.cat(scalars, dim=-1))
        h = h_scalar + self.sh_proj(sh)
        h = self.nl(h)
        h = self.dropout(h)
        return h
