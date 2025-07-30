"""
Encoders for node and edge raw features → hidden irreps.

Features
--------
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
from collections import OrderedDict

from net.common import Config, RadialMLP
from net.activations import make_nonlinearity


def _magnitude_splits(
    features: torch.Tensor,
    irreps: Irreps,
) -> dict[str, torch.Tensor]:
    """
    Split features by irrep and compute magnitude per irreducible component.
    Returns a mapping from angular momentum l to a 1D tensor of magnitudes.
    """
    mags: dict[str, torch.Tensor] = OrderedDict()
    # features: (M, D)
    start = 0
    for mul, ir in irreps:
        dim = ir.dim
        size = mul * dim
        # slice for this irrep
        chunk = features[:, start : start + size]
        # reshape to (M * mul, dim)
        if mul > 0 and dim > 0:
            reshaped = chunk.reshape(-1, dim)
            # magnitude across dim
            mag = torch.linalg.norm(reshaped, dim=1)
            mags[f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"] = mag
        start += size
    return mags


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
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.out_irreps = out_irreps
        self.info = info

        scalar_width = cfg.hidden_base_dim
        self.elem_emb = nn.Embedding(
            node_one_hot_dim,
            scalar_width,
        )
        nn.init.normal_(self.elem_emb.weight, std=0.2)

        self.lin = Linear(
            Irreps(f"{scalar_width}x0e"),
            out_irreps,
            internal_weights=True,
        )

        self.nl = make_nonlinearity(out_irreps, cfg)
        self.dropout = (
            Dropout(out_irreps, p=cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        node_type_idx: torch.Tensor,  # (N,)
        activation_mags: dict = None,
    ) -> torch.Tensor:
        emb = self.elem_emb(node_type_idx)
        h = self.lin(emb)
        h = self.nl(h)
        h = self.dropout(h)
        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_{self.info['name']}"
            splits = _magnitude_splits(h, self.out_irreps)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag
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
        out_irreps: Irreps,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.out_irreps = out_irreps
        self.sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
        self.info = info

        # 1) scalar embeddings ------------------------------------------------
        sc_width = self.cfg.hidden_base_dim
        self.edge_emb = nn.Embedding(
            n_edge_types,
            sc_width,
            dtype=self.cfg.dtype,
        )
        nn.init.normal_(self.edge_emb.weight, std=0.2)

        self.radial_net = RadialMLP(
            in_dim=self.cfg.n_radial,
            out_dim=sc_width,
            layers=self.cfg.radial_layers,
            act=self.cfg.activation_scalar,
            dtype=self.cfg.dtype,
        )

        scalar_input_ir = Irreps(f"{sc_width * 2}x0e")
        self.lin_scalar = Linear(
            scalar_input_ir,
            out_irreps,
            internal_weights=True,
        )

        # 2) spherical harmonics projector ------------------------------------
        self.sh_irreps = self.sh_irreps
        self.sh_proj = Linear(
            self.sh_irreps,
            out_irreps,
            internal_weights=True,
        )

        # 3) non-linearity + dropout
        self.nl = make_nonlinearity(out_irreps, cfg)
        self.dropout = (
            Dropout(out_irreps, p=cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        edge_type_idx: torch.Tensor,
        length_emb: torch.Tensor,
        sh: torch.Tensor,
        activation_mags: dict = None,
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
        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_{self.info['name']}"
            splits = _magnitude_splits(h, self.out_irreps)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag
        return h
