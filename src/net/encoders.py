"""
Encoders for node and edge raw features -> hidden irreps.

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
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, TensorSquare
from collections import OrderedDict

from net.common import Config, smooth_cutoff
from net.layer_norm import E3LayerNorm


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
    Node features = *only* element one-hot -> learned embedding (scalars).

    The embedding is mapped (Linear) to `irreps_out` (scalars only in our
    current setup, but we keep it general).
    """

    def __init__(
        self,
        node_one_hot_dim: int,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info
        self.irreps_out = Irreps(f"{self.cfg.hidden_base_dim}x0e")

        if self.irreps_out.lmax > 0:
            raise ValueError("NodeEncoder can only output scalar irreps (l=0).")

        self.elem_emb = nn.Embedding(
            node_one_hot_dim,
            self.cfg.hidden_base_dim,
        )
        nn.init.normal_(self.elem_emb.weight, std=0.2)

    # ------------------------------------------------------------------
    def forward(
        self,
        node_type_idx: torch.Tensor,  # (N,)
        activation_mags: dict = None,
    ) -> torch.Tensor:
        emb = self.elem_emb(node_type_idx)

        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_{self.info['name']}"
            splits = _magnitude_splits(emb, self.irreps_out)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag
        return emb


# --------------------------------------------------------------------------- #
class EdgeEncoder(nn.Module):
    """
    Encode per-edge raw features into hidden irreps.

    Inputs:
      - edge_type_idx: LongTensor[E] of edge-type indices (n_edge_types).
      - length_emb: Tensor[E, n_radial] radial distance embeddings.
      - sh: Tensor[E, sh_irreps.dim] spherical harmonics coefficients.
      - edge_length: Tensor[E] physical distances used by the smooth cutoff.

    Combines:
      1. Learned edge-type embedding.
      2. Radial distance embeddings.
      3. Spherical harmonics coefficients.

    Tensor[E, irreps_out.dim].
    """

    def __init__(
        self,
        n_edge_types: int,
        irreps_out: Irreps,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.style = self.cfg.edge_encoder_style.lower()
        self.sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
        self.info = info

        if self.style == "rich":
            self.irreps_out = irreps_out
            self.edge_emb = nn.Embedding(
                n_edge_types,
                self.cfg.edge_type_emb_dim,
                dtype=self.cfg.dtype,
            )
            nn.init.normal_(self.edge_emb.weight, std=0.2)
            self.sh_tensor_square = None
            sh_tp_irreps = self.sh_irreps
            if self.cfg.edge_encoder_use_sh_tensor_square:
                self.sh_tensor_square = TensorSquare(
                    self.sh_irreps, irreps_out=self.irreps_out
                )
                sh_tp_irreps = self.sh_tensor_square.irreps_out
            self.tp = FullyConnectedTensorProduct(
                Irreps(f"{self.cfg.edge_type_emb_dim}x0e"),
                (Irreps(f"{self.cfg.n_radial}x0e") + sh_tp_irreps).simplify(),
                self.irreps_out,
                internal_weights=True,
            )
            self.distance_proj = None
        elif self.style == "distance":
            self.irreps_out = Irreps(f"{self.cfg.hidden_base_dim}x0e")
            self.edge_emb = None
            self.tp = None
            self.distance_proj = nn.Linear(
                self.cfg.n_radial,
                self.cfg.hidden_base_dim,
                dtype=self.cfg.dtype,
            )
        else:
            raise ValueError(
                f"Unknown edge_encoder_style '{self.cfg.edge_encoder_style}'. "
                "Expected 'rich' or 'distance'."
            )
        self.norm = E3LayerNorm(self.irreps_out) if self.cfg.e3layernorm else None

    # ------------------------------------------------------------------
    def forward(
        self,
        edge_type_idx: torch.Tensor,
        length_emb: torch.Tensor,
        sh: torch.Tensor,
        edge_length: torch.Tensor,
        activation_mags: dict = None,
    ) -> torch.Tensor:
        """
        Return hidden edge features: Tensor[E, irreps_out.dim].
        """
        if self.style == "rich":
            type_emb = self.edge_emb(edge_type_idx)
            if self.sh_tensor_square is not None:
                sh = self.sh_tensor_square(sh)
            disp_emb = torch.cat([length_emb, sh], dim=-1)
            emb = self.tp(type_emb, disp_emb)
        else:
            emb = self.distance_proj(length_emb)

        if self.norm is not None:
            emb = self.norm(emb)

        emb = emb * smooth_cutoff(edge_length, self.cfg.cutoff_radius).unsqueeze(-1)

        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_{self.info['name']}"
            splits = _magnitude_splits(emb, self.irreps_out)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag
        return emb
