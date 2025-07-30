"""
layers.py
~~~~~~~~~

E(3)-equivariant message-passing blocks:

*   uses `torch_scatter.scatter` for edge→node aggregation
*   dropout is `e3nn.nn.Dropout` (acts on *all* irrep coeffs)
*   normalisation + activation selected via `make_nonlinearity`
*   optional equivariant `BatchNorm` (`cfg.batch_norm`)
"""

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter, scatter_softmax
from collections import OrderedDict

from e3nn.o3 import Irreps, Linear
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Dropout, BatchNorm

from net.common import Config, split_into_three
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


# ════════════════════════════════════════════════════════════════════════
# Edge update
# ════════════════════════════════════════════════════════════════════════
class EdgeUpdateBlock(nn.Module):
    """
    Compute per-edge updates from source and destination node features.

    For each edge (i → j):
      • Apply two linear transforms to node[i] and node[j].
      • Average (and optionally add residual edge features).
      • Apply equivariant nonlinearity and dropout.
    """

    def __init__(
        self,
        hidden_irreps: Irreps,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info
        if self.cfg.edge_update_node_combine == "tensor_product":
            self.tp = FullyConnectedTensorProduct(
                hidden_irreps, hidden_irreps, hidden_irreps
            )
        if self.cfg.edge_update == "concat":
            self.lin = Linear(hidden_irreps + hidden_irreps, hidden_irreps)
        else:
            self.lin = Linear(hidden_irreps, hidden_irreps)

        self.norm_act = make_nonlinearity(hidden_irreps, cfg)
        if self.cfg.dropout > 0.0:
            self.dropout = Dropout(hidden_irreps, p=self.cfg.dropout)

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index

        if self.cfg.edge_update_node_combine == "tensor_product":
            msg = self.tp(node[src], node[dst])
        elif self.cfg.edge_update_node_combine == "sum":
            msg = node[src] + node[dst]  # this is symmetric

        if self.cfg.edge_update == "residual":
            msg = self.lin(msg)
            msg = self.norm_act(msg)
            if self.dropout:
                msg = self.dropout(msg)
            msg = msg + edge
        else:
            if self.cfg.edge_update == "concat":
                msg = torch.cat([msg, edge], dim=-1)
            elif self.cfg.edge_update == "replace":
                msg = edge
            else:
                raise ValueError(f"Unknown edge update type: {self.cfg.edge_update}")
            msg = self.lin(msg)
            msg = self.norm_act(msg)
            if self.dropout:
                msg = self.dropout(msg)
        return msg


# ════════════════════════════════════════════════════════════════════════
# Node update
# ════════════════════════════════════════════════════════════════════════
class NodeUpdateBlock(nn.Module):
    """
    Aggregate edge messages to update node features.

    • Linear transform of incoming edge features.
    • Scatter-sum aggregation by destination node.
    • Optional self-MLP and residual connection.
    • Batch normalization, nonlinearity, and dropout.
    """

    def __init__(
        self,
        hidden_irreps: Irreps,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.cfg = cfg
        self.info = info

        if cfg.node_update == "concat":
            self.lin = Linear(hidden_irreps + hidden_irreps, hidden_irreps)
        else:
            self.lin = Linear(hidden_irreps, hidden_irreps)

        if self.cfg.node_update_use_attention:
            self.attn = Linear(
                hidden_irreps, (hidden_irreps + hidden_irreps).simplify()
            )

        self.norm_act = make_nonlinearity(hidden_irreps, self.cfg)

        if self.cfg.dropout > 0.0:
            self.dropout = Dropout(hidden_irreps, p=self.cfg.dropout)
        if self.cfg.batch_norm:
            self.bn = BatchNorm(hidden_irreps, affine=True)

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        # attention
        if self.cfg.node_update_use_attention:
            kqv = self.attn(edge)
            k, q, v = split_into_three(kqv, self.attn.irreps_out)
            # compute attention scores
            attn_scores = torch.einsum("ie,ie->i", k, q)
            # apply softmax to get attention weights
            attn_weights = scatter_softmax(
                attn_scores, dst, dim=0, dim_size=node.size(0)
            )
            # apply attention weights to values
            edge = v * attn_weights.unsqueeze(-1)

        upd = scatter(edge, dst, dim=0, dim_size=node.size(0), reduce="sum")
        if self.cfg.node_update == "residual":
            upd = self.lin(upd)
            if self.bn:
                upd = self.bn(upd)
            upd = self.norm_act(upd)
            if self.dropout > 0.0:
                upd = self.dropout(upd)
            upd = upd + node
        else:
            if self.cfg.node_update == "replace":
                upd = self.lin(upd)
            elif self.cfg.node_update == "concat":
                upd = torch.cat([upd, node], dim=-1)
                upd = self.lin(upd)
            else:
                raise ValueError(f"Unknown node update type: {self.cfg.node_update}")
            if self.bn:
                upd = self.bn(upd)
            upd = self.norm_act(upd)
            if self.dropout > 0.0:
                upd = self.dropout(upd)
        return upd


# ════════════════════════════════════════════════════════════════════════
# Message block
# ════════════════════════════════════════════════════════════════════════
class MessageBlock(nn.Module):
    """
    One message-passing step: optional edge update followed by node update.

    Args:
      node: Tensor of shape (N, hidden_dim)
      edge: Tensor of shape (E, hidden_dim)
      edge_index: LongTensor of shape (2, E) with source/dest indices

    Returns:
      Tuple (node_updated, edge_updated) of same shapes.
    """

    def __init__(
        self,
        hidden_irreps: Irreps,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info or {}
        self.hidden_irreps = hidden_irreps
        self.edge_upd = EdgeUpdateBlock(hidden_irreps, cfg, info=info)
        self.node_upd = NodeUpdateBlock(hidden_irreps, cfg, info=info)

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index, activation_mags: dict = None):
        edge = self.edge_upd(node, edge, edge_index)
        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_edge_{self.info['graph']}_layer_{self.info['layer']}"
            splits = _magnitude_splits(edge, self.hidden_irreps)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag

        node = self.node_upd(node, edge, edge_index)
        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_node_{self.info['graph']}_layer_{self.info['layer']}"
            splits = _magnitude_splits(node, self.hidden_irreps)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag
        return node, edge
