"""
layers.py
~~~~~~~~~

E(3)-equivariant message-passing blocks:

*   uses `torch_scatter.scatter` for edge→node aggregation
*   dropout is `e3nn.nn.Dropout` (acts on *all* irrep coeffs)
*   normalisation + activation selected via `make_nonlinearity`
"""

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter, scatter_softmax
from collections import OrderedDict

from e3nn.o3 import Irreps
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Dropout

from net.common import Config, split_into_three, E3MLP
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
        initial: Irreps | None = None,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info

        if initial:
            node_irreps = initial
        else:
            node_irreps = hidden_irreps

        if self.cfg.edge_update_node_combine == "concat":
            pre_lin_input_irreps = node_irreps + node_irreps
        else:
            pre_lin_input_irreps = node_irreps
        self.pre_lin = E3MLP(
            pre_lin_input_irreps,
            node_irreps,
            node_irreps,
            self.cfg.edge_update_pre_lin_mlp_n_layers,
            self.cfg,
        )

        self.tp = FullyConnectedTensorProduct(
            node_irreps, hidden_irreps, hidden_irreps, internal_weights=True
        )
        if self.cfg.edge_update == "concat":
            post_lin_input_irreps = node_irreps + hidden_irreps
        elif self.cfg.edge_update == "replace":
            post_lin_input_irreps = node_irreps
        else:
            post_lin_input_irreps = hidden_irreps
        self.post_lin = E3MLP(
            post_lin_input_irreps,
            hidden_irreps,
            hidden_irreps,
            self.cfg.edge_update_post_lin_mlp_n_layers,
            self.cfg,
        )

        self.norm_act = make_nonlinearity(hidden_irreps, cfg)

        if self.cfg.dropout > 0.0:
            self.dropout = Dropout(hidden_irreps, p=self.cfg.dropout)
        else:
            self.dropout = None

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        edge_old = edge

        if self.cfg.edge_update_node_combine == "concat":
            edge = torch.cat([node[src], node[dst]], dim=-1)
        elif self.cfg.edge_update_node_combine == "sum":
            edge = node[src] + node[dst]  # this is symmetric

        edge = self.pre_lin(edge)

        if self.cfg.edge_update == "tensor_product":
            edge = self.tp(edge, edge_old)
        elif self.cfg.edge_update == "concat":
            edge = torch.cat([edge, edge_old], dim=-1)
        elif self.cfg.edge_update == "replace":
            pass
        else:
            raise ValueError(f"Unknown edge update type: {self.cfg.edge_update}")

        edge = self.post_lin(edge)

        edge = self.norm_act(edge)

        if self.dropout:
            edge = self.dropout(edge)

        if self.cfg.edge_update_residual:
            edge = edge + edge_old

        return edge


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
        initial: Irreps | None = None,
        info: dict = None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.cfg = cfg
        self.initial = initial
        self.info = info

        if initial:
            node_irreps = initial
        else:
            node_irreps = hidden_irreps

        self.pre_lin = E3MLP(
            hidden_irreps,
            hidden_irreps,
            hidden_irreps,
            self.cfg.node_update_pre_lin_mlp_n_layers,
            self.cfg,
        )

        if self.cfg.node_update_message_agg == "attention":
            attn_output_irreps = (
                (hidden_irreps + hidden_irreps).sort().irreps.simplify()
            )
            self.attn = E3MLP(
                hidden_irreps,
                hidden_irreps,
                attn_output_irreps,
                self.cfg.node_update_attention_mlp_n_layers,
                self.cfg,
            )
        elif self.cfg.node_update_message_agg == "sum":
            pass
        else:
            raise ValueError(
                f"Unknown node update message aggregation: {self.cfg.node_update_message_agg}"
            )

        self.tp = FullyConnectedTensorProduct(
            node_irreps, hidden_irreps, hidden_irreps, internal_weights=True
        )

        if cfg.node_update == "concat":
            post_lin_input_irreps = node_irreps + hidden_irreps
        else:
            post_lin_input_irreps = hidden_irreps
        self.post_lin = E3MLP(
            post_lin_input_irreps,
            hidden_irreps,
            hidden_irreps,
            self.cfg.node_update_post_lin_mlp_n_layers,
            self.cfg,
        )

        self.norm_act = make_nonlinearity(hidden_irreps, self.cfg)

        if self.cfg.dropout > 0.0:
            self.dropout = Dropout(hidden_irreps, p=self.cfg.dropout)
        else:
            self.dropout = None

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        node_old = node

        # attention
        if self.cfg.node_update_message_agg == "attention":
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

        elif self.cfg.node_update_message_agg == "sum":
            edge = self.pre_lin(edge)

        # aggregate edge messages to nodes
        agg_msg = scatter(edge, dst, dim=0, dim_size=node.size(0), reduce="sum")

        if self.cfg.node_update == "concat":
            node = torch.cat([node_old, agg_msg], dim=-1)

        elif self.cfg.node_update == "tensor_product":
            node = self.tp(node_old, agg_msg)

        elif self.cfg.node_update == "replace" or (
            self.initial and self.cfg.node_update == "sum"
        ):
            node = agg_msg

        elif self.cfg.node_update == "sum":
            node = node_old + agg_msg

        else:
            raise ValueError(f"Unknown node update type: {self.cfg.node_update}")

        node = self.post_lin(node)

        node = self.norm_act(node)

        if self.dropout:
            node = self.dropout(node)

        if self.cfg.node_update_residual and not self.initial:
            node = node + node_old

        return node


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
        initial: Irreps | None = None,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info or {}
        self.hidden_irreps = hidden_irreps
        self.edge_upd = EdgeUpdateBlock(hidden_irreps, cfg, initial, info=info)
        self.node_upd = NodeUpdateBlock(hidden_irreps, cfg, initial, info=info)

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
