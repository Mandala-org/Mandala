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

from torch import nn
from torch_scatter import scatter

from e3nn.o3 import Irreps, Linear
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Dropout, BatchNorm

from net.common import Config
from net.activations import make_nonlinearity


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
    ):
        super().__init__()
        self.cfg = cfg

        self.lin_src = Linear(hidden_irreps, hidden_irreps)
        self.lin_dst = Linear(hidden_irreps, hidden_irreps)
        self.tp = FullyConnectedTensorProduct(
            hidden_irreps, hidden_irreps, hidden_irreps
        )

        self.norm_act = make_nonlinearity(hidden_irreps, cfg)
        self.dropout = (
            Dropout(hidden_irreps, p=cfg.dropout)
            if cfg.dropout > 0.0
            else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        # upd = 0.5 * (self.lin_src(node[src]) + self.lin_dst(node[dst]))
        msg_src = self.lin_src(node[src])
        msg_dst = self.lin_dst(node[dst])
        upd = self.tp(msg_src, msg_dst)
        if self.cfg.residual_connections:
            upd = upd + edge
        upd = self.norm_act(upd)
        upd = self.dropout(upd)
        return upd


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
    ):
        super().__init__()
        self.cfg = cfg

        self.lin_msg = Linear(hidden_irreps, hidden_irreps)

        if cfg.use_self_update:
            self.self_mlp = nn.Sequential(
                Linear(hidden_irreps, hidden_irreps),
                make_nonlinearity(hidden_irreps, cfg),
            )
        else:
            self.self_mlp = None

        self.norm_act = make_nonlinearity(hidden_irreps, cfg)
        self.dropout = (
            Dropout(hidden_irreps, p=cfg.dropout)
            if cfg.dropout > 0.0
            else nn.Identity()
        )

        self.bn = BatchNorm(hidden_irreps) if cfg.batch_norm else nn.Identity()

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        msg = self.lin_msg(edge)
        agg = scatter(msg, dst, dim=0, dim_size=node.size(0), reduce="sum")

        upd = agg
        if self.self_mlp is not None:
            upd = upd + self.self_mlp(node)
        if self.cfg.residual_connections:
            upd = upd + node

        upd = self.bn(upd)
        upd = self.norm_act(upd)
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
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.use_edge_updates:
            self.edge_upd = EdgeUpdateBlock(hidden_irreps, cfg)
        else:
            self.edge_upd = None
        self.node_upd = NodeUpdateBlock(hidden_irreps, cfg)

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        if self.edge_upd is not None:
            edge = self.edge_upd(node, edge, edge_index)
        node = self.node_upd(node, edge, edge_index)
        return node, edge
