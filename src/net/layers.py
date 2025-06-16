"""
layers.py
~~~~~~~~~

E(3)-equivariant message-passing blocks **v2**:

*   uses `torch_scatter.scatter` for edge→node aggregation
*   dropout is `e3nn.nn.Dropout` (acts on *all* irrep coeffs)
*   normalisation + activation selected via `make_nonlinearity`
*   optional equivariant `BatchNorm` (`hp.batch_norm`)
"""

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter

from e3nn.o3 import Irreps, Linear
from e3nn.nn import Dropout, BatchNorm

from net.common import HyperParams
from net.activations import make_nonlinearity


# ════════════════════════════════════════════════════════════════════════
# Edge update
# ════════════════════════════════════════════════════════════════════════
class EdgeUpdateBlock(nn.Module):
    def __init__(
        self,
        hidden_irreps: Irreps,
        hp: HyperParams,
        *,
        dtype=torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.hp = hp

        self.lin_src = Linear(hidden_irreps, hidden_irreps)
        self.lin_dst = Linear(hidden_irreps, hidden_irreps)

        self.norm_act = make_nonlinearity(hidden_irreps, hp)
        self.dropout = (
            Dropout(hidden_irreps, p=hp.dropout) if hp.dropout > 0.0 else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        upd = 0.5 * (self.lin_src(node[src]) + self.lin_dst(node[dst]))
        if self.hp.residual_connections:
            upd = upd + edge
        upd = self.norm_act(upd)
        upd = self.dropout(upd)
        return upd


# ════════════════════════════════════════════════════════════════════════
# Node update
# ════════════════════════════════════════════════════════════════════════
class NodeUpdateBlock(nn.Module):
    def __init__(
        self,
        hidden_irreps: Irreps,
        hp: HyperParams,
        *,
        dtype=torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.hp = hp

        self.lin_msg = Linear(hidden_irreps, hidden_irreps)

        if hp.use_self_update:
            self.self_mlp = nn.Sequential(
                Linear(hidden_irreps, hidden_irreps),
                make_nonlinearity(hidden_irreps, hp),
            )
        else:
            self.self_mlp = None

        self.norm_act = make_nonlinearity(hidden_irreps, hp)
        self.dropout = (
            Dropout(hidden_irreps, p=hp.dropout) if hp.dropout > 0.0 else nn.Identity()
        )

        self.bn = BatchNorm(hidden_irreps) if hp.batch_norm else nn.Identity()

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        msg = self.lin_msg(edge)
        agg = scatter(msg, dst, dim=0, dim_size=node.size(0), reduce="sum")

        upd = agg
        if self.self_mlp is not None:
            upd = upd + self.self_mlp(node)
        if self.hp.residual_connections:
            upd = upd + node

        upd = self.bn(upd)
        upd = self.norm_act(upd)
        upd = self.dropout(upd)
        return upd


# ════════════════════════════════════════════════════════════════════════
# Message block
# ════════════════════════════════════════════════════════════════════════
class MessageBlock(nn.Module):
    def __init__(
        self,
        hidden_irreps: Irreps,
        hp: HyperParams,
        *,
        dtype=torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if hp.use_edge_updates:
            self.edge_upd = EdgeUpdateBlock(
                hidden_irreps, hp, dtype=dtype, device=device
            )
        else:
            self.edge_upd = None
        self.node_upd = NodeUpdateBlock(hidden_irreps, hp, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    def forward(self, node, edge, edge_index):
        if self.edge_upd is not None:
            edge = self.edge_upd(node, edge, edge_index)
        node = self.node_upd(node, edge, edge_index)
        return node, edge
