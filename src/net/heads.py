"""
heads.py
~~~~~~~~

Deep, equivariant per-pair read-out heads.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

import torch
from torch import nn
from e3nn.o3 import Irreps, Linear
from e3nn.nn import Dropout, BatchNorm

from core.block_irrep_mapper import BlockIrrepMapper

from net.common import HyperParams
from net.activations import make_nonlinearity


class DeepHead(nn.Module):
    """
    Shared trunk + per-pair final Linear → irrep vectors.
    """

    def __init__(
        self,
        in_irreps: Irreps,
        pair_keys: List[str],
        mapper: BlockIrrepMapper,
        hp: HyperParams,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hp = hp
        self.in_irreps = in_irreps
        self.pair_keys = pair_keys

        # 0) shared mapper – **passed in**, no local creation
        self.mapper: BlockIrrepMapper = mapper.to(self.device)

        # 1) deep trunk ---------------------------------------------------
        layers: List[nn.Module] = []
        cur_ir = in_irreps
        for _ in range(max(1, hp.head_depth)):
            # scale multiplicities by hidden_mul (clip for safety)
            parts = []
            for mul, ir in cur_ir:
                mul_new = int(round(mul * hp.head_hidden_mul))
                mul_new = max(1, min(mul_new, int(mul * hp.hidden_mul_clip)))
                parts.append((mul_new, ir))
            next_ir = Irreps(parts).simplify()

            lin = Linear(cur_ir, next_ir)
            lin = lin.to(self.device)
            layers.append(lin)

            if hp.batch_norm:
                layers.append(BatchNorm(next_ir).to(self.device))

            layers.append(make_nonlinearity(next_ir, hp))

            if hp.dropout > 0.0:
                layers.append(Dropout(next_ir, p=hp.dropout))

            cur_ir = next_ir

        self.trunk = nn.Sequential(*layers)
        self.trunk_out_irreps = cur_ir

        # 2) last-mile Linear per pair -----------------------------------
        last = {}
        for key in pair_keys:
            el_a, el_b = key.split("-")
            out_ir = self.mapper._maps[(el_a, el_b)].rtp.irreps_out
            proj = Linear(self.trunk_out_irreps, out_ir).to(self.device)
            last[key] = proj
        self.last_mlps = nn.ModuleDict(last)

    # ------------------------------------------------------------------
    def forward(
        self,
        edge_feat: torch.Tensor,  # (E, trunk_in_dim)
        edge_type_idx: torch.Tensor,  # (E,)  long, maps to self.pair_keys
        edge_index: torch.Tensor,  # (2, E)
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        h = self.trunk(edge_feat)

        out_vec = defaultdict(list)
        out_edges = defaultdict(list)
        for idx, key in enumerate(self.pair_keys):
            mask = edge_type_idx == idx
            if torch.any(mask):
                vecs = self.last_mlps[key](h[mask])
                out_vec[key].append(vecs)
                out_edges[key].append(edge_index[:, mask])

        result = {}
        for key in out_vec:
            result[key] = {
                "vectors": torch.cat(out_vec[key], dim=0),
                "edges": torch.cat(out_edges[key], dim=1),
            }
        return result
