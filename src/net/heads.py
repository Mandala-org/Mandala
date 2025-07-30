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

from net.common import Config
from net.activations import make_nonlinearity


class DeepHead(nn.Module):
    """
    Shared trunk + per-pair final Linear → irrep vectors.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        pair_keys: List[str],
        mapper: BlockIrrepMapper,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.irreps_in = irreps_in
        self.pair_keys = pair_keys
        self.info = info

        # 0) shared mapper
        self.mapper: BlockIrrepMapper = mapper

        # 1) deep trunk ---------------------------------------------------
        layers: List[nn.Module] = []
        cur_ir = irreps_in
        for _ in range(cfg.head_depth):
            # scale multiplicities by hidden_mul
            parts = []
            for mul, ir in cur_ir:
                mul_new = int(round(mul * cfg.head_hidden_mul))
                parts.append((mul_new, ir))
            next_ir = Irreps(parts).simplify()

            lin = Linear(cur_ir, next_ir)
            layers.append(lin)

            if cfg.batch_norm:
                layers.append(BatchNorm(next_ir))

            layers.append(make_nonlinearity(next_ir, cfg))

            if cfg.dropout > 0.0:
                layers.append(Dropout(next_ir, p=cfg.dropout))

            cur_ir = next_ir

        self.trunk = nn.Sequential(*layers)
        self.trunk_irreps_out = cur_ir

        # 2) last-mile Linear per pair -----------------------------------
        last = {}
        for key in pair_keys:
            el_a, el_b = key.split("-")
            out_ir = self.mapper._maps[(el_a, el_b)].rtp.irreps_out
            proj = Linear(self.trunk_irreps_out, out_ir)
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
