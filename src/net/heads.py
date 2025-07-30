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
from e3nn.nn import Dropout

from core.block_irrep_mapper import BlockIrrepMapper

from net.common import Config
from net.activations import make_nonlinearity


class DeepHead(nn.Module):
    """
    Shared trunk + per-pair final Linear → irrep vectors.
    """

    def __init__(
        self,
        irreps_hidden: Irreps,
        pair_keys: List[str],
        mapper: BlockIrrepMapper,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.irreps_hidden = irreps_hidden
        self.pair_keys = pair_keys
        self.info = info

        # 0) shared mapper
        self.mapper: BlockIrrepMapper = mapper

        # 1) deep trunk ---------------------------------------------------
        layers: List[nn.Module] = []
        for _ in range(self.cfg.neck_depth):
            lin = Linear(irreps_hidden, irreps_hidden)
            layers.append(lin)
            layers.append(make_nonlinearity(irreps_hidden, self.cfg))

            if self.cfg.dropout > 0.0:
                layers.append(Dropout(irreps_hidden, p=self.cfg.dropout))

        self.trunk = nn.Sequential(*layers)

        # 2) last-mile MLP per pair -----------------------------------
        last = {}
        for key in pair_keys:
            layers = []
            for i in range(self.cfg.head_depth - 1):
                layers.append(Linear(irreps_hidden, irreps_hidden))
                layers.append(make_nonlinearity(irreps_hidden, self.cfg))
                if self.cfg.dropout > 0.0:
                    layers.append(Dropout(irreps_hidden, p=self.cfg.dropout))
            # final layer is linear, no nonlinearity
            layers.append(Linear(irreps_hidden, mapper.get_pair_irreps(key)))
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
