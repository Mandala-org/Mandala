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
from e3nn.o3 import Irreps
from e3nn.o3 import TensorSquare

from core.block_irrep_mapper import BlockIrrepMapper

from net.common import Config, E3MLP


class DeepHead(nn.Module):
    """
    Shared trunk + per-pair final Linear → irrep vectors.
    """

    def __init__(
        self,
        irreps_hidden: Irreps,
        irreps_neck: Irreps,
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
        self.trunk = E3MLP(
            irreps_hidden,
            irreps_hidden,
            irreps_hidden,
            self.cfg.head_trunk_mlp_n_layers,
            self.cfg,
            activate_last=True,
        )
        self.tensor_square = TensorSquare(irreps_hidden, irreps_neck)

        # 2) last-mile MLP per pair -----------------------------------
        last = {}
        for key in pair_keys:
            last[key] = E3MLP(
                irreps_neck,
                irreps_neck,
                irreps_neck,
                self.cfg.head_last_mlp_n_layers,
                self.cfg,
                activate_last=True,
            )
        self.last_mlps = nn.ModuleDict(last)

        # 3) final projection
        final_proj = {}
        for key in pair_keys:
            final_proj[key] = E3MLP(
                irreps_neck,
                irreps_neck,
                mapper.get_pair_irreps(key),
                self.cfg.head_final_proj_mlp_n_layers,
                self.cfg,
            )
        self.final_projs = nn.ModuleDict(final_proj)

        # 4) log scale for each pair -----------------------------------
        if self.cfg.head_use_mlp_log_scale:
            self.log_scales = {}
            for key in pair_keys:
                log_scale_mlp = E3MLP(
                    irreps_neck,
                    Irreps("64x0e"),
                    Irreps("1x0e"),
                    self.cfg.head_log_scale_mlp_n_layers,
                    self.cfg,
                )
                self.log_scales[key] = log_scale_mlp
            self.log_scales = nn.ModuleDict(self.log_scales)
        else:
            self.log_scales = None

    # ------------------------------------------------------------------
    def forward(
        self,
        edge_feat: torch.Tensor,  # (E, trunk_in_dim)
        edge_type_idx: torch.Tensor,  # (E,)  long, maps to self.pair_keys
        edge_index: torch.Tensor,  # (2, E)
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        h = self.trunk(edge_feat)
        h = self.tensor_square(h)

        out_vec = defaultdict(list)
        out_edges = defaultdict(list)
        for idx, key in enumerate(self.pair_keys):
            mask = edge_type_idx == idx
            if torch.any(mask):
                vecs = self.last_mlps[key](h[mask])
                projected_vecs = self.final_projs[key](vecs)
                if self.log_scales is not None:
                    log_scale = self.log_scales[key](h[mask])
                    projected_vecs = torch.exp(log_scale) * projected_vecs
                out_vec[key].append(projected_vecs)
                out_edges[key].append(edge_index[:, mask])

        result = {}
        for key in out_vec:
            result[key] = {
                "vectors": torch.cat(out_vec[key], dim=0),
                "edges": torch.cat(out_edges[key], dim=1),
            }
        return result
