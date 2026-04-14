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
from e3nn.o3 import Irreps, TensorSquare

from core.block_irrep_mapper import BlockIrrepMapper

from net.common import Config, E3MLP


class DeepHead(nn.Module):
    """
    Split readout head with separate projection paths for:
    - diagonal zero-shift self edges
    - shifted-self edges (optional)
    - off-diagonal edges
    """

    def __init__(
        self,
        irreps_diag_in: Irreps,
        irreps_edge_in: Irreps,
        irreps_neck: Irreps,
        pair_keys: List[str],
        mapper: BlockIrrepMapper,
        cfg: Config,
        info: dict = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.irreps_diag_in = Irreps(irreps_diag_in)
        self.irreps_edge_in = Irreps(irreps_edge_in)
        self.irreps_neck = Irreps(irreps_neck)
        self.pair_keys = pair_keys
        self.info = info

        self.mapper: BlockIrrepMapper = mapper
        self.separate_shifted_self = bool(self.cfg.separate_shifted_self)
        self.use_node_embeddings_for_self_edges = bool(
            self.cfg.head_use_node_embeddings_for_self_edges
        )
        self.use_tensor_square = bool(self.cfg.head_use_tensor_square)

        self.diag_trunk = E3MLP(
            self.irreps_diag_in,
            self.irreps_diag_in,
            self.irreps_neck,
            self.cfg.neck_depth,
            self.cfg,
            activate_last=True,
        )
        self.offdiag_trunk = E3MLP(
            self.irreps_edge_in,
            self.irreps_edge_in,
            self.irreps_neck,
            self.cfg.neck_depth,
            self.cfg,
            activate_last=True,
        )
        self.shifted_self_trunk = None
        if self.separate_shifted_self:
            self.shifted_self_trunk = E3MLP(
                self.irreps_edge_in,
                self.irreps_edge_in,
                self.irreps_neck,
                self.cfg.neck_depth,
                self.cfg,
                activate_last=True,
            )

        self.diag_tensor_square = None
        self.offdiag_tensor_square = None
        self.shifted_self_tensor_square = None
        if self.use_tensor_square:
            self.diag_tensor_square = TensorSquare(
                self.irreps_neck, irreps_out=self.irreps_neck
            )
            self.offdiag_tensor_square = TensorSquare(
                self.irreps_neck, irreps_out=self.irreps_neck
            )
            if self.separate_shifted_self:
                self.shifted_self_tensor_square = TensorSquare(
                    self.irreps_neck, irreps_out=self.irreps_neck
                )

        self.diag_projs = nn.ModuleDict()
        self.offdiag_projs = nn.ModuleDict()
        self.shifted_self_projs = nn.ModuleDict()
        for key in pair_keys:
            self.diag_projs[key] = E3MLP(
                self.irreps_neck,
                self.irreps_neck,
                mapper.get_pair_irreps(key),
                self.cfg.head_depth,
                self.cfg,
                activate_last=False,
            )
            self.offdiag_projs[key] = E3MLP(
                self.irreps_neck,
                self.irreps_neck,
                mapper.get_pair_irreps(key),
                self.cfg.head_depth,
                self.cfg,
                activate_last=False,
            )
            if self.separate_shifted_self:
                self.shifted_self_projs[key] = E3MLP(
                    self.irreps_neck,
                    self.irreps_neck,
                    mapper.get_pair_irreps(key),
                    self.cfg.head_depth,
                    self.cfg,
                    activate_last=False,
                )

        self.diag_log_scales = None
        self.offdiag_log_scales = None
        self.shifted_self_log_scales = None
        if self.cfg.head_use_mlp_log_scale:
            diag_log_scales = {}
            offdiag_log_scales = {}
            shifted_self_log_scales = {}
            for key in pair_keys:
                diag_log_scales[key] = E3MLP(
                    self.irreps_neck,
                    Irreps("64x0e"),
                    Irreps("1x0e"),
                    self.cfg.head_log_scale_mlp_n_layers,
                    self.cfg,
                )
                offdiag_log_scales[key] = E3MLP(
                    self.irreps_neck,
                    Irreps("64x0e"),
                    Irreps("1x0e"),
                    self.cfg.head_log_scale_mlp_n_layers,
                    self.cfg,
                )
                if self.separate_shifted_self:
                    shifted_self_log_scales[key] = E3MLP(
                        self.irreps_neck,
                        Irreps("64x0e"),
                        Irreps("1x0e"),
                        self.cfg.head_log_scale_mlp_n_layers,
                        self.cfg,
                    )
            self.diag_log_scales = nn.ModuleDict(diag_log_scales)
            self.offdiag_log_scales = nn.ModuleDict(offdiag_log_scales)
            if self.separate_shifted_self:
                self.shifted_self_log_scales = nn.ModuleDict(shifted_self_log_scales)

    # ------------------------------------------------------------------
    def forward(
        self,
        node_feat: torch.Tensor,  # (N, diag_in_dim)
        edge_feat: torch.Tensor,  # (E, edge_in_dim)
        edge_type_idx: torch.Tensor,  # (E,)  long, maps to self.pair_keys
        edges_5d: torch.Tensor,  # (5, E) = [sx, sy, sz, src, dst]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out_vec = defaultdict(list)
        out_edges = defaultdict(list)
        for idx, key in enumerate(self.pair_keys):
            mask = edge_type_idx == idx
            if torch.any(mask):
                selected_edges = edges_5d[:, mask]
                selected_edge_feat = edge_feat[mask]

                src = selected_edges[3]
                dst = selected_edges[4]
                is_same_atom = src == dst
                is_zero_shift = (selected_edges[:3] == 0).all(dim=0)
                is_diag = is_same_atom & is_zero_shift
                if self.separate_shifted_self:
                    is_shifted_self = is_same_atom & (~is_zero_shift)
                    is_offdiag = ~is_same_atom
                else:
                    is_shifted_self = torch.zeros_like(is_diag, dtype=torch.bool)
                    is_offdiag = ~is_diag

                if is_diag.any():
                    if self.use_node_embeddings_for_self_edges:
                        diag_input = node_feat[src[is_diag]]
                    else:
                        diag_input = selected_edge_feat[is_diag]
                    diag_hidden = self.diag_trunk(diag_input)
                    if self.diag_tensor_square is not None:
                        diag_hidden = self.diag_tensor_square(diag_hidden)
                    diag_vectors = self.diag_projs[key](diag_hidden)
                    if self.diag_log_scales is not None:
                        diag_vectors = (
                            torch.exp(self.diag_log_scales[key](diag_hidden))
                            * diag_vectors
                        )
                    out_vec[key].append(diag_vectors)
                    out_edges[key].append(selected_edges[:, is_diag])

                if is_shifted_self.any():
                    shifted_hidden = self.shifted_self_trunk(
                        selected_edge_feat[is_shifted_self]
                    )
                    if self.shifted_self_tensor_square is not None:
                        shifted_hidden = self.shifted_self_tensor_square(shifted_hidden)
                    shifted_vectors = self.shifted_self_projs[key](shifted_hidden)
                    if self.shifted_self_log_scales is not None:
                        shifted_vectors = (
                            torch.exp(self.shifted_self_log_scales[key](shifted_hidden))
                            * shifted_vectors
                        )
                    out_vec[key].append(shifted_vectors)
                    out_edges[key].append(selected_edges[:, is_shifted_self])

                if is_offdiag.any():
                    offdiag_hidden = self.offdiag_trunk(selected_edge_feat[is_offdiag])
                    if self.offdiag_tensor_square is not None:
                        offdiag_hidden = self.offdiag_tensor_square(offdiag_hidden)
                    offdiag_vectors = self.offdiag_projs[key](offdiag_hidden)
                    if self.offdiag_log_scales is not None:
                        offdiag_vectors = (
                            torch.exp(self.offdiag_log_scales[key](offdiag_hidden))
                            * offdiag_vectors
                        )
                    out_vec[key].append(offdiag_vectors)
                    out_edges[key].append(selected_edges[:, is_offdiag])

        result = {}
        for key in out_vec:
            result[key] = {
                "vectors": torch.cat(out_vec[key], dim=0),
                "edges": torch.cat(out_edges[key], dim=1),
            }
        return result
