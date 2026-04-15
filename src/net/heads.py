"""
heads.py
~~~~~~~~

Deep, equivariant per-pair read-out heads.
"""

from __future__ import annotations
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
        head_variant = self.cfg.head_e3mlp_variant or self.cfg.e3mlp_variant
        head_layers = int(self.cfg.head_e3mlp_layers)

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
                head_layers,
                self.cfg,
                activate_last=False,
                variant=head_variant,
                post_scale=self.cfg.head_diag_output_scale,
            )
            self.offdiag_projs[key] = E3MLP(
                self.irreps_neck,
                self.irreps_neck,
                mapper.get_pair_irreps(key),
                head_layers,
                self.cfg,
                activate_last=False,
                variant=head_variant,
                post_scale=self.cfg.head_offdiag_output_scale,
            )
            if self.separate_shifted_self:
                self.shifted_self_projs[key] = E3MLP(
                    self.irreps_neck,
                    self.irreps_neck,
                    mapper.get_pair_irreps(key),
                    head_layers,
                    self.cfg,
                    activate_last=False,
                    variant=head_variant,
                    post_scale=self.cfg.head_diag_output_scale,
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
        pred_pair_edges_static: Dict[str, torch.Tensor],
        edge_partitions: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        result = {}
        for key in self.pair_keys:
            if key not in pred_pair_edges_static or key not in edge_partitions:
                continue
            key_edges = pred_pair_edges_static[key]
            parts = edge_partitions[key]
            global_idx = parts["global_idx"]
            key_edge_feat = edge_feat.index_select(0, global_idx)
            pair_vectors = key_edge_feat.new_zeros(
                (key_edges.shape[1], self.mapper.get_pair_irreps(key).dim)
            )

            diag_local_idx = parts["diag_local_idx"]
            if diag_local_idx.numel() > 0:
                if self.use_node_embeddings_for_self_edges:
                    diag_src = key_edges[3].index_select(0, diag_local_idx)
                    diag_input = node_feat.index_select(0, diag_src)
                else:
                    diag_input = key_edge_feat.index_select(0, diag_local_idx)
                diag_hidden = self.diag_trunk(diag_input)
                if self.diag_tensor_square is not None:
                    diag_hidden = self.diag_tensor_square(diag_hidden)
                diag_vectors = self.diag_projs[key](diag_hidden)
                if self.diag_log_scales is not None:
                    diag_vectors = (
                        torch.exp(self.diag_log_scales[key](diag_hidden)) * diag_vectors
                    )
                pair_vectors.index_copy_(0, diag_local_idx, diag_vectors)

            shifted_self_local_idx = parts["shifted_self_local_idx"]
            if shifted_self_local_idx.numel() > 0:
                shifted_input = key_edge_feat.index_select(0, shifted_self_local_idx)
                shifted_hidden = self.shifted_self_trunk(shifted_input)
                if self.shifted_self_tensor_square is not None:
                    shifted_hidden = self.shifted_self_tensor_square(shifted_hidden)
                shifted_vectors = self.shifted_self_projs[key](shifted_hidden)
                if self.shifted_self_log_scales is not None:
                    shifted_vectors = (
                        torch.exp(self.shifted_self_log_scales[key](shifted_hidden))
                        * shifted_vectors
                    )
                pair_vectors.index_copy_(0, shifted_self_local_idx, shifted_vectors)

            offdiag_local_idx = parts["offdiag_local_idx"]
            if offdiag_local_idx.numel() > 0:
                offdiag_input = key_edge_feat.index_select(0, offdiag_local_idx)
                offdiag_hidden = self.offdiag_trunk(offdiag_input)
                if self.offdiag_tensor_square is not None:
                    offdiag_hidden = self.offdiag_tensor_square(offdiag_hidden)
                offdiag_vectors = self.offdiag_projs[key](offdiag_hidden)
                if self.offdiag_log_scales is not None:
                    offdiag_vectors = (
                        torch.exp(self.offdiag_log_scales[key](offdiag_hidden))
                        * offdiag_vectors
                    )
                pair_vectors.index_copy_(0, offdiag_local_idx, offdiag_vectors)

            result[key] = pair_vectors
        return result
