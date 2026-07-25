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
        self.head_pair_mode = str(getattr(self.cfg, "head_pair_mode", "split")).lower()
        if self.head_pair_mode not in {"split", "shared_conditioned"}:
            raise ValueError(
                "head_pair_mode must be one of 'split' or 'shared_conditioned'."
            )
        head_variant = self.cfg.head_e3mlp_variant or self.cfg.e3mlp_variant
        head_layers = int(self.cfg.head_e3mlp_layers)
        self.pair_key_to_index = {key: idx for idx, key in enumerate(pair_keys)}
        self.pair_condition_irreps = Irreps(f"{len(pair_keys)}x0e")
        self.shared_proj_in_irreps = (
            self.irreps_neck + self.pair_condition_irreps
        ).simplify()

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
        self.shared_diag_proj = None
        self.shared_offdiag_proj = None
        self.shared_shifted_self_proj = None
        self._pair_output_indices: dict[str, torch.Tensor] = {}
        self._shared_output_irreps = None
        if self.head_pair_mode == "split":
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
        else:
            self._shared_output_irreps = _build_shared_output_irreps(mapper, pair_keys)
            for key in pair_keys:
                indices = _build_pair_output_indices(
                    self._shared_output_irreps,
                    mapper.get_pair_irreps(key),
                )
                self.register_buffer(
                    f"pair_output_idx_{_sanitize_pair_key(key)}",
                    indices,
                    persistent=False,
                )
                self._pair_output_indices[key] = indices
            self.shared_diag_proj = E3MLP(
                self.shared_proj_in_irreps,
                self.shared_proj_in_irreps,
                self._shared_output_irreps,
                head_layers,
                self.cfg,
                activate_last=False,
                variant=head_variant,
                post_scale=self.cfg.head_diag_output_scale,
            )
            self.shared_offdiag_proj = E3MLP(
                self.shared_proj_in_irreps,
                self.shared_proj_in_irreps,
                self._shared_output_irreps,
                head_layers,
                self.cfg,
                activate_last=False,
                variant=head_variant,
                post_scale=self.cfg.head_offdiag_output_scale,
            )
            if self.separate_shifted_self:
                self.shared_shifted_self_proj = E3MLP(
                    self.shared_proj_in_irreps,
                    self.shared_proj_in_irreps,
                    self._shared_output_irreps,
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
            result[key] = self.forward_pair_chunk(
                key=key,
                node_feat=node_feat,
                key_edge_feat=key_edge_feat,
                key_edges=key_edges,
            )
        return result

    def forward_pair_chunk(
        self,
        *,
        key: str,
        node_feat: torch.Tensor,
        key_edge_feat: torch.Tensor,
        key_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one contiguous chunk of a single ordered element pair.

        ``key_edges`` follows the usual ``(shift_x, shift_y, shift_z, src,
        dst)`` layout.  Deriving the partitions from this chunk preserves the
        existing head semantics while avoiding a full pair-sized activation.
        """
        if key not in self.pair_key_to_index:
            raise KeyError(f"Unknown pair key {key!r}")
        if key_edge_feat.shape[0] != key_edges.shape[1]:
            raise ValueError(
                "Pair feature/edge length mismatch: "
                f"features={key_edge_feat.shape[0]} edges={key_edges.shape[1]}"
            )

        pair_vectors = key_edge_feat.new_zeros(
            (key_edges.shape[1], self.mapper.get_pair_irreps(key).dim)
        )
        pair_dtype = pair_vectors.dtype

        def _match_pair_dtype(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == pair_dtype else t.to(dtype=pair_dtype)

        src = key_edges[3]
        dst = key_edges[4]
        is_same_atom = src == dst
        is_zero_shift = (key_edges[:3] == 0).all(dim=0)
        diag_local_idx = torch.nonzero(
            is_same_atom & is_zero_shift, as_tuple=False
        ).flatten()
        if self.separate_shifted_self:
            shifted_self_local_idx = torch.nonzero(
                is_same_atom & (~is_zero_shift), as_tuple=False
            ).flatten()
            offdiag_local_idx = torch.nonzero(~is_same_atom, as_tuple=False).flatten()
        else:
            shifted_self_local_idx = key_edges.new_empty((0,), dtype=torch.long)
            offdiag_local_idx = torch.nonzero(
                ~(is_same_atom & is_zero_shift), as_tuple=False
            ).flatten()

        if diag_local_idx.numel() > 0:
            if self.use_node_embeddings_for_self_edges:
                diag_input = node_feat.index_select(
                    0, src.index_select(0, diag_local_idx)
                )
            else:
                diag_input = key_edge_feat.index_select(0, diag_local_idx)
            diag_hidden = self.diag_trunk(diag_input)
            if self.diag_tensor_square is not None:
                diag_hidden = self.diag_tensor_square(diag_hidden)
            diag_proj = (
                self.diag_projs[key]
                if self.head_pair_mode == "split"
                else self.shared_diag_proj
            )
            diag_vectors = self._project_pair(key, diag_hidden, diag_proj)
            if self.diag_log_scales is not None:
                diag_vectors = (
                    torch.exp(self.diag_log_scales[key](diag_hidden)) * diag_vectors
                )
            pair_vectors.index_copy_(0, diag_local_idx, _match_pair_dtype(diag_vectors))

        if shifted_self_local_idx.numel() > 0:
            if self.shifted_self_trunk is None:
                raise RuntimeError("Missing shifted-self trunk for shifted-self edges.")
            shifted_hidden = self.shifted_self_trunk(
                key_edge_feat.index_select(0, shifted_self_local_idx)
            )
            if self.shifted_self_tensor_square is not None:
                shifted_hidden = self.shifted_self_tensor_square(shifted_hidden)
            shifted_proj = (
                self.shifted_self_projs[key]
                if self.head_pair_mode == "split"
                else self.shared_shifted_self_proj
            )
            shifted_vectors = self._project_pair(key, shifted_hidden, shifted_proj)
            if self.shifted_self_log_scales is not None:
                shifted_vectors = (
                    torch.exp(self.shifted_self_log_scales[key](shifted_hidden))
                    * shifted_vectors
                )
            pair_vectors.index_copy_(
                0, shifted_self_local_idx, _match_pair_dtype(shifted_vectors)
            )

        if offdiag_local_idx.numel() > 0:
            offdiag_hidden = self.offdiag_trunk(
                key_edge_feat.index_select(0, offdiag_local_idx)
            )
            if self.offdiag_tensor_square is not None:
                offdiag_hidden = self.offdiag_tensor_square(offdiag_hidden)
            offdiag_proj = (
                self.offdiag_projs[key]
                if self.head_pair_mode == "split"
                else self.shared_offdiag_proj
            )
            offdiag_vectors = self._project_pair(key, offdiag_hidden, offdiag_proj)
            if self.offdiag_log_scales is not None:
                offdiag_vectors = (
                    torch.exp(self.offdiag_log_scales[key](offdiag_hidden))
                    * offdiag_vectors
                )
            pair_vectors.index_copy_(
                0, offdiag_local_idx, _match_pair_dtype(offdiag_vectors)
            )
        return pair_vectors

    def _project_pair(
        self, key: str, hidden: torch.Tensor, proj: nn.Module | None
    ) -> torch.Tensor:
        if proj is None:
            raise RuntimeError(f"Missing projection module for pair {key!r}.")
        if self.head_pair_mode == "split":
            return proj(hidden)
        conditioned = torch.cat(
            [hidden, self._pair_condition_features(key, hidden)], dim=-1
        )
        shared_output = proj(conditioned)
        gather_idx = self._pair_output_indices[key]
        return shared_output.index_select(-1, gather_idx.to(shared_output.device))

    def _pair_condition_features(self, key: str, hidden: torch.Tensor) -> torch.Tensor:
        out = hidden.new_zeros(hidden.shape[0], len(self.pair_keys))
        out[:, self.pair_key_to_index[key]] = 1.0
        return out


def _sanitize_pair_key(key: str) -> str:
    return key.replace("-", "_")


def _build_shared_output_irreps(
    mapper: BlockIrrepMapper,
    pair_keys: list[str],
) -> Irreps:
    counts: dict[str, int] = defaultdict(int)
    order: list[tuple[int, int]] = []
    for key in pair_keys:
        for mul, ir in mapper.get_pair_irreps(key):
            ir_key = f"{ir.l}{'e' if ir.p == 1 else 'o'}"
            counts[ir_key] = max(counts[ir_key], int(mul))
            order.append((ir.l, ir.p))
    seen: set[str] = set()
    parts: list[str] = []
    for l_value, parity in sorted(set(order), key=lambda item: (item[0], item[1])):
        ir_key = f"{l_value}{'e' if parity == 1 else 'o'}"
        if ir_key in seen:
            continue
        seen.add(ir_key)
        parts.append(f"{counts[ir_key]}x{ir_key}")
    return Irreps("+".join(parts)).simplify() if parts else Irreps("")


def _build_pair_output_indices(
    shared_irreps: Irreps,
    pair_irreps: Irreps,
) -> torch.Tensor:
    shared_cursor = 0
    shared_ranges: dict[str, list[int]] = {}
    for mul, ir in shared_irreps:
        ir_key = f"{ir.l}{'e' if ir.p == 1 else 'o'}"
        dim = ir.dim
        slots = []
        for copy_idx in range(int(mul)):
            slots.extend(
                range(
                    shared_cursor + copy_idx * dim, shared_cursor + (copy_idx + 1) * dim
                )
            )
        shared_ranges[ir_key] = slots
        shared_cursor += int(mul) * dim

    gather: list[int] = []
    per_irrep_offsets: dict[str, int] = defaultdict(int)
    for mul, ir in pair_irreps:
        ir_key = f"{ir.l}{'e' if ir.p == 1 else 'o'}"
        dim = ir.dim
        start = per_irrep_offsets[ir_key] * dim
        stop = start + int(mul) * dim
        gather.extend(shared_ranges[ir_key][start:stop])
        per_irrep_offsets[ir_key] += int(mul)
    return torch.tensor(gather, dtype=torch.long)
