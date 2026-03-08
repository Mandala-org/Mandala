from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from external.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Rotation,
)
from external.equiformer_v2.so2_ops import SO2_Convolution
from external.equiformer_v2.edge_rot_mat import init_edge_rot_mat


class SO2OpsEquiformerDirectMasked(nn.Module):
    """
    EquiformerV2 SO(2) block wrapper with precomputed pack/unpack indices
    to reduce Python work inside forward.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        edge_dim: int,
        max_m: int | None = None,
        edge_channels_list: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.edge_dim = int(edge_dim)

        self._parities = (+1, -1)
        self._in_blocks = self._build_blocks(self.irreps_in)
        self._out_blocks = self._build_blocks(self.irreps_out)

        self._parity_lmax: Dict[int, int] = {}
        self._parity_in_channels: Dict[int, int] = {}
        self._parity_out_channels: Dict[int, int] = {}

        for p in self._parities:
            lmax_in = max((b["l"] for b in self._in_blocks if b["p"] == p), default=0)
            lmax_out = max((b["l"] for b in self._out_blocks if b["p"] == p), default=0)
            self._parity_lmax[p] = max(lmax_in, lmax_out)

            in_ch = max((b["mul"] for b in self._in_blocks if b["p"] == p), default=0)
            out_ch = max((b["mul"] for b in self._out_blocks if b["p"] == p), default=0)
            self._parity_in_channels[p] = in_ch
            self._parity_out_channels[p] = out_ch

        self._max_m: Dict[int, int] = {}
        for p in self._parities:
            lmax = self._parity_lmax[p]
            self._max_m[p] = lmax if max_m is None else min(int(max_m), lmax)

        self._mapping = nn.ModuleDict()
        self._rotation = nn.ModuleDict()
        self._so2 = nn.ModuleDict()

        # Precomputed scatter/gather indices for packing/unpacking.
        self._pack_idx: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._unpack_idx: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for p in self._parities:
            if self._parity_lmax[p] == 0 and self._parity_in_channels[p] == 0:
                continue

            lmax_list = [self._parity_lmax[p]]
            mmax_list = [self._max_m[p]]

            self._mapping[f"p{p}"] = CoefficientMappingModule(lmax_list, mmax_list)
            self._rotation[f"p{p}"] = SO3_Rotation(self._parity_lmax[p])

            sphere_channels = self._parity_in_channels[p]
            m_output_channels = self._parity_out_channels[p]

            if sphere_channels == 0 or m_output_channels == 0:
                continue

            if edge_channels_list is None:
                edge_channels_list_p = [self.edge_dim, self.edge_dim]
            else:
                edge_channels_list_p = list(edge_channels_list)

            self._so2[f"p{p}"] = SO2_Convolution(
                sphere_channels,
                m_output_channels,
                lmax_list,
                mmax_list,
                self._mapping[f"p{p}"],
                internal_weights=False,
                edge_channels_list=edge_channels_list_p,
                extra_m0_output_channels=None,
            )

            pack_in, pack_out = self._build_pack_indices(
                p, self._in_blocks, self._parity_lmax[p], sphere_channels
            )
            unpack_in, unpack_out = self._build_unpack_indices(
                p, self._out_blocks, self._parity_lmax[p], m_output_channels
            )

            self.register_buffer(f"_pack_in_{p}", pack_in)
            self.register_buffer(f"_pack_out_{p}", pack_out)
            self.register_buffer(f"_unpack_in_{p}", unpack_in)
            self.register_buffer(f"_unpack_out_{p}", unpack_out)

            self._pack_idx[p] = (pack_in, pack_out)
            self._unpack_idx[p] = (unpack_in, unpack_out)

    @staticmethod
    def _build_blocks(irreps: Irreps) -> List[dict]:
        blocks: List[dict] = []
        offset = 0
        for mul, ir in irreps:
            dim = ir.dim
            blocks.append(
                {
                    "mul": mul,
                    "l": ir.l,
                    "p": ir.p,
                    "dim": dim,
                    "slice": slice(offset, offset + mul * dim),
                }
            )
            offset += mul * dim
        return blocks

    @staticmethod
    def _build_pack_indices(
        parity: int, blocks: List[dict], lmax: int, channels: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_idx: List[int] = []
        out_idx: List[int] = []
        for blk in blocks:
            if blk["p"] != parity:
                continue
            l = blk["l"]
            dim = blk["dim"]
            start = l * l
            sl = blk["slice"]
            for c in range(blk["mul"]):
                for m in range(dim):
                    in_idx.append(sl.start + c * dim + m)
                    out_idx.append((start + m) * channels + c)
        return torch.tensor(in_idx, dtype=torch.long), torch.tensor(out_idx, dtype=torch.long)

    @staticmethod
    def _build_unpack_indices(
        parity: int, blocks: List[dict], lmax: int, channels: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_idx: List[int] = []
        out_idx: List[int] = []
        for blk in blocks:
            if blk["p"] != parity:
                continue
            l = blk["l"]
            dim = blk["dim"]
            start = l * l
            sl = blk["slice"]
            for c in range(blk["mul"]):
                for m in range(dim):
                    in_idx.append((start + m) * channels + c)
                    out_idx.append(sl.start + c * dim + m)
        return torch.tensor(in_idx, dtype=torch.long), torch.tensor(out_idx, dtype=torch.long)

    def _pack_parity_masked(self, x: torch.Tensor, parity: int, lmax: int, channels: int) -> SO3_Embedding:
        B = x.shape[0]
        eq_flat = x.new_zeros(B, (lmax + 1) ** 2 * channels)
        pack_in = getattr(self, f"_pack_in_{parity}")
        pack_out = getattr(self, f"_pack_out_{parity}")
        eq_flat.index_copy_(1, pack_out, x[:, pack_in])
        emb = SO3_Embedding(B, [lmax], channels, x.device, x.dtype)
        emb.set_embedding(eq_flat.view(B, (lmax + 1) ** 2, channels))
        return emb

    def _unpack_parity_masked(self, emb: SO3_Embedding, parity: int, out: torch.Tensor) -> None:
        eq_flat = emb.embedding.reshape(emb.embedding.shape[0], -1)
        unpack_in = getattr(self, f"_unpack_in_{parity}")
        unpack_out = getattr(self, f"_unpack_out_{parity}")
        out[:, unpack_out] = eq_flat[:, unpack_in]

    def forward(self, x: torch.Tensor, r_hat: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != r_hat.shape[0] or x.shape[0] != edge_emb.shape[0]:
            raise ValueError("x, r_hat, edge_emb must share batch dimension")

        edge_rot_mat = init_edge_rot_mat(r_hat)
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)

        for p in self._parities:
            key = f"p{p}"
            if key not in self._so2:
                continue

            lmax = self._parity_lmax[p]
            channels_in = self._parity_in_channels[p]
            if channels_in == 0:
                continue

            mapping = self._mapping[key]
            rot = self._rotation[key]
            rot.set_wigner(edge_rot_mat)

            x_emb = self._pack_parity_masked(x, p, lmax, channels_in)
            x_emb._rotate([rot], [lmax], [self._max_m[p]])
            y_emb = self._so2[key](x_emb, edge_emb)
            y_emb._rotate_inv([rot], mapping)
            self._unpack_parity_masked(y_emb, p, out)

        return out
