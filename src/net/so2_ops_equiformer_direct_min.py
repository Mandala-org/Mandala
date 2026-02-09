from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from net.equiformer_v2.so3 import SO3_Embedding, CoefficientMappingModule, SO3_Rotation
from net.equiformer_v2.so2_ops import SO2_Convolution
from net.equiformer_v2.edge_rot_mat import init_edge_rot_mat


class SO2OpsEquiformerDirect(nn.Module):
    """EquiformerV2 SO(2) block wrapper with minimal commentary."""

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

    def _pack_parity(
        self,
        x: torch.Tensor,
        parity: int,
        lmax: int,
        channels: int,
        blocks: List[dict],
    ) -> SO3_Embedding:
        B = x.shape[0]
        emb = SO3_Embedding(B, [lmax], channels, x.device, x.dtype)
        for blk in blocks:
            if blk["p"] != parity:
                continue
            l = blk["l"]
            dim = blk["dim"]
            start = l * l
            sl = blk["slice"]
            x_blk = x[:, sl].view(B, blk["mul"], dim)
            for c in range(blk["mul"]):
                emb.embedding[:, start : start + dim, c] = x_blk[:, c, :]
        return emb

    def _unpack_parity(
        self,
        emb: SO3_Embedding,
        parity: int,
        blocks: List[dict],
        out: torch.Tensor,
    ) -> None:
        for blk in blocks:
            if blk["p"] != parity:
                continue
            l = blk["l"]
            dim = blk["dim"]
            start = l * l
            sl = blk["slice"]
            for c in range(blk["mul"]):
                out[:, sl.start + c * dim : sl.start + (c + 1) * dim] = emb.embedding[
                    :, start : start + dim, c
                ]

    def forward(
        self,
        x: torch.Tensor,
        r_hat: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
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

            x_emb = self._pack_parity(x, p, lmax, channels_in, self._in_blocks)
            x_emb._rotate([rot], [lmax], [self._max_m[p]])
            y_emb = self._so2[key](x_emb, edge_emb)
            y_emb._rotate_inv([rot], mapping)
            self._unpack_parity(y_emb, p, self._out_blocks, out)

        return out
