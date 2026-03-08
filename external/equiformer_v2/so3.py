"""
Minimal vendored subset of EquiformerV2's SO(3) utilities used by Mandala.
"""

import math

import torch
import torch.nn as nn
from e3nn import o3

from .wigner import wigner_D


class CoefficientMappingModule(nn.Module):
    def __init__(self, lmax_list, mmax_list):
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.device = "cpu"

        l_harmonic = torch.tensor([], device=self.device).long()
        m_harmonic = torch.tensor([], device=self.device).long()
        m_complex = torch.tensor([], device=self.device).long()
        res_size = torch.zeros([self.num_resolutions], device=self.device).long()

        offset = 0
        for i in range(self.num_resolutions):
            for l in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], l)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                m_complex = torch.cat([m_complex, m], dim=0)
                m_harmonic = torch.cat([m_harmonic, torch.abs(m).long()], dim=0)
                l_harmonic = torch.cat([l_harmonic, m.fill_(l).long()], dim=0)
            res_size[i] = len(l_harmonic) - offset
            offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        to_m = torch.zeros([num_coefficients, num_coefficients], device=self.device)
        m_size = torch.zeros([max(self.mmax_list) + 1], device=self.device).long()

        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("res_size", res_size)
        self.register_buffer("to_m", to_m.detach())
        self.register_buffer("m_size", m_size)

        self.lmax_cache = None
        self.mmax_cache = None
        self.mask_indices_cache = None
        self.rotate_inv_rescale_cache = None

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(l_harmonic), device=self.device)
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=self.device).long()
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    def coefficient_idx(self, lmax, mmax):
        if self.lmax_cache == lmax and self.mmax_cache == mmax:
            if self.mask_indices_cache is not None:
                return self.mask_indices_cache

        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        self.device = mask.device
        indices = torch.arange(len(mask), device=self.device)
        mask_indices = torch.masked_select(indices, mask)
        self.lmax_cache = lmax
        self.mmax_cache = mmax
        self.mask_indices_cache = mask_indices
        return self.mask_indices_cache

    def get_rotate_inv_rescale(self, lmax, mmax):
        if self.lmax_cache == lmax and self.mmax_cache == mmax:
            if self.rotate_inv_rescale_cache is not None:
                return self.rotate_inv_rescale_cache

        if self.mask_indices_cache is None:
            self.coefficient_idx(lmax, mmax)

        rotate_inv_rescale = torch.ones(
            (1, (lmax + 1) ** 2, (lmax + 1) ** 2), device=self.device
        )
        for l in range(lmax + 1):
            if l <= mmax:
                continue
            start_idx = l**2
            length = 2 * l + 1
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[
                :, start_idx : start_idx + length, start_idx : start_idx + length
            ] = rescale_factor

        rotate_inv_rescale = rotate_inv_rescale[:, :, self.mask_indices_cache]
        self.rotate_inv_rescale_cache = rotate_inv_rescale
        return self.rotate_inv_rescale_cache

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lmax_list={self.lmax_list}, "
            f"mmax_list={self.mmax_list})"
        )


class SO3_Embedding:
    def __init__(self, length, lmax_list, num_channels, device, dtype):
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients += int((lmax_list[i] + 1) ** 2)

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    def clone(self):
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone

    def set_embedding(self, embedding):
        self.length = len(embedding)
        self.embedding = embedding

    def set_lmax_mmax(self, lmax_list, mmax_list):
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    def _expand_edge(self, edge_index):
        self.set_embedding(self.embedding[edge_index])

    def expand_edge(self, edge_index):
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    def _reduce_edge(self, edge_index, num_nodes):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)

    def _l_primary(self, mapping):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)

    def _rotate(self, so3_rotation, lmax_list, mmax_list):
        if self.num_resolutions == 1:
            embedding_rotate = so3_rotation[0].rotate(
                self.embedding, lmax_list[0], mmax_list[0]
            )
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        so3_rotation[i].rotate(
                            embedding_i, lmax_list[i], mmax_list[i]
                        ),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    def _rotate_inv(self, so3_rotation, mapping_reduced):
        if self.num_resolutions == 1:
            embedding_rotate = so3_rotation[0].rotate_inv(
                self.embedding, self.lmax_list[0], self.mmax_list[0]
            )
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mapping_reduced.res_size[i]
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        so3_rotation[i].rotate_inv(
                            embedding_i, self.lmax_list[i], self.mmax_list[i]
                        ),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)


class SO3_Rotation(nn.Module):
    def __init__(self, lmax):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

    def set_wigner(self, rot_mat3x3):
        self.device = rot_mat3x3.device
        self.dtype = rot_mat3x3.dtype
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()
        self.wigner = self.wigner.detach()
        self.wigner_inv = self.wigner_inv.detach()

    def rotate(self, embedding, out_lmax, out_mmax):
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)

    def rotate_inv(self, embedding, in_lmax, in_mmax):
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask]
        wigner_inv_rescale = self.mapping.get_rotate_inv_rescale(in_lmax, in_mmax)
        wigner_inv = wigner_inv * wigner_inv_rescale
        return torch.bmm(wigner_inv, embedding)

    def RotationToWignerDMatrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        rot = (
            o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(rot[..., 0, 2], rot[..., 0, 0])

        size = (end_lmax + 1) ** 2 - start_lmax**2
        wigner = torch.zeros(len(alpha), size, size, device=self.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()
