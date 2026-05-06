from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from e3nn.o3 import Irrep

from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix, IrrepsBlockData


@dataclass(frozen=True, slots=True)
class IrrepProjector:
    pair_key: str
    indices: torch.Tensor
    dim_i: int
    dim_j: int


def get_all_irreps(mapper: BlockIrrepMapper) -> list[Irrep]:
    irreps_set: set[Irrep] = set()
    for edge_type in mapper.edge_types:
        pair_irreps = mapper.get_pair_irreps(edge_type)
        for mul, irrep in pair_irreps:
            if mul > 0:
                irreps_set.add(irrep)
    return sorted(irreps_set, key=lambda ir: (ir.l, ir.p))


def build_irrep_projector_cache(
    mapper: BlockIrrepMapper,
    irreps: Iterable[Irrep] | None = None,
) -> dict[str, dict[str, IrrepProjector]]:
    if irreps is None:
        irreps = get_all_irreps(mapper)

    cache: dict[str, dict[str, IrrepProjector]] = {str(ir): {} for ir in irreps}
    allowed_irreps = set(cache.keys())

    for pair_key in mapper.edge_types:
        pair_irreps = mapper.get_pair_irreps(pair_key)
        map_key = mapper._canonical_pair(pair_key)
        item = mapper._lookup(map_key)

        start = 0
        idx_parts_by_irrep: dict[str, list[torch.Tensor]] = {}
        for mul, irrep in pair_irreps:
            term_dim = mul * irrep.dim
            irrep_key = str(irrep)
            if term_dim > 0 and irrep_key in allowed_irreps:
                idx_parts_by_irrep.setdefault(irrep_key, []).append(
                    torch.arange(start, start + term_dim, dtype=torch.long)
                )
            start += term_dim

        for irrep_key, idx_parts in idx_parts_by_irrep.items():
            indices = (
                idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
            )
            cache[irrep_key][pair_key] = IrrepProjector(
                pair_key=pair_key,
                indices=indices,
                dim_i=item.dim_i,
                dim_j=item.dim_j,
            )

    return cache


def project_irrep_vectors_to_blocks(
    vectors_full: torch.Tensor,
    projector: IrrepProjector,
    mapper: BlockIrrepMapper,
) -> torch.Tensor:
    pair = mapper._canonical_pair(projector.pair_key)
    q_full = mapper._get_q(pair)
    idx_for_vectors = projector.indices.to(device=vectors_full.device)
    idx_for_q = projector.indices.to(device=q_full.device)

    vec_sel = vectors_full.index_select(1, idx_for_vectors)
    q_subset = q_full.index_select(0, idx_for_q)
    if vec_sel.dtype != q_subset.dtype:
        vec_sel = vec_sel.to(dtype=q_subset.dtype)

    block_flat = vec_sel @ q_subset
    return block_flat.view(vec_sel.shape[0], projector.dim_i, projector.dim_j)


def filter_irreps_block_data_by_irrep(
    irreps_block_data: IrrepsBlockData,
    target_irrep: Irrep | str,
    mapper: BlockIrrepMapper,
) -> IrrepsBlockData:
    if isinstance(target_irrep, str):
        target_irrep = Irrep(target_irrep)

    filtered_vectors: dict[str, torch.Tensor] = {}
    for edge_type, vectors in irreps_block_data.pair_vectors.items():
        pair_irreps = mapper.get_pair_irreps(edge_type)
        filtered_vec = torch.zeros_like(vectors)
        start_idx = 0
        for mul, irrep in pair_irreps:
            irrep_dim = irrep.dim * mul
            if irrep == target_irrep:
                filtered_vec[:, start_idx : start_idx + irrep_dim] = vectors[
                    :, start_idx : start_idx + irrep_dim
                ]
            start_idx += irrep_dim
        filtered_vectors[edge_type] = filtered_vec

    return IrrepsBlockData(
        atoms=irreps_block_data.atoms,
        atom_counts=irreps_block_data.atom_counts,
        pair_vectors=filtered_vectors,
        pair_edges=irreps_block_data.pair_edges,
        lookup=irreps_block_data.lookup,
        orbital_cfg=irreps_block_data.orbital_cfg,
        basis=irreps_block_data.basis,
    )


def split_hamiltonian_by_irrep(
    H_matrix: BlockMatrix,
    mapper: BlockIrrepMapper,
    target_irrep: Irrep | str,
) -> BlockMatrix:
    irreps_data = H_matrix.to_vectors(mapper)
    filtered_irreps = filter_irreps_block_data_by_irrep(
        irreps_data, target_irrep, mapper
    )
    return filtered_irreps.to_blocks(mapper)


def compute_irrep_metrics(
    pred_H_irreps: IrrepsBlockData,
    target_H_irreps: IrrepsBlockData,
    all_irreps: Iterable[Irrep],
    mapper: BlockIrrepMapper,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rel_eps = 1e-12

    for irrep in all_irreps:
        pred_irrep_filtered = filter_irreps_block_data_by_irrep(
            pred_H_irreps, irrep, mapper
        )
        target_irrep_filtered = filter_irreps_block_data_by_irrep(
            target_H_irreps, irrep, mapper
        )

        pred_irrep_blocks = pred_irrep_filtered.to_blocks(mapper)
        target_irrep_blocks = target_irrep_filtered.to_blocks(mapper)

        sum_l1_elem = 0.0
        sum_l2_elem = 0.0
        total_elements = 0
        sum_l1_block = 0.0
        sum_l1_target_block = 0.0
        sum_l2_block_sq = 0.0
        sum_l2_target_block_sq = 0.0
        total_blocks = 0
        sum_l1_target_full = 0.0
        sum_l2_target_full_sq = 0.0

        for key in target_irrep_blocks.pair_blocks.keys():
            if key not in pred_irrep_blocks.pair_blocks:
                continue
            pred_blocks = pred_irrep_blocks.pair_blocks[key]
            targ_blocks_full = target_irrep_blocks.pair_blocks[key]
            target_n = targ_blocks_full.shape[0]
            if target_n <= 0:
                continue
            if pred_blocks.shape[0] < target_n:
                raise ValueError(
                    f"Predicted irrep blocks for key '{key}' are too short: "
                    f"pred_len={pred_blocks.shape[0]} target_len={target_n}"
                )

            pred_sel = pred_blocks[:target_n]
            targ_sel = targ_blocks_full[:target_n]
            diff = pred_sel - targ_sel

            l1_error_blocks = torch.sum(torch.abs(diff), dim=(1, 2))
            l2_error_blocks_sq = torch.sum(diff**2, dim=(1, 2))
            l1_target_blocks = torch.sum(torch.abs(targ_sel), dim=(1, 2))
            l2_target_blocks_sq = torch.sum(targ_sel**2, dim=(1, 2))

            sum_l1_elem += torch.sum(l1_error_blocks).item()
            sum_l2_elem += torch.sum(l2_error_blocks_sq).item()
            total_elements += diff.numel()
            sum_l1_block += torch.sum(l1_error_blocks).item()
            sum_l1_target_block += torch.sum(l1_target_blocks).item()
            sum_l2_block_sq += torch.sum(l2_error_blocks_sq).item()
            sum_l2_target_block_sq += torch.sum(l2_target_blocks_sq).item()
            total_blocks += target_n
            sum_l1_target_full += torch.sum(l1_target_blocks).item()
            sum_l2_target_full_sq += torch.sum(l2_target_blocks_sq).item()

        if total_elements <= 0:
            continue

        l1_elem = sum_l1_elem / total_elements
        l2_elem = (sum_l2_elem / total_elements) ** 0.5
        l1_block = sum_l1_block / max(total_blocks, 1)
        l2_block = (sum_l2_block_sq / max(total_blocks, 1)) ** 0.5
        l1_block_rel = sum_l1_block / (sum_l1_target_block + rel_eps)
        l2_block_rel = (sum_l2_block_sq**0.5) / (sum_l2_target_block_sq**0.5 + rel_eps)
        l1_full_rel = sum_l1_elem / (sum_l1_target_full + rel_eps)
        l2_full_rel = (sum_l2_elem**0.5) / (sum_l2_target_full_sq**0.5 + rel_eps)

        irrep_str = str(irrep)
        metrics[f"{irrep_str}_l1_elem"] = l1_elem
        metrics[f"{irrep_str}_l2_elem"] = l2_elem
        metrics[f"{irrep_str}_l1_block"] = l1_block
        metrics[f"{irrep_str}_l1_block_rel"] = l1_block_rel
        metrics[f"{irrep_str}_l2_block"] = l2_block
        metrics[f"{irrep_str}_l2_block_rel"] = l2_block_rel
        metrics[f"{irrep_str}_l1_full_rel"] = l1_full_rel
        metrics[f"{irrep_str}_l2_full_rel"] = l2_full_rel

    return metrics
