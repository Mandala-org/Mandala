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
    return build_irrep_block_matrix_cache(H_matrix, mapper, [target_irrep])[
        str(Irrep(target_irrep) if isinstance(target_irrep, str) else target_irrep)
    ]


def build_irrep_block_matrix_cache(
    matrix: BlockMatrix,
    mapper: BlockIrrepMapper,
    all_irreps: Iterable[Irrep] | None = None,
) -> dict[str, BlockMatrix]:
    matrix_irreps = matrix.to_vectors(mapper)
    if all_irreps is None:
        all_irreps = get_all_irreps(mapper)

    cache: dict[str, BlockMatrix] = {}
    for irrep in all_irreps:
        filtered_irreps = filter_irreps_block_data_by_irrep(
            matrix_irreps, irrep, mapper
        )
        cache[str(irrep)] = filtered_irreps.to_blocks(mapper)
    return cache


def compute_irrep_metrics(
    pred_H: BlockMatrix,
    target_H: BlockMatrix,
    all_irreps: Iterable[Irrep],
    mapper: BlockIrrepMapper,
    *,
    pred_irrep_blocks: dict[str, BlockMatrix] | None = None,
    target_irrep_blocks: dict[str, BlockMatrix] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rel_eps = 1e-12
    if pred_irrep_blocks is None:
        pred_irrep_blocks = build_irrep_block_matrix_cache(pred_H, mapper, all_irreps)
    if target_irrep_blocks is None:
        target_irrep_blocks = build_irrep_block_matrix_cache(
            target_H, mapper, all_irreps
        )

    for irrep in all_irreps:
        irrep_key = str(irrep)
        pred_irrep = pred_irrep_blocks[irrep_key]
        target_irrep = target_irrep_blocks[irrep_key]

        sum_l2_block_sq = 0.0
        sum_l2_target_block_sq = 0.0
        total_blocks = 0

        for key in target_irrep.pair_blocks.keys():
            if key not in pred_irrep.pair_blocks:
                continue
            pred_blocks = pred_irrep.pair_blocks[key]
            targ_blocks_full = target_irrep.pair_blocks[key]
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

            l2_error_blocks_sq = torch.sum(diff**2, dim=(1, 2))
            l2_target_blocks_sq = torch.sum(targ_sel**2, dim=(1, 2))

            sum_l2_block_sq += torch.sum(l2_error_blocks_sq).item()
            sum_l2_target_block_sq += torch.sum(l2_target_blocks_sq).item()
            total_blocks += target_n

        if total_blocks <= 0:
            continue

        l2_block = (sum_l2_block_sq / max(total_blocks, 1)) ** 0.5
        l2_block_rel = (sum_l2_block_sq**0.5) / (sum_l2_target_block_sq**0.5 + rel_eps)

        metrics[f"{irrep_key}_l2_block_abs"] = l2_block
        metrics[f"{irrep_key}_l2_block_rel"] = l2_block_rel

    return metrics


def compute_hamiltonian_mae_contributions(
    pred_H: BlockMatrix,
    target_H: BlockMatrix,
    mapper: BlockIrrepMapper,
    *,
    all_irreps: Iterable[Irrep] | None = None,
    compute_irrep_sums: bool = True,
    compute_pair_sums: bool = True,
    pred_irrep_blocks: dict[str, BlockMatrix] | None = None,
    target_irrep_blocks: dict[str, BlockMatrix] | None = None,
) -> dict[str, object]:
    if all_irreps is None:
        all_irreps = get_all_irreps(mapper)

    pair_abs_sums: dict[str, float] = {}
    irrep_abs_sums: dict[str, float] = {str(ir): 0.0 for ir in all_irreps}
    total_abs_sum = 0.0
    total_count = 0
    if compute_irrep_sums:
        if pred_irrep_blocks is None:
            pred_irrep_blocks = build_irrep_block_matrix_cache(
                pred_H, mapper, all_irreps
            )
        if target_irrep_blocks is None:
            target_irrep_blocks = build_irrep_block_matrix_cache(
                target_H, mapper, all_irreps
            )

    for pair_key, target_blocks in target_H.pair_blocks.items():
        if pair_key not in pred_H.pair_blocks:
            continue
        pred_blocks = pred_H.pair_blocks[pair_key]
        if pred_blocks.shape != target_blocks.shape:
            raise ValueError(
                f"Hamiltonian irrep contribution logging requires matching shapes for key '{pair_key}', "
                f"got pred_shape={tuple(pred_blocks.shape)} target_shape={tuple(target_blocks.shape)}"
            )
        diff = pred_blocks - target_blocks
        abs_sum = float(torch.sum(torch.abs(diff)).item())
        if compute_pair_sums:
            pair_abs_sums[pair_key] = abs_sum
        total_abs_sum += abs_sum
        total_count += int(diff.numel())

    if compute_irrep_sums:
        for irrep in all_irreps:
            irrep_key = str(irrep)
            pred_irrep = pred_irrep_blocks[irrep_key]
            target_irrep = target_irrep_blocks[irrep_key]

            irrep_abs_sum = 0.0
            for pair_key, target_blocks in target_irrep.pair_blocks.items():
                if pair_key not in pred_irrep.pair_blocks:
                    continue
                pred_blocks = pred_irrep.pair_blocks[pair_key]
                if pred_blocks.shape != target_blocks.shape:
                    raise ValueError(
                        f"Hamiltonian irrep contribution logging requires matching shapes for key '{pair_key}' "
                        f"at irrep '{irrep_key}', got pred_shape={tuple(pred_blocks.shape)} "
                        f"target_shape={tuple(target_blocks.shape)}"
                    )
                irrep_abs_sum += float(
                    torch.sum(torch.abs(pred_blocks - target_blocks)).item()
                )
            irrep_abs_sums[irrep_key] = irrep_abs_sum

    return {
        "total_abs_sum": total_abs_sum,
        "total_count": total_count,
        "pair_abs_sums": pair_abs_sums,
        "irrep_abs_sums": irrep_abs_sums,
    }
