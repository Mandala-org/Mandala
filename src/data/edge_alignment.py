from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import build_trace_alignment_from_pair_edges
from data.block_matrix import BlockMatrix


Edge5D = Tuple[int, int, int, int, int]


def block_matrix_edges_by_key(
    block_matrix: BlockMatrix,
) -> dict[str, list[Edge5D]]:
    grouped: dict[str, list[Edge5D]] = {}
    for key, edges_t in block_matrix.pair_edges.items():
        grouped[key] = [tuple(map(int, row)) for row in edges_t.t().tolist()]
    return grouped


def graph_edges_by_key(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    atoms_list: list[str],
) -> dict[str, list[Edge5D]]:
    grouped: dict[str, list[Edge5D]] = {}
    n_edges = edge_index.shape[1]
    for e in range(n_edges):
        src = int(edge_index[0, e].item())
        dst = int(edge_index[1, e].item())
        key = f"{atoms_list[src]}-{atoms_list[dst]}"
        grouped.setdefault(key, []).append(
            (
                int(edge_shift[0, e].item()),
                int(edge_shift[1, e].item()),
                int(edge_shift[2, e].item()),
                src,
                dst,
            )
        )
    return grouped


def build_global_edge_tensors_from_pair_edges(
    pair_edges_by_key: dict[str, torch.Tensor],
    ordered_keys: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_edges = [
        pair_edges_by_key[key]
        for key in ordered_keys
        if key in pair_edges_by_key and pair_edges_by_key[key].shape[1] > 0
    ]
    if not ordered_edges:
        raise RuntimeError("Cannot build graph edge tensors from empty pair_edges.")
    all_edges = torch.cat(ordered_edges, dim=1).to(device=device, dtype=torch.long)
    edge_index = all_edges[3:5]
    edge_shift = all_edges[:3]
    return edge_index, edge_shift


def edge5_distance(
    edge: Edge5D,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> float:
    sx, sy, sz, src, dst = edge
    src_t = torch.tensor(src, device=positions.device, dtype=torch.long)
    dst_t = torch.tensor(dst, device=positions.device, dtype=torch.long)
    disp = positions[dst_t] - positions[src_t]
    if box is not None:
        shift = torch.tensor(
            [sx, sy, sz], device=positions.device, dtype=positions.dtype
        )
        disp = disp + shift @ box
    return float(torch.linalg.norm(disp).item())


def reconcile_graph_edges_to_target(
    *,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    reference_matrix: BlockMatrix,
    atoms_list: list[str],
    mapper: BlockIrrepMapper,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    snapshot_label: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    validate_graph_prefix_matches_target(
        edge_index=edge_index,
        edge_shift=edge_shift,
        reference_matrix=reference_matrix,
        atoms_list=atoms_list,
        positions=positions,
        box=box,
        snapshot_label=snapshot_label,
    )
    return edge_index, edge_shift, False


def validate_graph_prefix_matches_target(
    *,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    reference_matrix: BlockMatrix,
    atoms_list: list[str],
    positions: torch.Tensor,
    box: torch.Tensor | None,
    snapshot_label: str,
) -> None:
    target_by_key = block_matrix_edges_by_key(reference_matrix)
    graph_by_key = graph_edges_by_key(edge_index, edge_shift, atoms_list)

    target_keys = sorted(target_by_key)
    graph_keys = sorted(graph_by_key)
    if target_keys != graph_keys:
        missing_in_graph = sorted(set(target_keys) - set(graph_keys))
        extra_in_graph = sorted(set(graph_keys) - set(target_keys))
        raise RuntimeError(
            f"[GRAPH] {snapshot_label} key-set mismatch: "
            f"missing_in_graph={missing_in_graph}, extra_in_graph={extra_in_graph}"
        )

    mismatch_lines: list[str] = []
    for key in target_keys:
        target_edges = target_by_key[key]
        graph_edges = graph_by_key[key]
        if len(graph_edges) < len(target_edges):
            raise RuntimeError(
                f"[GRAPH] {snapshot_label} key={key} graph prefix too short: "
                f"graph_len={len(graph_edges)} target_len={len(target_edges)}"
            )

        for idx, (target_edge, graph_edge) in enumerate(zip(target_edges, graph_edges)):
            if target_edge == graph_edge:
                continue
            mismatch_lines.append(
                "[GRAPH DIAGNOSTIC] "
                f"{snapshot_label} key={key} idx={idx} "
                f"expected=(sx,sy,sz,i,j,dist)=({target_edge[0]}, {target_edge[1]}, {target_edge[2]}, {target_edge[3]}, {target_edge[4]}, "
                f"{edge5_distance(target_edge, positions, box):.6f}) "
                f"got=(sx,sy,sz,i,j,dist)=({graph_edge[0]}, {graph_edge[1]}, {graph_edge[2]}, {graph_edge[3]}, {graph_edge[4]}, "
                f"{edge5_distance(graph_edge, positions, box):.6f})"
            )
            break

    if mismatch_lines:
        for line in mismatch_lines[:10]:
            print(line)
        raise RuntimeError(
            f"[GRAPH] {snapshot_label} edge prefix mismatch. "
            "The cutoff graph prefix does not exactly match the target matrix prefix."
        )


def _group_graph_edges_by_key(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    edge_type_idx: torch.Tensor,
    atoms_list: list[str],
    edge_types: list[str],
) -> Dict[str, List[Edge5D]]:
    grouped: Dict[str, List[Edge5D]] = defaultdict(list)

    n_edges = edge_index.shape[1]
    for e in range(n_edges):
        src = int(edge_index[0, e].item())
        dst = int(edge_index[1, e].item())
        sx = int(edge_shift[0, e].item())
        sy = int(edge_shift[1, e].item())
        sz = int(edge_shift[2, e].item())

        key_from_type = edge_types[int(edge_type_idx[e].item())]
        key_from_atoms = f"{atoms_list[src]}-{atoms_list[dst]}"
        if key_from_type != key_from_atoms:
            raise RuntimeError(
                f"Edge-type mismatch at edge {e}: {key_from_type=} vs {key_from_atoms=}"
            )

        grouped[key_from_type].append((sx, sy, sz, src, dst))

    return grouped


def strict_edge_alignment_check(
    target_matrix: BlockMatrix,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    edge_type_idx: torch.Tensor,
    atoms_list: list[str],
    edge_types: list[str],
    *,
    matrix_name: str,
    require_exact: bool = False,
) -> None:
    if not require_exact:
        return

    graph_edges_by_key = _group_graph_edges_by_key(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=edge_types,
    )

    target_keys = set(target_matrix.pair_edges.keys())
    graph_keys = set(graph_edges_by_key.keys())
    if target_keys != graph_keys:
        missing_in_graph = sorted(target_keys - graph_keys)
        extra_in_graph = sorted(graph_keys - target_keys)
        raise RuntimeError(
            f"[{matrix_name}] key-set mismatch. "
            f"Missing in graph={missing_in_graph}, extra in graph={extra_in_graph}"
        )

    for key, target_edges_t in target_matrix.pair_edges.items():
        target_edges: List[Edge5D] = [
            tuple(map(int, row)) for row in target_edges_t.t().tolist()
        ]
        graph_edges = graph_edges_by_key.get(key, [])

        if len(graph_edges) < len(target_edges):
            raise RuntimeError(
                f"[{matrix_name}] edge prefix too short for key '{key}': "
                f"graph_len={len(graph_edges)} target_len={len(target_edges)}"
            )

        for idx, (target_edge, graph_edge) in enumerate(zip(target_edges, graph_edges)):
            if target_edge != graph_edge:
                raise RuntimeError(
                    f"[{matrix_name}] edge prefix mismatch for key '{key}' at idx={idx}: "
                    f"{target_edge} != {graph_edge}"
                )


def strict_reverse_edge_check(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    *,
    edge_set_name: str = "graph",
) -> None:
    counts: Counter[Edge5D] = Counter()
    n_edges = edge_index.shape[1]
    for e in range(n_edges):
        edge = (
            int(edge_shift[0, e].item()),
            int(edge_shift[1, e].item()),
            int(edge_shift[2, e].item()),
            int(edge_index[0, e].item()),
            int(edge_index[1, e].item()),
        )
        counts[edge] += 1

    mismatches = []
    for edge, count in counts.items():
        sx, sy, sz, i, j = edge
        reverse = (-sx, -sy, -sz, j, i)
        reverse_count = counts.get(reverse, 0)
        if reverse_count != count:
            mismatches.append((edge, count, reverse, reverse_count))
            if len(mismatches) >= 10:
                break

    if mismatches:
        details = "; ".join(
            [
                f"edge={edge} count={count} reverse={reverse} reverse_count={reverse_count}"
                for edge, count, reverse, reverse_count in mismatches
            ]
        )
        raise RuntimeError(
            f"[{edge_set_name}] reverse-edge check failed. "
            f"Found {len(mismatches)} mismatches (showing up to 10): {details}"
        )


def build_prediction_edge_metadata(
    *,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    edge_type_idx: torch.Tensor,
    atoms: tuple[str, ...],
    edge_types: list[str],
    edge_type2idx: dict[str, int],
    separate_shifted_self: bool,
) -> dict[str, object]:
    edges_5d = torch.cat([edge_shift, edge_index], dim=0)
    pred_pair_edges_static: dict[str, torch.Tensor] = {}
    pred_lookup_static: dict[Edge5D, tuple[str, int]] = {}
    pred_edge_keys: list[str] = []
    edge_partitions: dict[str, dict[str, torch.Tensor]] = {}

    for key in edge_types:
        type_idx = edge_type2idx[key]
        global_idx = torch.nonzero(edge_type_idx == type_idx, as_tuple=False).flatten()
        if global_idx.numel() == 0:
            continue
        key_edges = edges_5d.index_select(1, global_idx)
        pred_pair_edges_static[key] = key_edges
        pred_edge_keys.append(key)

        for local_idx, edge_5d in enumerate(key_edges.t().tolist()):
            sx, sy, sz, i, j = edge_5d
            pred_lookup_static[(int(sx), int(sy), int(sz), int(i), int(j))] = (
                key,
                local_idx,
            )

        src = key_edges[3]
        dst = key_edges[4]
        is_same_atom = src == dst
        is_zero_shift = (key_edges[:3] == 0).all(dim=0)
        is_diag = is_same_atom & is_zero_shift
        if separate_shifted_self:
            is_shifted_self = is_same_atom & (~is_zero_shift)
            is_offdiag = ~is_same_atom
        else:
            is_shifted_self = torch.zeros_like(is_diag, dtype=torch.bool)
            is_offdiag = ~is_diag

        edge_partitions[key] = {
            "global_idx": global_idx,
            "diag_local_idx": torch.nonzero(is_diag, as_tuple=False).flatten(),
            "shifted_self_local_idx": torch.nonzero(
                is_shifted_self, as_tuple=False
            ).flatten(),
            "offdiag_local_idx": torch.nonzero(is_offdiag, as_tuple=False).flatten(),
        }

    pred_trace_alignment = build_trace_alignment_from_pair_edges(pred_pair_edges_static)
    return {
        "pred_pair_edges_static": pred_pair_edges_static,
        "pred_lookup_static": pred_lookup_static,
        "pred_edge_keys": tuple(pred_edge_keys),
        "pred_edge_key_set": set(pred_edge_keys),
        "pred_trace_alignment": pred_trace_alignment,
        "edge_partitions": edge_partitions,
    }
