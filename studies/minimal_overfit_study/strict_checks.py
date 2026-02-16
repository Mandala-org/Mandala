"""
Strict alignment checks for minimal overfit study.

These checks verify that graph edges and block-matrix edges are aligned in the
same deterministic order per edge type, so loss is computed on the intended
blocks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch


Edge5D = Tuple[int, int, int, int, int]


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
    target_matrix,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    edge_type_idx: torch.Tensor,
    atoms_list: list[str],
    edge_types: list[str],
    *,
    matrix_name: str,
    require_exact: bool = False,
) -> None:
    """
    Validate per-key edge ordering between model graph edges and target matrix edges.

    Behavior:
    - Always checks that the overlapping prefix used by current loss logic aligns
      exactly edge-by-edge.
    - If `require_exact=True`, also requires identical edge counts and full equality.
    """
    graph_edges_by_key = _group_graph_edges_by_key(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=edge_types,
    )

    for key, target_edges_t in target_matrix.pair_edges.items():
        target_edges: List[Edge5D] = [
            tuple(map(int, row)) for row in target_edges_t.t().tolist()
        ]
        graph_edges = graph_edges_by_key.get(key, [])

        min_n = min(len(target_edges), len(graph_edges))
        for idx in range(min_n):
            if target_edges[idx] != graph_edges[idx]:
                raise RuntimeError(
                    f"[{matrix_name}] edge order mismatch for key '{key}' at idx={idx}: "
                    f"{target_edges[idx]} != {graph_edges[idx]}"
                )

        if require_exact and len(target_edges) != len(graph_edges):
            raise RuntimeError(
                f"[{matrix_name}] edge count mismatch for key '{key}': "
                f"target={len(target_edges)} vs graph={len(graph_edges)}"
            )

    if require_exact:
        target_keys = set(target_matrix.pair_edges.keys())
        graph_keys = set(graph_edges_by_key.keys())
        if target_keys != graph_keys:
            missing_in_graph = sorted(target_keys - graph_keys)
            missing_in_target = sorted(graph_keys - target_keys)
            raise RuntimeError(
                f"[{matrix_name}] key set mismatch. "
                f"Missing in graph={missing_in_graph}, missing in target={missing_in_target}"
            )

