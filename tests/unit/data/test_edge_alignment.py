from collections import Counter

import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from data.edge_alignment import strict_edge_alignment_check


def _make_matrix(edges: torch.Tensor) -> BlockMatrix:
    key = "H-H"
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    blocks = torch.zeros(edges.shape[1], 1, 1)
    lookup = {tuple(edge.tolist()): (key, idx) for idx, edge in enumerate(edges.t())}
    return BlockMatrix(
        atoms=("H", "H"),
        atom_counts=Counter(("H", "H")),
        pair_blocks={key: blocks},
        pair_edges={key: edges},
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis="e3nn",
    )


@pytest.mark.unit
def test_strict_edge_alignment_check_allows_prefix_match_without_mutation():
    target_edges = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    graph_edges = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=torch.long,
    )
    target = _make_matrix(target_edges)
    graph_index = graph_edges[3:]
    graph_shift = graph_edges[:3]
    original_index = graph_index.clone()
    original_shift = graph_shift.clone()

    strict_edge_alignment_check(
        target,
        edge_index=graph_index,
        edge_shift=graph_shift,
        edge_type_idx=torch.zeros(graph_edges.shape[1], dtype=torch.long),
        atoms_list=["H", "H"],
        edge_types=["H-H"],
        matrix_name="hamiltonian",
        require_exact=True,
    )

    assert torch.equal(graph_index, original_index)
    assert torch.equal(graph_shift, original_shift)


@pytest.mark.unit
def test_strict_edge_alignment_check_rejects_prefix_mismatch():
    target_edges = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    graph_edges = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=torch.long,
    )
    target = _make_matrix(target_edges)
    with pytest.raises(RuntimeError, match="edge prefix mismatch"):
        strict_edge_alignment_check(
            target,
            edge_index=graph_edges[3:],
            edge_shift=graph_edges[:3],
            edge_type_idx=torch.zeros(graph_edges.shape[1], dtype=torch.long),
            atoms_list=["H", "H"],
            edge_types=["H-H"],
            matrix_name="hamiltonian",
            require_exact=True,
        )
