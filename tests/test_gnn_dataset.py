"""
Integration tests for the **E3GNNDataset** with two different graphs.

We critically check:

* self-edges are removed
* x_gnn edges form a strict subset of x_matrix edges
* per-edge - per-vector alignment (counts & one-hot types)
"""

from pathlib import Path
import torch
import pytest

from data.gnn_dataset import E3GNNDataset
from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot import Snapshot


@pytest.fixture(scope="module")
def dataset():
    snap_H20 = Snapshot.from_openmx(
        matrix_path=Path("data/small/H2O/original/H2O.matrix"),
        info_path=Path("data/small/H2O/original/H2O.info.out"),
    )

    mapper = BlockIrrepMapper(snap_H20.hamiltonian.orbital_cfg)

    return E3GNNDataset(
        [snap_H20],
        mapper,
        cutoff_gnn=4.0,
        cutoff_matrix=7.5,
    )


# --------------------------------------------------------------------------- #
def test_edge_sets(dataset):
    x_gnn, x_mat, y = dataset[0]

    # ----  no self-edges in either graph
    assert torch.all(x_gnn["edge_index"][0] != x_gnn["edge_index"][1])
    assert torch.all(x_mat["edge_index"][0] != x_mat["edge_index"][1])

    # ----  gnn edge set ⊂ matrix edge set
    gnn_edges = {tuple(e.tolist()) for e in x_gnn["edge_index"].t()}
    mat_edges = {tuple(e.tolist()) for e in x_mat["edge_index"].t()}
    assert gnn_edges.issubset(mat_edges)

    # ----  counts of off-diagonal overlap vectors = number of edges
    n_vec_gnn = sum(t.shape[0] for t in x_gnn["overlap_vectors_offdiag"].values())
    n_vec_mat = sum(t.shape[0] for t in x_mat["overlap_vectors_offdiag"].values())
    assert n_vec_gnn == x_gnn["edge_index"].shape[1]
    assert n_vec_mat == x_mat["edge_index"].shape[1]

    # ----  diagonal vectors unaffected by graph choice
    diag_cnt = sum(t.shape[0] for t in x_gnn["overlap_vectors_diag"].values())
    assert diag_cnt > 0
    assert diag_cnt == sum(t.shape[0] for t in x_mat["overlap_vectors_diag"].values())

    # ----  one-hot sanity (exactly one ‘1’ per row)
    oh = x_gnn["edge_one_hot"]
    assert torch.allclose(oh.sum(dim=1), torch.ones_like(oh[:, 0]))

    # ----  SH & radial embed sizes
    assert x_gnn["edge_sh"].shape[1] == dataset.sh_irreps.dim
    assert x_gnn["edge_length_emb"].shape[1] == dataset.n_radial
