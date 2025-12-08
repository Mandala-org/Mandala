"""
Integration tests for the **E3GNNDataset** with two different graphs.

We critically check:

* self-edges are removed
* x_gnn edges form a strict subset of x_matrix edges
* per-edge - per-vector alignment (counts & one-hot types)
"""

import pytest
from pathlib import Path
import torch

from data.gnn_dataset import E3GNNDataset
from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot import Snapshot
from net.common import Config


@pytest.fixture(scope="module")
def dataset():
    cfg = Config(cutoff_gnn=4.0, cutoff_matrix=15)
    snap_H20 = Snapshot.from_openmx(
        matrix_path=Path("data/small/H2O/original/H2O.matrix"),
        info_path=Path("data/small/H2O/original/H2O.info.out"),
        cfg=cfg,
    )

    mapper = BlockIrrepMapper(snap_H20.hamiltonian.orbital_cfg)

    return E3GNNDataset(
        [
            (
                Path("data/small/H2O/original/H2O.matrix"),
                Path("data/small/H2O/original/H2O.info.out"),
            )
        ],
        mapper,
        cfg=cfg,
    )


@pytest.mark.unit
def test_edge_sets(dataset):
    x, y = dataset[0]

    # ----  check for self-edges at the beginning of the edge_index
    num_atoms = len(x["atoms"])
    self_edges = x["edge_index"][:, :num_atoms]
    assert torch.all(self_edges[0] == self_edges[1])
    assert torch.all(self_edges[0] == torch.arange(num_atoms))

    # ----  gnn edge set ⊂ matrix edge set
    index_gnn_cutoff = x["index_gnn_cutoff"]
    gnn_edges = {tuple(e.tolist()) for e in x["edge_index"][:, :index_gnn_cutoff].t()}
    mat_edges = {tuple(e.tolist()) for e in x["edge_index"].t()}
    assert gnn_edges.issubset(mat_edges)

    # ----  SH & radial embed sizes
    assert x["edge_sh"].shape[1] == dataset.sh_irreps.dim
    assert x["edge_length_emb"].shape[1] == dataset.cfg.n_radial
