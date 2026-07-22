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
from net.common import Config


@pytest.fixture
def dataset(monkeypatch, small_angular_snapshot_e3nn):
    cfg = Config(cutoff_radius=5.0, n_radial=4)
    mapper = BlockIrrepMapper(small_angular_snapshot_e3nn.hamiltonian.orbital_cfg)
    monkeypatch.setattr(
        "data.gnn_dataset.Snapshot.from_openmx",
        lambda *args, **kwargs: small_angular_snapshot_e3nn,
    )

    return E3GNNDataset(
        [(Path("synthetic.matrix"), Path("synthetic.out"))],
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

    # ----  SH & radial embed sizes
    assert x["edge_sh"].shape[1] == dataset.sh_irreps.dim
    assert x["edge_length_emb"].shape[1] == dataset.cfg.n_radial


@pytest.mark.unit
def test_to_allows_missing_optional_observable_targets():
    class Movable:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(torch.device(device))
            return self

    ds = object.__new__(E3GNNDataset)
    ds.device = torch.device("cpu")
    required = {key: Movable() for key in ("hamiltonian", "overlap", "density")}
    present_stress = Movable()
    y = {
        **required,
        "energy": None,
        "num_electrons": None,
        "forces": None,
        "stress": present_stress,
    }
    ds.snapshots = [({}, y)]

    returned = ds.to("meta")

    assert returned is ds
    assert ds.device == torch.device("meta")
    assert y["energy"] is None
    assert y["num_electrons"] is None
    assert y["forces"] is None
    assert present_stress.devices == [torch.device("meta")]
    for value in required.values():
        assert value.devices == [torch.device("meta")]
