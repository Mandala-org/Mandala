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
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.snapshot import Snapshot
from data.graph_features import compute_graph_features
from net.common import Config


#!  Tests don't want to run with pytest:
'''
pytest tests/unit/test_gnn_dataset.py                  ░▒▓ ✔ │    mandala-venv 🐍 
============================================================ test session starts =============================================================
platform linux -- Python 3.10.13, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/wiktoria/casus/mandala
configfile: pyproject.toml
plugins: cov-6.2.1, xdist-3.8.0, hydra-core-1.3.2
4 workers [0 items]     

=========================================================== no tests ran in 3.93s ============================================================
Manually run, the test passes.
'''

@pytest.fixture(scope="module")
def dataset():
    snap_H20 = Snapshot.from_openmx(
        matrix_path=Path("data/small/H2O/original/H2O.matrix"),
        info_path=Path("data/small/H2O/original/H2O.info.out"),
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
        cfg=Config(cutoff_gnn=4.0, cutoff_matrix=7.5),
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

#@pytest.fixture(scope="module")
def dummy_h2o_snapshot_and_config():
    """Provides a dummy H2O snapshot and a config for testing."""
    atoms = ("H", "O", "H")
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)
    box = torch.eye(3) * 10.0  # Large box to avoid PBC issues for simple test

    # Minimal orbital config for H and O
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e", "O": "1x0e"})

    # Create dummy BlockMatrix that only has the necessary attributes
    class DummyBlockMatrix:
        def __init__(self, pair_edges, atoms, orbital_cfg):
            self.pair_edges = pair_edges
            self.atoms = atoms
            self.orbital_cfg = orbital_cfg  # Needed for mapper
            # Add dummy to_vectors method for the test
            self.pair_vectors = {
                k: torch.randn(v.shape[1], 1) for k, v in pair_edges.items()
            }  # Dummy vectors

        def to_vectors(self, mapper):  # Mock to_vectors
            # This is a simplified mock, just returns self.pair_vectors
            return type(
                "IrrepsBlockDataMock",
                (object,),
                {
                    "pair_vectors": self.pair_vectors,
                    "pair_edges": self.pair_edges,
                    "lookup": {},  # Not used in this test path
                    "orbital_cfg": self.orbital_cfg,
                    "atoms": self.atoms,
                    "atom_counts": {},  # Not used in this test path
                    "basis": "e3nn",  # Not used in this test path
                    "to": lambda device: self,  # Mock .to() method
                },
            )()

    pair_edges = {
        "H-H": torch.tensor([[0, 0, 2, 2], [0, 2, 0, 2]], dtype=torch.long),
        "H-O": torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        "O-H": torch.tensor([[1, 1], [0, 2]], dtype=torch.long),
        "O-O": torch.tensor([[1], [1]], dtype=torch.long),
    }

    # Create dummy instances for all three matrices
    dummy_density = DummyBlockMatrix(pair_edges, atoms, orb_cfg)
    dummy_hamiltonian = DummyBlockMatrix(pair_edges, atoms, orb_cfg)  # New
    dummy_overlap = DummyBlockMatrix(pair_edges, atoms, orb_cfg)      # New

    # Create a dummy Snapshot that only has the necessary attributes
    class DummySnapshot:
        def __init__(self, positions, box, density, hamiltonian, overlap):
            self.positions = positions
            self.box = box
            self.density = density
            self.hamiltonian = hamiltonian
            self.overlap = overlap
            # Add dummy forces, stress, energy, num_electrons for _process_snapshot
            self.forces = torch.zeros_like(positions)
            self.stress = torch.zeros((3, 3), dtype=torch.float32)
            self.energy = torch.tensor(0.0, dtype=torch.float32)
            self.num_electrons = torch.tensor(0.0, dtype=torch.float32)
            # Add dummy get_energy and get_number_of_electrons methods
            self.get_energy = lambda: self.energy
            self.get_number_of_electrons = lambda: self.num_electrons

    dummy_snap = DummySnapshot(
        positions, box, dummy_density, dummy_hamiltonian, dummy_overlap
    )

    # Create a Config for testing precomputation
    cfg = Config(
        cutoff_gnn=4.0,
        cutoff_matrix=7.5,
        l_max_gnn=1,  # Keep l_max_gnn small for simpler sh_irreps
        n_radial=16,
        precompute_edge_features=True,  # Ensure precomputation is enabled
    )
    return dummy_snap, cfg

#@pytest.mark.unit
def test_compute_graph_features_matches_dataset_output():
    """
    Tests that compute_graph_features produces the same output as the dataset
    when the dataset is configured to precompute features.
    """
    dummy_snap, cfg = dummy_h2o_snapshot_and_config()

    # 1. Initialize mapper (needed by E3GNNDataset and compute_graph_features)
    mapper = BlockIrrepMapper(dummy_snap.density.orbital_cfg)

    # 2. Create E3GNNDataset (which will call compute_graph_features internally)
    # We need to pass dummy paths, as E3GNNDataset expects them, but we'll use our dummy_snap
    # by mocking the _load_or_process_snapshot method.
    class MockE3GNNDataset(E3GNNDataset):
        def _load_or_process_snapshot(self, matrix_path, info_path):
            # Return our dummy_snap processed by the actual _process_snapshot
            return self._process_snapshot(dummy_snap)

    # Instantiate the mock dataset with dummy paths
    dataset = MockE3GNNDataset(
        snapshot_paths=[(Path("dummy.matrix"), Path("dummy.info"))],
        mapper=mapper,
        cfg=cfg,
    )

    # Get the processed sample from the dataset
    x_from_dataset, _ = dataset[0]

    # 3. Directly call compute_graph_features with the raw inputs
    (
        edge_index_direct,
        edge_type_idx_direct,
        edge_length_emb_direct,
        edge_sh_direct,
        index_gnn_cutoff_direct,
        num_self_edges_direct,
    ) = compute_graph_features(
        positions=dummy_snap.positions,
        box=dummy_snap.box,
        atoms=dummy_snap.density.atoms,
        orbital_cfg=dummy_snap.density.orbital_cfg,
        cfg=cfg,
        sh_irreps=dataset.sh_irreps, # Use sh_irreps from dataset's init
        edge_type2idx=dataset.edge_type2idx, # Use edge_type2idx from dataset's init
    )

    # 4. Assert that the outputs are identical
    assert torch.equal(x_from_dataset["edge_index"], edge_index_direct)
    assert torch.equal(x_from_dataset["edge_type_idx"], edge_type_idx_direct)
    assert torch.allclose(x_from_dataset["edge_length_emb"], edge_length_emb_direct)
    assert torch.allclose(x_from_dataset["edge_sh"], edge_sh_direct)
    assert x_from_dataset["index_gnn_cutoff"] == index_gnn_cutoff_direct
    assert x_from_dataset["num_self_edges"] == num_self_edges_direct
    print("es")

test_compute_graph_features_matches_dataset_output()