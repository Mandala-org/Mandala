from data.gnn_dataset import E3GNNDataset
from core.orbital_irrep_config import OrbitalIrrepConfig, BlockIrrepMapper
from data.snapshot import Snapshot
from pathlib import Path
import pytest
import torch
from net.common import Config
from net.e3gnn import E3GNN

@pytest.fixture(scope="module")
def dummy_h2o_integration_data():
    """Provides a dummy H2O snapshot and mapper for integration testing."""
    atoms = ("H", "O", "H")
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)
    box = torch.eye(3) * 10.0

    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e", "O": "1x0e"})

    class DummyBlockMatrix:
        def __init__(self, pair_edges, atoms, orbital_cfg):
            self.pair_edges = pair_edges
            self.atoms = atoms
            self.orbital_cfg = orbital_cfg
            self.pair_vectors = {k: torch.randn(v.shape[1], 1) for k, v in pair_edges.items()}

        def to_vectors(self, mapper):
            return type('IrrepsBlockDataMock', (object,), {
                'pair_vectors': self.pair_vectors,
                'pair_edges': self.pair_edges,
                'lookup': {},
                'orbital_cfg': self.orbital_cfg,
                'atoms': self.atoms,
                'atom_counts': {},
                'basis': 'e3nn',
                'to': lambda device: self
            })()

    pair_edges = {
        "H-H": torch.tensor([[0, 0, 2, 2], [0, 2, 0, 2]], dtype=torch.long),
        "H-O": torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        "O-H": torch.tensor([[1, 1], [0, 2]], dtype=torch.long),
        "O-O": torch.tensor([[1], [1]], dtype=torch.long),
    }

    dummy_density = DummyBlockMatrix(pair_edges, atoms, orb_cfg)
    dummy_hamiltonian = DummyBlockMatrix(pair_edges, atoms, orb_cfg)
    dummy_overlap = DummyBlockMatrix(pair_edges, atoms, orb_cfg)

    class DummySnapshot:
        def __init__(self, positions, box, density, hamiltonian, overlap):
            self.positions = positions
            self.box = box
            self.density = density
            self.hamiltonian = hamiltonian
            self.overlap = overlap
            self.forces = torch.zeros_like(positions)
            self.stress = torch.zeros((3, 3), dtype=torch.float32)
            self.energy = torch.tensor(0.0, dtype=torch.float32)
            self.num_electrons = torch.tensor(0.0, dtype=torch.float32)
            self.get_energy = lambda: self.energy
            self.get_number_of_electrons = lambda: self.num_electrons

    dummy_snap = DummySnapshot(positions, box, dummy_density, dummy_hamiltonian, dummy_overlap)
    mapper = BlockIrrepMapper(orb_cfg)

    return dummy_snap, mapper

@pytest.mark.integration
def test_e3gnn_on_the_fly_features():
    """
    Tests that E3GNN produces identical outputs whether features are precomputed
    or computed on-the-fly.
    """
    dummy_snap, mapper = dummy_h2o_integration_data()

    # --- Setup for precomputed features ---
    cfg_precompute = Config(
        cutoff_gnn=4.0,
        cutoff_matrix=7.5,
        l_max_gnn=1,
        n_radial=16,
        precompute_edge_features=True,  # Precompute
    )

    # Mock the dataset to use our dummy snapshot
    class MockE3GNNDatasetPrecompute(E3GNNDataset):
        def _load_or_process_snapshot(self, matrix_path, info_path):
            return self._process_snapshot(dummy_snap)

    dataset_precompute = MockE3GNNDatasetPrecompute(
        snapshot_paths=[(Path("dummy.matrix"), Path("dummy.info"))],
        mapper=mapper,
        cfg=cfg_precompute,
    )
    x_precompute, _ = dataset_precompute[0]
    model_precompute = E3GNN(mapper, cfg_precompute)

    # --- Setup for on-the-fly features ---
    cfg_onthefly = Config(
        cutoff_gnn=4.0,
        cutoff_matrix=7.5,
        l_max_gnn=1,
        n_radial=16,
        precompute_edge_features=False,  # On-the-fly
    )

    # Mock the dataset to use our dummy snapshot
    class MockE3GNNDatasetOntheFly(E3GNNDataset):
        def _load_or_process_snapshot(self, matrix_path, info_path):
            return self._process_snapshot(dummy_snap)

    dataset_onthefly = MockE3GNNDatasetOntheFly(
        snapshot_paths=[(Path("dummy.matrix"), Path("dummy.info"))],
        mapper=mapper,
        cfg=cfg_onthefly,
    )
    x_onthefly, _ = dataset_onthefly[0]
    model_onthefly = E3GNN(mapper, cfg_onthefly)

    # --- Run forward passes ---
    preds_precompute = model_precompute(x_precompute)
    preds_onthefly = model_onthefly(x_onthefly)

    # --- Assertions ---
    # Compare predictions for Hamiltonian, Overlap, and Density
    for matrix_name in ["hamiltonian", "overlap", "density"]:
        pred_precompute_vectors = preds_precompute[matrix_name].pair_vectors
        pred_onthefly_vectors = preds_onthefly[matrix_name].pair_vectors

        # Check that keys are the same
        assert set(pred_precompute_vectors.keys()) == set(pred_onthefly_vectors.keys())

        # Check that vectors for each key are numerically close
        for key in pred_precompute_vectors.keys():
            assert torch.allclose(
                pred_precompute_vectors[key],
                pred_onthefly_vectors[key],
                atol=1e-6,  # Adjust tolerance if needed for floating point
                rtol=1e-5,
            ), f"Mismatch in {matrix_name} for key {key} between precompute and on-the-fly"


