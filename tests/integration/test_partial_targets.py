import pytest
import torch
from pathlib import Path

from net.common import Config
from data.factory import DatasetFactory
from net.e3gnn import E3GNN
from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot import Snapshot


@pytest.fixture(scope="module")
def h2o_training_data():
    # Create a simple H2O dataset
    matrix_path = Path("data/small/H2O/original/H2O.matrix")
    info_path = Path("data/small/H2O/original/H2O.info.out")
    cfg = Config()

    return matrix_path, info_path, cfg


@pytest.mark.integration
def test_hamiltonian_only_training_step(h2o_training_data):
    """
    Tests that a full training step can be run when the model is configured
    to train only on the Hamiltonian matrix.
    """
    matrix_path, info_path, base_cfg = h2o_training_data

    # 1. Configure the system for Hamiltonian-only training
    base_cfg.available_targets = ["hamiltonian"]

    # 2. Create the Dataset
    factory = DatasetFactory(base_cfg)
    factory.add_snapshot(matrix_path, info_path, cfg=base_cfg)
    train_ds, _, mapper = factory.create()

    # Get a sample from the dataset
    x, y = train_ds[0]

    # Assert that the dataset correctly provides only the Hamiltonian target
    assert "hamiltonian" in y
    assert "overlap" not in y
    assert "density" not in y
    assert "energy" not in y # Should not be computed as density is missing
    assert "num_electrons" not in y # Should not be computed as overlap is missing

    # 3. Create the E3GNN Model
    model = E3GNN(
        mapper=mapper,
        cfg=base_cfg,
    )

    # Assert that the model correctly built only one head
    assert list(model.heads.keys()) == ["hamiltonian"]

    # 4. Run a single training step
    loss = model.training_step((x, y), batch_idx=0)

    # Assert that the loss is a valid number
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss > 0