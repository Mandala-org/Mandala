import pytest
import torch

from net.common import HyperParams
from net.e3gnn import E3GNN


@pytest.mark.integration
def test_force_prediction(factory_results):
    """
    Test that force prediction runs without errors and returns a tensor of the correct shape.
    """
    from omegaconf import OmegaConf
    from dataclasses import asdict
    from data.factory import DatasetFactory
    from pathlib import Path

    # Create a new dataset with enable_positions_grad=True
    fac = DatasetFactory(enable_positions_grad=True)
    fac.add_snapshot(
        Path("data/small/H2O/original/H2O.matrix"),
        Path("data/small/H2O/original/H2O.info.out"),
    )
    train_ds, _, mapper = fac.create()

    # initialize model with default hyperparameters
    hp = HyperParams()
    mock_cfg = OmegaConf.create(
        {"model": asdict(hp), "training": {"lr": 1e-3}, "logging": {"pedantic": False}}
    )
    model = E3GNN(mapper, train_ds.edge_types, mock_cfg, device="cpu")
    # take first sample from training dataset
    x_gnn, x_mat, y = train_ds[0]
    # predict forces
    forces = model.predict_forces(x_gnn, x_mat)
    assert isinstance(forces, torch.Tensor)
    assert forces.shape == x_gnn["positions"].shape
