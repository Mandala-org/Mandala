import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN


@pytest.mark.integration
def test_force_prediction():
    """
    Test that force prediction runs without errors and returns a tensor of the correct shape.
    """
    from data.factory import DatasetFactory

    # 1. Create a dataset with enable_forces=True
    fac = DatasetFactory(Config(enable_forces=True))

    # Add a small, real data snapshot
    fac.add_snapshot(
        "data/small/H2O/original/H2O.matrix",
        "data/small/H2O/original/H2O.info.out",
    )
    train_ds, _, mapper = fac.create()

    # initialize model with default hyperparameters
    cfg = Config()
    model = E3GNN(mapper, cfg)
    # take first sample from training dataset
    x, y = train_ds[0]
    # predict forces
    forces = model.predict_forces(x)
    assert isinstance(forces, torch.Tensor)
    assert forces.shape == x["positions"].shape
