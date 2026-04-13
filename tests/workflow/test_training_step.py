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
    fac = DatasetFactory(Config(enable_forces=True, cutoff_radius=5.0, verbosity=0))

    # Add a small, real data snapshot
    fac.add_snapshot(
        "data/small/H2O/original/H2O.matrix",
        "data/small/H2O/original/H2O.info.out",
    )
    train_ds, _, mapper = fac.create()

    # initialize model with default hyperparameters
    cfg = Config(enable_forces=True, cutoff_radius=5.0, safety_checks=True, verbosity=0)
    model = E3GNN(mapper, cfg)
    # take first sample from training dataset
    x, y = train_ds[0]
    # predict forces
    forces = model.predict_forces(x)
    assert isinstance(forces, torch.Tensor)
    assert forces.shape == x["positions"].shape
    assert torch.isfinite(forces).all()


@pytest.mark.integration
def test_training_step_with_force_loss():
    from data.factory import DatasetFactory

    fac = DatasetFactory(
        Config(enable_forces=True, train_on_forces=True, cutoff_radius=5.0, verbosity=0)
    )
    fac.add_snapshot(
        "data/small/H2O/original/H2O.matrix",
        "data/small/H2O/original/H2O.info.out",
    )
    train_ds, _, mapper = fac.create()

    cfg = Config(
        enable_forces=True,
        train_on_forces=True,
        loss_coef_forces=1e-6,
        cutoff_radius=5.0,
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(mapper, cfg)
    batch = train_ds[0]

    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
