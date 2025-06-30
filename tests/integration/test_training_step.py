import pytest
import torch

from net.common import HyperParams
from net.e3gnn import E3GNN


@pytest.mark.integration
def test_training_step_smoke(factory_results):
    """
    Smoke-test that training_step runs without shape errors and returns a scalar loss.
    """
    from omegaconf import OmegaConf
    from dataclasses import asdict

    train_ds, _, mapper = factory_results
    # initialize model with default hyperparameters
    hp = HyperParams()
    mock_cfg = OmegaConf.create(
        {"model": asdict(hp), "training": {"lr": 1e-3}, "logging": {"pedantic": False}}
    )
    model = E3GNN(mapper, train_ds.edge_types, mock_cfg, device="cpu")
    # take first sample from training dataset
    x_gnn, x_mat, y = train_ds[0]
    # call training_step and verify scalar loss
    loss = model.training_step((x_gnn, x_mat, y), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
