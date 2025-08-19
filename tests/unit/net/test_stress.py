import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN
from data.factory import DatasetFactory


@pytest.mark.unit
def test_stress_with_real_data():
    """
    Tests that a randomly initialized network produces non-zero forces
    using a real data sample from the E3GNNDataset. This ensures that
    the stress computation is functioning correctly.
    """

    # Set up config
    cfg = Config(
        cutoff_gnn=3.0,
        cutoff_matrix=4.0,
        l_max_gnn=2,
        n_radial=64,
        num_layers_gnn=3,
        hidden_base_dim=16,
        num_layers_matrix=2,
        lr=1e-3,
        enable_stress=True,
        pedantic=True,
    )

    # 1. Create a DatasetFactory with position gradients enabled
    factory = DatasetFactory(cfg)

    # Add a small, real data snapshot
    factory.add_snapshot(
        "data/small/H2O/original/H2O.matrix",
        "data/small/H2O/original/H2O.info.out",
        # "data/big/silicon/900K/Si_DM",
        # "data/big/silicon/900K/info.txt",
    )

    # Create the dataset and mapper
    train_ds, _, mapper = factory.create()

    # Get a sample. The dataset should have set requires_grad on positions.
    x, _ = train_ds[0]

    # 2. Set up the model
    model = E3GNN(mapper, cfg)

    # 3. Predict stress and check that it's not all zeros
    stress = model.predict_stress(x)

    assert stress.shape == (3, 3)
    assert not torch.isnan(stress).any(), "Stress contains NaN values."
    assert not torch.allclose(
        stress, torch.zeros_like(stress)
    ), "Stress is all zero, gradients are likely detached."
