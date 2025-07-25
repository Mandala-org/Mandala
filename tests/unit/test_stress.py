import torch

from net.common import Config
from net.e3gnn import E3GNN
from data.factory import DatasetFactory


def test_stress_with_real_data():
    """
    Tests that a randomly initialized network produces non-zero forces
    using a real data sample from the E3GNNDataset. This ensures that
    the stress computation is functioning correctly.
    """

    # 2. Set up config
    cfg = {
        "hidden_base_dim": 16,
        "l_max": 2,
        "n_radial": 64,
        "enable_stress": True,
        "cutoff_gnn": 3.0,
        "cutoff_matrix": 4.0,
        "lr": 1e-3,
        "pedantic": True,  # Enable pedantic checks
    }

    cfg = Config(**cfg)

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

    model = E3GNN(mapper, cfg)

    # 3. Predict stress and check that it's not all zeros
    stress = model.predict_stress(x)

    assert stress.shape == (3, 3)
    assert not torch.allclose(
        stress, torch.zeros_like(stress)
    ), "Stress is all zero, gradients are likely detached."
