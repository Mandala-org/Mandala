import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN


@pytest.mark.unit
def test_stress_with_real_data(small_angular_dataset_e3nn):
    """
    Tests that a randomly initialized network produces non-zero forces
    using a real data sample from the E3GNNDataset. This ensures that
    the stress computation is functioning correctly.
    """

    # Set up config
    cfg = Config(
        cutoff_radius=5.0,
        l_max=1,
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        matrix_targets=["hamiltonian", "overlap", "density"],
        lr=1e-3,
        enable_stress=True,
        safety_checks=True,
    )

    train_ds, mapper, _ = small_angular_dataset_e3nn

    # Get a sample. The dataset should have set requires_grad on positions.
    x, _ = train_ds[0]
    x = dict(x)
    x["edge_shift"] = x["edge_shift"].clone()
    # The synthetic dataset otherwise has only zero-shift edges, for which the
    # energy is exactly independent of the periodic box.
    x["edge_shift"][:, 2] = torch.tensor([-1, 0, 0])
    x["edge_shift"][:, 3] = torch.tensor([1, 0, 0])

    # 2. Set up the model
    model = E3GNN(mapper, cfg)

    # 3. Predict stress and check that it's not all zeros
    stress = model.predict_stress(x)

    assert stress.shape == (3, 3)
    assert not torch.isnan(stress).any(), "Stress contains NaN values."
    assert (
        torch.count_nonzero(stress).item() > 0
    ), "Stress is exactly zero, gradients are likely detached."
