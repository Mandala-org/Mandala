import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN


def add_test_envelope_metadata(x):
    edge_count = x["edge_index"].shape[1]
    params = torch.zeros(edge_count, 5, dtype=x["positions"].dtype)
    params[:, 1] = -2.0  # modest exponential decay
    params[:, 3] = 4.0
    params[:, 4] = -1.0
    x["edge_envelope_family"] = "slater_soft_cutoff"
    x["edge_envelope_pair_params"] = params


@pytest.mark.unit
@pytest.mark.parametrize("envelope_mode", ["off", "multiply_prediction"])
def test_non_zero_forces_with_real_data(small_angular_dataset_e3nn, envelope_mode):
    """
    Tests that a randomly initialized network produces non-zero forces
    using a real data sample from the E3GNNDataset. This ensures that
    the entire data processing pipeline correctly propagates gradients.
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
        enable_forces=True,
        hamiltonian_envelope_mode=envelope_mode,
        safety_checks=True,
    )

    train_ds, mapper, _ = small_angular_dataset_e3nn

    # Get a sample. The dataset should have set requires_grad on positions.
    x, _ = train_ds[0]
    x = dict(x)
    if envelope_mode != "off":
        add_test_envelope_metadata(x)

    # 2. Set up the model
    model = E3GNN(mapper, cfg)

    # 3. Predict forces and check that they are not all zero
    forces = model.predict_forces(x)

    assert forces.shape == x["positions"].shape
    assert not torch.isnan(forces).any(), "Forces contain NaN values."
    assert torch.isfinite(forces).all()
    assert not torch.allclose(
        forces, torch.zeros_like(forces)
    ), "Forces are all zero, gradients are likely detached."


@pytest.mark.unit
def test_energy_derivatives_require_all_three_matrix_predictions(
    small_angular_dataset_e3nn,
):
    _, mapper, _ = small_angular_dataset_e3nn
    cfg = Config(
        cutoff_radius=5.0,
        l_max=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        matrix_targets=["hamiltonian"],
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(mapper, cfg)

    with pytest.raises(ValueError, match="missing: density, overlap"):
        model.predictions_to_snapshot(
            {"hamiltonian": object()},
            {"positions": torch.zeros(2, 3), "box": torch.eye(3)},
        )
