import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN


@pytest.mark.unit
def test_non_zero_forces_with_real_data(small_angular_dataset_e3nn):
    """Dataset-to-energy forces match a centered coordinate difference."""

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
        safety_checks=True,
    )

    train_ds, mapper, _ = small_angular_dataset_e3nn

    # Keep the graph and its periodic shifts fixed while perturbing only
    # coordinates. Double precision makes the central difference useful as an
    # oracle for the force sign and magnitude.
    x, _ = train_ds[0]
    x = {
        key: (
            value.to(torch.float64)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value.clone() if isinstance(value, torch.Tensor) else value
        )
        for key, value in x.items()
    }

    # 2. Set up the model
    torch.manual_seed(31)
    model = E3GNN(mapper, cfg).double().eval()

    # 3. Predict forces and check that they are not all zero
    forces = model.predict_forces(x)

    assert forces.shape == x["positions"].shape
    assert not torch.isnan(forces).any(), "Forces contain NaN values."
    assert not torch.allclose(
        forces, torch.zeros_like(forces)
    ), "Forces are all zero, gradients are likely detached."

    def energy_at(positions):
        perturbed = dict(x, positions=positions)
        predictions = model(perturbed, physical_derivatives=True)
        return model.predictions_to_snapshot(
            predictions, positions, perturbed["box"], x=perturbed
        ).get_energy()

    # Check the component with the largest predicted magnitude to avoid a
    # near-zero finite-difference comparison on this randomly initialized net.
    atom, axis = torch.unravel_index(forces.abs().argmax(), forces.shape)
    step = 1e-5
    plus = x["positions"].clone()
    minus = x["positions"].clone()
    plus[atom, axis] += step
    minus[atom, axis] -= step
    finite_difference_force = -(energy_at(plus) - energy_at(minus)) / (2 * step)
    torch.testing.assert_close(
        forces[atom, axis], finite_difference_force.squeeze(), rtol=2e-3, atol=2e-5
    )


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
            torch.zeros(2, 3),
            torch.eye(3),
        )
