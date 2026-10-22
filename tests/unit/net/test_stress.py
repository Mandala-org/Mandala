import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN


@pytest.mark.unit
def test_stress_with_real_data(small_angular_dataset_e3nn):
    """Periodic-cell stress matches diagonal and shear energy differences."""

    train_ds, mapper, _ = small_angular_dataset_e3nn
    x, _ = train_ds[0]
    x = dict(x)
    # A non-collinear molecule in a skew cell makes strain components mix;
    # otherwise the old collinear/cubic fixture only exercises xx stress.
    x["positions"] = torch.tensor(
        [[0.0, 0.0, 0.0], [1.7, 1.2, 0.8]], dtype=x["positions"].dtype
    )
    x["box"] = torch.tensor(
        [[4.0, 0.8, 0.4], [0.2, 4.5, 0.7], [0.5, 0.3, 5.0]],
        dtype=x["box"].dtype,
    )
    x["edge_shift"] = x["edge_shift"].clone()
    # This fixture has four directed edges. Give one pair opposite lattice
    # shifts so the energy depends on the cell through periodic geometry.
    x["edge_shift"][:, 2] = torch.tensor([-1, 0, 0])
    x["edge_shift"][:, 3] = torch.tensor([1, 0, 0])
    torch.testing.assert_close(
        x["edge_shift"][:, 2] + x["edge_shift"][:, 3],
        torch.zeros(3, dtype=x["edge_shift"].dtype),
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(743)
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
        model = E3GNN(mapper, cfg).double().eval()

        def cast_tree(value):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    return value.double()
                if value.is_complex():
                    return value.to(torch.complex128)
                return value
            if isinstance(value, dict):
                return {key: cast_tree(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cast_tree(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cast_tree(item) for item in value)
            return value

        x = cast_tree(x)
        stress = model.predict_stress(x)
        assert stress.shape == (3, 3)
        assert torch.isfinite(stress).all()
        torch.testing.assert_close(stress, stress.T, rtol=0, atol=1e-12)
        assert torch.count_nonzero(stress) >= 5
        assert stress[0, 1].abs() > 1e-7
        assert stress[0, 2].abs() > 1e-7
        assert stress[1, 2].abs() > 1e-7

        volume = torch.det(x["box"])
        base_box = x["box"]

        def energy_at_strain(strain):
            inputs = dict(x)
            inputs["box"] = base_box @ (torch.eye(3, dtype=torch.float64) + strain)
            # Static edges are retained, while geometry and radial caches are
            # removed so every strain evaluation must rebuild them from the
            # perturbed cell and the directed periodic shifts.
            for key in (
                "edge_vectors",
                "edge_lengths",
                "edge_sh",
                "edge_length_emb",
                "message_vectors",
                "message_lengths",
            ):
                inputs.pop(key, None)
            predictions = model(inputs, physical_derivatives=True)
            return model.predictions_to_snapshot(
                predictions, inputs["positions"], inputs["box"], x=inputs
            ).get_energy()

        epsilon = 1e-5
        for first in range(3):
            strain = torch.zeros((3, 3), dtype=torch.float64)
            strain[first, first] = epsilon
            finite_difference = (
                (energy_at_strain(strain) - energy_at_strain(-strain))
                / (2 * epsilon * volume)
            ).reshape(())
            torch.testing.assert_close(
                stress[first, first], finite_difference, rtol=2e-4, atol=2e-7
            )

        for first, second in ((0, 1), (0, 2), (1, 2)):
            strain = torch.zeros((3, 3), dtype=torch.float64)
            strain[first, second] = epsilon
            strain[second, first] = epsilon
            finite_difference = (
                (energy_at_strain(strain) - energy_at_strain(-strain))
                / (2 * epsilon * volume)
            ).reshape(())
            torch.testing.assert_close(
                2 * stress[first, second],
                finite_difference,
                rtol=3e-4,
                atol=3e-7,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
