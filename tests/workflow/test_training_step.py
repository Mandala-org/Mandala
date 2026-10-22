from types import SimpleNamespace

import pytest
import torch

from net.common import Config
from net.e3gnn import E3GNN
from data.factory import DatasetFactory
from data.snapshot import Snapshot

pytestmark = [pytest.mark.integration, pytest.mark.workflow]


def _tiny_config(**overrides):
    values = dict(
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
        safety_checks=True,
        log_per_irrep_metrics=False,
        log_hamiltonian_irrep_contrib_metrics=False,
        log_hamiltonian_pair_contrib_metrics=False,
        verbosity=0,
    )
    values.update(overrides)
    return Config(**values)


@pytest.fixture
def tiny_dataset(monkeypatch, small_angular_snapshot_e3nn):
    monkeypatch.setattr(
        Snapshot,
        "from_openmx",
        lambda *args, **kwargs: small_angular_snapshot_e3nn,
    )
    monkeypatch.setattr(
        DatasetFactory,
        "_load_info",
        lambda self, *paths: SimpleNamespace(orbital_set={"H": "1s1p"}),
    )
    fac = DatasetFactory(
        _tiny_config(
            enable_forces=True,
            matrix_targets=["hamiltonian", "overlap", "density"],
        )
    )
    fac.add_snapshot("synthetic.matrix", "synthetic.out")
    train_ds, _, mapper = fac.create()
    return train_ds, mapper


@pytest.mark.integration
def test_training_step_with_force_loss(tiny_dataset):
    train_ds, mapper = tiny_dataset
    torch.manual_seed(433)

    cfg = _tiny_config(
        enable_forces=True,
        train_on_forces=True,
        loss_coef_forces=1e-6,
        matrix_targets=["hamiltonian", "overlap", "density"],
    )
    model = E3GNN(mapper, cfg)
    model.log_dict = lambda *args, **kwargs: None
    x, y = train_ds[0]
    y = dict(y, forces=torch.full_like(x["positions"], 0.25))
    batch = (x, y)
    parameters = tuple(model.parameters())
    model.cfg.loss_coef_forces = 0.0
    baseline = model.training_step(batch, 0)
    baseline_grad = torch.autograd.grad(baseline, parameters, allow_unused=True)

    coefficient = 2.3
    forces = model.predict_forces(x)
    assert torch.isfinite(forces).all()
    expected_force_loss = coefficient * (forces - y["forces"]).square().mean()
    expected_grad = torch.autograd.grad(
        expected_force_loss, parameters, allow_unused=True
    )
    assert any(grad is not None and grad.abs().max() > 1e-10 for grad in expected_grad)

    model.cfg.loss_coef_forces = coefficient
    total = model.training_step(batch, 0)
    torch.testing.assert_close(
        total - baseline, expected_force_loss, rtol=1e-4, atol=1e-6
    )
    total_grad = torch.autograd.grad(total, parameters, allow_unused=True)
    for parameter, initial, actual, force in zip(
        parameters, baseline_grad, total_grad, expected_grad
    ):
        zero = torch.zeros_like(parameter)
        torch.testing.assert_close(
            (zero if actual is None else actual)
            - (zero if initial is None else initial),
            zero if force is None else force,
            rtol=2e-3,
            atol=2e-6,
        )


@pytest.mark.integration
def test_training_step_stops_on_nan_loss(tiny_dataset):
    train_ds, mapper = tiny_dataset

    cfg = _tiny_config(loss_l1_fraction=0.0)
    model = E3GNN(mapper, cfg)
    batch = train_ds[0]

    monkey_trainer = SimpleNamespace(should_stop=False)
    object.__setattr__(model, "_trainer", monkey_trainer)
    model.log_dict = lambda *args, **kwargs: None
    model._weighted_edge_block_losses = lambda **kwargs: (
        torch.tensor(float("nan")),
        torch.tensor(float("nan")),
    )

    loss = model.training_step(batch, 0)

    assert loss is None
    assert model._nan_loss_detected is True
    assert monkey_trainer.should_stop is True
