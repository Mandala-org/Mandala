import copy

import pytest
import torch

from net.training_policies import (
    SpeciesIrrepAffine,
    masked_error,
    make_optimizer,
    SmoothedPlateau,
)


def records():
    return [
        {
            "id": "train-a",
            "split": "train",
            "pair_type": torch.tensor([0, 0]),
            "coupled": torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]),
        }
    ]


def test_statistics_fit_only_exact_training_manifest_and_checkpoint_roundtrip():
    scale = SpeciesIrrepAffine(1, "1x0e+1x1o")
    scale.fit(records(), training_ids=["train-a"])
    torch.testing.assert_close(scale.mean[0], torch.tensor([2.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(
        scale.std[0, 1:],
        torch.full((3,), torch.tensor([2.0, 3.0, 4.0, 4.0, 5.0, 6.0]).std()),
    )
    restored = SpeciesIrrepAffine(1, "1x0e+1x1o")
    restored.load_state_dict(copy.deepcopy(scale.state_dict()))
    assert restored.training_manifest_hash == scale.training_manifest_hash
    torch.testing.assert_close(restored.std, scale.std)
    with pytest.raises(ValueError, match="manifest"):
        scale.fit(records(), training_ids=["train-a", "held-out"])
    leaked = records()
    leaked[0]["split"] = "val"
    with pytest.raises(ValueError, match="training"):
        scale.fit(leaked, training_ids=["train-a"])


@pytest.mark.parametrize("gradient_mode", ["identity", "physical"])
def test_scaling_value_and_surrogate_gradient(gradient_mode):
    scale = SpeciesIrrepAffine(1, "1x0e", gradient_mode=gradient_mode)
    scale.fitted.fill_(True)
    scale.std.fill_(3.0)
    scale.mean.fill_(2.0)
    x = torch.tensor([[4.0]], requires_grad=True)
    y = scale(x, torch.tensor([0]))
    torch.testing.assert_close(y, torch.tensor([[14.0]]))
    (gradient,) = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(
        gradient, torch.full_like(x, 1.0 if gradient_mode == "identity" else 3.0)
    )
    (actual_derivative,) = torch.autograd.grad(
        scale(x, torch.tensor([0]), physical_derivatives=True).sum(), x
    )
    torch.testing.assert_close(actual_derivative, torch.full_like(x, 3.0))


def test_loss_reduces_selected_real_components_globally():
    x = torch.tensor([[1.0 + 2.0j, 5.0 + 6.0j], [3.0 + 4.0j, 7.0 + 8.0j]])
    mask = torch.tensor([[True, False], [True, False]])
    torch.testing.assert_close(
        masked_error(x, torch.zeros_like(x), mask), torch.tensor(15.0)
    )
    torch.testing.assert_close(
        masked_error(x, torch.zeros_like(x), mask, kind="mae"),
        (torch.tensor(5.0).sqrt() + 5) / 2,
    )
    assert masked_error(
        x, torch.zeros_like(x), mask, complex_reduction="components"
    ) == torch.tensor(7.5)
    assert masked_error(
        x, torch.zeros_like(x), mask, kind="mae", complex_reduction="components"
    ) == torch.tensor(2.5)
    with pytest.raises(ValueError, match="no supervised"):
        masked_error(x, x, torch.zeros_like(mask))


def test_optimizer_defaults_and_smoothed_scheduler_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = make_optimizer(
        [parameter], {"kind": "adam", "lr": 0.01, "weight_decay": 0}
    )
    assert isinstance(optimizer, torch.optim.Adam)
    scheduler = SmoothedPlateau(optimizer, ema_alpha=0.1, patience=0, factor=0.5)
    scheduler.step(1.0)
    scheduler.step(3.0)
    assert scheduler.smoothed_metric == pytest.approx(1.2)
    restored = SmoothedPlateau(optimizer, ema_alpha=0.5)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.smoothed_metric == scheduler.smoothed_metric
    assert restored.ema_alpha == 0.1


def test_lr_termination_checked_after_scheduler_update():
    from types import SimpleNamespace
    from net.e3gnn import E3GNN

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = make_optimizer([parameter], {"kind": "adam", "lr": 0.01})
    scheduler = SmoothedPlateau(optimizer, ema_alpha=1.0, patience=0, factor=0.5)
    module = SimpleNamespace(
        cfg=SimpleNamespace(pipeline={}, optimization={"stop_lr": 0.005}),
        trainer=SimpleNamespace(should_stop=False),
    )
    E3GNN.lr_scheduler_step(module, scheduler, 1.0)
    assert not module.trainer.should_stop
    E3GNN.lr_scheduler_step(module, scheduler, 2.0)
    assert optimizer.param_groups[0]["lr"] == 0.005
    assert module.trainer.should_stop


def test_surrogate_gradient_differs_from_finite_difference_physical_gradient():
    scale = SpeciesIrrepAffine(1, "1x0e", gradient_mode="identity").double()
    scale.fitted.fill_(True)
    scale.std.fill_(4.0)
    x = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    pair = torch.tensor([0])
    finite_difference = (scale(x + 1e-5, pair) - scale(x - 1e-5, pair)) / 2e-5
    (physical,) = torch.autograd.grad(
        scale(x, pair, physical_derivatives=True).sum(), x
    )
    (surrogate,) = torch.autograd.grad(scale(x, pair).sum(), x)
    torch.testing.assert_close(physical, finite_difference)
    assert not torch.allclose(surrogate, finite_difference)
