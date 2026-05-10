from __future__ import annotations

from types import SimpleNamespace

from net.time_budget import WallClockBudgetCallback


def test_wall_clock_budget_requests_stop_after_validation_epoch():
    clock_values = iter([100.0, 112.0])
    callback = WallClockBudgetCallback(
        budget_seconds=10.0,
        clock=lambda: next(clock_values),
    )
    summary = {}
    trainer = SimpleNamespace(
        sanity_checking=False,
        should_stop=False,
        current_epoch=3,
        logger=SimpleNamespace(experiment=SimpleNamespace(summary=summary)),
    )

    callback.on_fit_start(trainer, SimpleNamespace())
    callback.on_train_epoch_start(trainer, SimpleNamespace())
    callback.on_validation_epoch_end(trainer, SimpleNamespace())

    assert trainer.should_stop is True
    assert summary["timing/max_wall_clock_seconds"] == 10.0
    assert summary["timing/stopped_due_to_wall_clock_budget"] is True
    assert summary["timing/stop_epoch"] == 3
    assert summary["timing/termination_reason"] == "wall_clock_budget"


def test_wall_clock_budget_ignores_validation_before_training_starts():
    callback = WallClockBudgetCallback(
        budget_seconds=10.0,
        clock=lambda: 100.0,
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        should_stop=False,
        current_epoch=0,
        logger=None,
    )

    callback.on_fit_start(trainer, SimpleNamespace())
    callback.on_validation_epoch_end(trainer, SimpleNamespace())

    assert trainer.should_stop is False
