from __future__ import annotations

import time
from typing import Any, Callable

import pytorch_lightning as pl


class WallClockBudgetCallback(pl.Callback):
    """Stop training after the current epoch once a wall-clock budget is exceeded."""

    def __init__(
        self,
        *,
        budget_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        self.budget_seconds = float(budget_seconds)
        self._clock = clock or time.perf_counter
        self._fit_start_time: float | None = None
        self._stop_requested = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._fit_start_time = self._clock()
        self._stop_requested = False

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or self._stop_requested:
            return
        if self._fit_start_time is None:
            self._fit_start_time = self._clock()
        elapsed = self._clock() - self._fit_start_time
        if elapsed < self.budget_seconds:
            return
        self._stop_requested = True
        trainer.should_stop = True
        current_epoch = int(trainer.current_epoch)
        print(
            "--- Wall-clock budget reached after epoch "
            f"{current_epoch + 1} ({elapsed:.1f}s >= {self.budget_seconds:.1f}s); "
            "stopping before the next epoch and continuing with normal finalization. ---",
            flush=True,
        )
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            run.summary["timing/max_wall_clock_seconds"] = self.budget_seconds
            run.summary["timing/elapsed_at_stop_request_seconds"] = elapsed
            run.summary["timing/stopped_due_to_wall_clock_budget"] = True
            run.summary["timing/stop_epoch"] = current_epoch
            run.summary["timing/termination_reason"] = "wall_clock_budget"
        except Exception:
            pass


def _get_logger_run(trainer: pl.Trainer) -> Any | None:
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return None
    try:
        return logger.experiment
    except Exception:
        return None
