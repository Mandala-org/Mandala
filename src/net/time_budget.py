from __future__ import annotations

from datetime import datetime, timezone
import os
import socket
import time
from typing import Any, Callable

import pytorch_lightning as pl


class RunBookkeepingCallback(pl.Callback):
    """Record lightweight scheduler and runtime provenance in W&B summaries."""

    _SCHEDULER_ENV_KEYS = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "PBS_JOBID",
        "LSB_JOBID",
        "WANDB_SWEEP_ID",
        "WANDB_RUN_ID",
    )

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        super().__init__()
        self._clock = clock or time.perf_counter
        self._fit_start: float | None = None
        self._epoch_start: float | None = None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def on_fit_start(self, trainer, pl_module) -> None:
        self._fit_start = self._clock()
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            run.summary["runtime/hostname"] = socket.gethostname()
            run.summary["runtime/pid"] = os.getpid()
            run.summary["timing/fit_start_utc"] = self._utc_now()
            budget = getattr(pl_module.cfg, "max_wall_clock_seconds", None)
            if budget is not None:
                run.summary["timing/planned_fit_budget_seconds"] = float(budget)
            for key in self._SCHEDULER_ENV_KEYS:
                value = os.getenv(key)
                if value:
                    run.summary[f"runtime/env/{key.lower()}"] = value
        except Exception:
            pass

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = self._clock()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._epoch_start is None:
            return
        epoch_seconds = self._clock() - self._epoch_start
        elapsed = (
            self._clock() - self._fit_start if self._fit_start is not None else 0.0
        )
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            run.log(
                {
                    "epoch": int(trainer.current_epoch),
                    "timing/epoch_seconds": float(epoch_seconds),
                    "timing/fit_elapsed_seconds": float(elapsed),
                }
            )
            run.summary["timing/last_completed_epoch"] = int(trainer.current_epoch)
            run.summary["timing/last_epoch_seconds"] = float(epoch_seconds)
            run.summary["timing/fit_elapsed_seconds"] = float(elapsed)
        except Exception:
            pass

    def on_fit_end(self, trainer, pl_module) -> None:
        run = _get_logger_run(trainer)
        if run is None:
            return
        elapsed = (
            self._clock() - self._fit_start if self._fit_start is not None else 0.0
        )
        try:
            run.summary["timing/fit_end_utc"] = self._utc_now()
            run.summary["timing/fit_elapsed_seconds"] = float(elapsed)
            if not run.summary.get("timing/termination_reason"):
                run.summary["timing/termination_reason"] = "fit_completed"
        except Exception:
            pass

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            run.summary["timing/exception_utc"] = self._utc_now()
            run.summary["timing/termination_reason"] = "exception"
            run.summary["timing/exception_type"] = type(exception).__name__
            run.summary["timing/exception_message"] = str(exception)[:1000]
            run.summary["timing/exception_epoch"] = int(trainer.current_epoch)
        except Exception:
            pass


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
        self._fit_start_time = None
        self._stop_requested = False

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._fit_start_time is None:
            self._fit_start_time = self._clock()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or self._stop_requested:
            return
        if self._fit_start_time is None:
            return
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
