from __future__ import annotations

import signal

import pytorch_lightning as pl

_INTERRUPT_REQUESTED = False
_HANDLERS_INSTALLED = False


def interrupt_requested() -> bool:
    return _INTERRUPT_REQUESTED


def clear_interrupt_request() -> None:
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False


def request_interrupt() -> None:
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True


def install_signal_handlers(label: str = "Process") -> None:
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return

    def _handle_signal(signum, frame):  # noqa: ARG001
        global _INTERRUPT_REQUESTED
        if _INTERRUPT_REQUESTED:
            return
        _INTERRUPT_REQUESTED = True
        sig_name = signal.Signals(signum).name
        print(
            f"--- {label} received {sig_name}; requesting graceful stop ---", flush=True
        )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    _HANDLERS_INSTALLED = True


class GracefulInterruptCallback(pl.Callback):
    def __init__(self, label: str = "Process") -> None:
        super().__init__()
        self.label = label
        self._printed = False

    def _maybe_stop(self, trainer: pl.Trainer) -> None:
        if not interrupt_requested():
            return
        trainer.should_stop = True
        if not self._printed:
            print(
                f"--- {self.label} stopping gracefully at the next safe point ---",
                flush=True,
            )
            self._printed = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):  # noqa: ARG002
        self._maybe_stop(trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):  # noqa: ARG002
        self._maybe_stop(trainer)

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        self._maybe_stop(trainer)
