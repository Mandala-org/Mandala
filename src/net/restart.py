"""Epoch-boundary recovery for long, segmented training runs.

Best-model files are for selection. Only latest_checkpoint.pt is a recovery
checkpoint: written before an epoch starts or after fit has finished, never
inside validation (before ReduceLROnPlateau has advanced).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler):
    """Shuffle by absolute epoch, independent of process restarts."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self), generator=generator).tolist())


class EpochBoundaryCheckpoint(pl.Callback):
    """Persist selection state and restore RNG after startup/loader setup."""

    def __init__(self, output_dir: Path, monitor: str):
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.best_score = None
        self.best_epoch = None
        self._rng = None
        self._boundary = False
        self._at_epoch_start = False

    @property
    def state_key(self):
        return f"EpochBoundaryCheckpoint[{self.monitor}]"

    def state_dict(self):
        return {"best_score": self.best_score, "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict):
        self.best_score = state_dict["best_score"]
        self.best_epoch = state_dict["best_epoch"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self._boundary and self._at_epoch_start:
            # Lightning calls this hook after marking the next epoch "ready",
            # but before resetting the previous epoch's batch counters. Encode
            # the equivalent completed boundary (as on_fit_end does), otherwise
            # recovery can skip the epoch. Fail closed on an unfamiliar schema.
            progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]
            for key in ("total", "current"):
                counts = progress[key]
                completed = counts["completed"]
                if not (
                    counts["ready"] == completed + 1
                    and counts["started"] == counts["processed"] == completed
                ):
                    raise RuntimeError(
                        "Unsupported Lightning epoch-boundary checkpoint state."
                    )
                counts["ready"] = completed
        if self._boundary and self.best_score is not None:
            # Validation can publish a newer best before a crash. Bundle the
            # matching best state with the recovery boundary so rollback never
            # reads a selection file from the abandoned/replayed epoch.
            checkpoint["boundary_best_checkpoint"] = torch.load(
                self.output_dir / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
        checkpoint["epoch_boundary_resume"] = {
            "schema_version": 1,
            "safe_to_resume": self._boundary,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get("epoch_boundary_resume", {})
        if not state.get("safe_to_resume"):
            raise ValueError(
                "Continuation requires an epoch-boundary recovery checkpoint."
            )
        self._rng = state
        best = checkpoint.get("boundary_best_checkpoint")
        if best is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.output_dir / "best_model.recovery.tmp"
            torch.save(best, temporary)
            os.replace(temporary, self.output_dir / "best_model.pt")

    def _save(self, trainer, filename, *, boundary=False):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / filename
        temporary = destination.with_suffix(".tmp")
        self._boundary = boundary
        try:
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, destination)
        finally:
            self._boundary = False

    def on_train_epoch_start(self, trainer, pl_module):
        sampler = trainer.train_dataloader.sampler
        if isinstance(sampler, EpochShuffleSampler):
            sampler.set_epoch(trainer.current_epoch)
        if self._rng is not None:
            random.setstate(self._rng["python"])
            np.random.set_state(self._rng["numpy"])
            torch.set_rng_state(self._rng["torch"].cpu())
            if self._rng["cuda"]:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in self._rng["cuda"]]
                )
            self._rng = None
        self._at_epoch_start = True
        try:
            self._save(trainer, "latest_checkpoint.pt", boundary=True)
        finally:
            self._at_epoch_start = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        score = float(trainer.callback_metrics[self.monitor])
        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite checkpoint monitor: {self.monitor}")
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_epoch = int(trainer.current_epoch)
            self._save(trainer, "best_model.pt")

    def on_fit_end(self, trainer, pl_module):
        self._save(trainer, "latest_checkpoint.pt", boundary=True)

    # On interruption preserve the last complete boundary; replay its epoch.
    # Never overwrite it with a partially completed optimizer/scheduler state.
