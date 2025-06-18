"""
Benchmarking callback for PyTorch Lightning.
Measures data loading time, forward pass, block mapping, observable evaluation,
batch and epoch timing, and outputs a YAML report at the end of training.
"""

import os
import time

import numpy as np
import pytorch_lightning as pl
import yaml


class BenchmarkCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_stats = []
        self.val_stats = []
        self._prev_train_end = None
        self._prev_val_end = None

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._train_start = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        t_end = time.perf_counter()
        t_start = self._train_start
        load_time = t_start - (self._prev_train_end or t_start)
        step_time = t_end - t_start
        times = getattr(pl_module, "_last_batch_times", {})
        stats = {
            "load_time": load_time,
            "forward_time": times.get("forward", 0.0),
            "map_time": times.get("map", 0.0),
            "obs_time": times.get("obs", 0.0),
            "step_time": step_time,
        }
        self.train_stats.append(stats)
        self._prev_train_end = t_end

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._val_start = time.perf_counter()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        t_end = time.perf_counter()
        t_start = self._val_start
        load_time = t_start - (self._prev_val_end or t_start)
        step_time = t_end - t_start
        times = getattr(pl_module, "_last_batch_times", {})
        stats = {
            "load_time": load_time,
            "forward_time": times.get("forward", 0.0),
            "map_time": times.get("map", 0.0),
            "obs_time": times.get("obs", 0.0),
            "step_time": step_time,
        }
        self.val_stats.append(stats)
        self._prev_val_end = t_end

    def on_fit_end(self, trainer, pl_module):
        # Summarize stats
        def summarize(stats_list):
            if not stats_list:
                return {}
            summary = {}
            keys = stats_list[0].keys()
            arrs = {k: np.array([s[k] for s in stats_list]) for k in keys}
            for k, arr in arrs.items():
                summary[k] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                }
            return summary

        report = {
            "run_name": getattr(trainer.logger, "name", None),
            "devices": getattr(trainer, "devices", None),
            "precision": getattr(trainer, "precision", None),
            "hparams": {},
            "train_summary": summarize(self.train_stats),
            "val_summary": summarize(self.val_stats),
        }
        # include config from WandbLogger if available
        try:
            cfg = trainer.logger.experiment.config
            report["hparams"] = dict(cfg)
        except Exception:
            pass

        # write YAML
        os.makedirs("benchmarks", exist_ok=True)
        fname = os.path.join("benchmarks", f"{report['run_name']}_bench.yaml")
        with open(fname, "w") as f:
            yaml.safe_dump(report, f)
