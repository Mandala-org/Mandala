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
from typing import Any


class BenchmarkCallback(pl.Callback):
    """
    Benchmarking callback for PyTorch Lightning with configurable verbosity:
      0 = disabled
      1 = run-level summary only
      2 = + per-epoch summaries
      3 = + operator-level profiling via torch.profiler
    """

    def __init__(self, verbosity: int = 1, log_activation_mag: bool = False):
        super().__init__()
        self.verbosity = int(verbosity)
        # whether to collect activation magnitude statistics
        self.log_activation_mag = bool(log_activation_mag)
        # overall per-batch stats
        self.train_stats: list[dict] = []
        self.val_stats: list[dict] = []
        self._prev_train_end: float | None = None
        self._prev_val_end: float | None = None
        # per-epoch buffers (v>=2)
        if self.verbosity >= 2:
            self._epoch_train_stats: dict[int, dict] = {}
            self._epoch_val_stats: dict[int, dict] = {}
        # prepare for loader profiling
        self.loader_times: list[float] = []
        # torch.profiler handle (v>=3)
        self.profiler = None
        # record environment
        self.environment: dict = {}
        # activation magnitudes storage
        self.train_activation_mags: dict[str, list[np.ndarray]] = {}
        self.val_activation_mags: dict[str, list[np.ndarray]] = {}

    # helper to summarize a list of per-batch dicts
    def _summarize(self, lst: list[dict]) -> dict:
        if not lst:
            return {}
        summary: dict = {}
        keys = lst[0].keys()
        arrs = {k: np.array([s.get(k, 0.0) for s in lst]) for k in keys}
        for k, arr in arrs.items():
            summary[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "median": float(np.median(arr)),
                "max": float(arr.max()),
            }
            # summary[k] = {}
            # summary[k]["mean"] = float(arr.mean())
            # summary[k]["std"] = float(arr.std())
            # summary[k]["min"] = float(arr.min())
            # summary[k]["median"] = float(np.median(arr))
            # summary[k]["max"] = float(arr.max())

        return summary

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.verbosity == 0:
            return
        # record environment and timestamp
        import datetime as dt
        import platform
        import torch

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_info: dict = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                gpu_info[i] = {"name": prop.name, "total_memory": prop.total_memory}
        self.environment = {
            "timestamp": ts,
            "cpu_count": os.cpu_count(),
            "cpu_processor": platform.processor(),
            "gpu": gpu_info,
        }
        # prepare dataset profiling
        self.loader_times = []
        try:
            tlds = trainer.train_dataloader
            tlds = tlds if isinstance(tlds, (list, tuple)) else [tlds]
        except Exception:
            tlds = []
        for ld in tlds:
            try:
                ld.dataset.loader_times = self.loader_times
            except Exception:
                pass
        try:
            vlds = trainer.val_dataloaders
            vlds = vlds if isinstance(vlds, (list, tuple)) else [vlds]
        except Exception:
            vlds = []
        for ld in vlds:
            try:
                ld.dataset.loader_times = self.loader_times
            except Exception:
                pass
        # start torch profiler if requested
        if self.verbosity >= 3:
            try:
                import torch.profiler

                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                )
                self.profiler.__enter__()
            except Exception:
                self.profiler = None

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.verbosity < 2:
            return
        self._cur_train_stats: list[dict] = []
        self._prev_train_end = None

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.verbosity < 2:
            return
        self._cur_val_stats: list[dict] = []
        self._prev_val_end = None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.verbosity < 2:
            return
        ep = trainer.current_epoch
        self._epoch_train_stats[ep] = self._summarize(self._cur_train_stats)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.verbosity < 2:
            return
        ep = trainer.current_epoch
        self._epoch_val_stats[ep] = self._summarize(self._cur_val_stats)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.verbosity == 0:
            return
        self._train_start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.verbosity == 0:
            return
        t_end = time.perf_counter()
        t_start = getattr(self, "_train_start", t_end)
        # data loading time since last batch end
        load_time = t_start - (self._prev_train_end or t_start)
        # total batch duration
        step_time = t_end - t_start
        # forward/map/obs from module timing
        times = getattr(pl_module, "_last_batch_times", {})
        # backward and optimizer timings if captured
        back_time = getattr(self, "_after_backward", None)
        opt_before = getattr(self, "_before_opt_step", None)
        opt_after = getattr(self, "_after_opt_step", None)
        backward_time = (
            (opt_before - back_time)
            if (back_time is not None and opt_before is not None)
            else 0.0
        )
        optimizer_time = (
            (opt_after - opt_before)
            if (opt_before is not None and opt_after is not None)
            else 0.0
        )
        stats = {
            "load_time": load_time,
            "forward_time": times.get("forward", 0.0),
            "map_time": times.get("map", 0.0),
            "obs_time": times.get("obs", 0.0),
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "step_time": step_time,
        }
        # collect activation magnitudes if enabled
        if self.log_activation_mag:
            last_mags = getattr(pl_module, "_last_activation_mags", {}) or {}
            for tag, mag in last_mags.items():
                try:
                    arr = mag.detach().cpu().numpy().ravel()
                except Exception:
                    continue
                self.train_activation_mags.setdefault(tag, []).append(arr)
        # record overall stats
        self.train_stats.append(stats)
        # record epoch-level if enabled
        if self.verbosity >= 2:
            buf = getattr(self, "_cur_train_stats", None)
            if buf is not None:
                buf.append(stats)
        self._prev_train_end = t_end
        # step profiler if enabled
        if self.verbosity >= 3 and self.profiler is not None:
            try:
                self.profiler.step()
            except Exception:
                pass

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.verbosity == 0:
            return
        self._val_start = time.perf_counter()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.verbosity == 0:
            return
        t_end = time.perf_counter()
        t_start = getattr(self, "_val_start", t_end)
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
        # collect activation magnitudes if enabled
        if self.log_activation_mag:
            last_mags = getattr(pl_module, "_last_activation_mags", {}) or {}
            for tag, mag in last_mags.items():
                try:
                    arr = mag.detach().cpu().numpy().ravel()
                except Exception:
                    continue
                self.val_activation_mags.setdefault(tag, []).append(arr)
        self.val_stats.append(stats)
        if self.verbosity >= 2:
            buf = getattr(self, "_cur_val_stats", None)
            if buf is not None:
                buf.append(stats)
        self._prev_val_end = t_end

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # skip if disabled
        if self.verbosity == 0:
            return

        # helper: compute statistics with mean/median/std/min/max
        def summarize(lst: list[dict]) -> dict:
            if not lst:
                return {}
            summary: dict = {}
            keys = lst[0].keys()
            arrs = {k: np.array([s.get(k, 0.0) for s in lst]) for k in keys}
            for k, arr in arrs.items():
                summary[k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "median": float(np.median(arr)),
                    "max": float(arr.max()),
                }
            return summary

        # build report
        report: dict = {}
        report["run_name"] = getattr(trainer.logger, "name", None)
        report["devices"] = getattr(trainer, "devices", None)
        report["precision"] = getattr(trainer, "precision", None)
        # environment snapshot
        report["environment"] = self.environment or {}
        # include hyperparams if available
        try:
            cfg = trainer.logger.experiment.config
            report["hparams"] = dict(cfg)
        except Exception:
            report["hparams"] = {}
        # loader summary
        report["loader_summary"] = summarize(self.loader_times)
        # run-level summaries
        report["train_summary"] = summarize(self.train_stats)
        report["val_summary"] = summarize(self.val_stats)
        # epoch-level summaries (v>=2)
        if self.verbosity >= 2:
            report["epochs"] = {
                "train": self._epoch_train_stats,
                "val": self._epoch_val_stats,
            }
        # activation magnitude summaries if enabled
        if self.log_activation_mag:
            act_summary: dict[str, Any] = {"train": {}, "val": {}}

            # helper to summarize 1D numpy arrays
            def _summ(arrs: list[np.ndarray]) -> dict[str, float]:
                allv = np.concatenate(arrs) if arrs else np.array([])
                if allv.size == 0:
                    return {}
                return {
                    "mean": float(allv.mean()),
                    "std": float(allv.std()),
                    "min": float(allv.min()),
                    "median": float(np.median(allv)),
                    "max": float(allv.max()),
                }

            for split, mags_dict in [
                ("train", self.train_activation_mags),
                ("val", self.val_activation_mags),
            ]:
                for tag, arrs in mags_dict.items():
                    parts = tag.split("_")
                    layer_name = "_".join(parts[1:-1])
                    irrep_str = parts[-1]
                    if layer_name not in act_summary[split]:
                        act_summary[split][layer_name] = {}
                    act_summary[split][layer_name][irrep_str] = _summ(arrs)
            report["activation_magnitudes"] = act_summary
        # operator profiling (v>=3)
        if self.verbosity >= 3 and self.profiler is not None:
            try:
                self.profiler.__exit__(None, None, None)
                # export trace
                trace_path = os.path.join(
                    "benchmarks",
                    f"{report['run_name']}_{self.environment.get('timestamp')}_trace.json",
                )
                self.profiler.export_chrome_trace(trace_path)
                # top ops
                ka = self.profiler.key_averages()
                top = sorted(ka, key=lambda x: x.cpu_time_total, reverse=True)[:10]
                report["top_ops"] = [
                    {
                        "key": op.key,
                        "cpu_time_total": op.cpu_time_total,
                        "cuda_time_total": getattr(op, "cuda_time_total", 0),
                    }
                    for op in top
                ]
            except Exception:
                pass
        # write YAML to file with timestamp
        os.makedirs("benchmarks", exist_ok=True)
        fname = os.path.join(
            "benchmarks",
            f"{report['run_name']}_{self.environment.get('timestamp')}_bench.yaml",
        )
        with open(fname, "w") as f:
            yaml.safe_dump(report, f, sort_keys=False)
        # log key metrics to logger (e.g. WandB)
        try:
            lm = getattr(trainer.logger, "log_metrics", None)
            if callable(lm):
                metrics: dict = {}
                ts = report["train_summary"]
                for k in [
                    "step_time",
                    "load_time",
                    "forward_time",
                    "backward_time",
                    "optimizer_time",
                ]:
                    if k in ts:
                        metrics[f"bench_train_{k}_mean"] = ts[k]["mean"]
                # include activation magnitude means
                if self.log_activation_mag and "activation_magnitudes" in report:
                    for tag, summ in report["activation_magnitudes"]["train"].items():
                        if "mean" in summ:
                            metrics[f"bench_train_{tag}_mean"] = summ["mean"]
                lm(metrics)
        except Exception:
            pass
