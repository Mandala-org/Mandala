from __future__ import annotations

import argparse
import ast
import dataclasses
import os
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from net.artifacts import (
    ArtifactCheckpointCallback,
    RevertOnSpikeCallback,
)  # noqa: E402
from net.benchmark import BenchmarkCallback  # noqa: E402
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from net.run_logging import (  # noqa: E402
    log_config,
    log_cutoff_application,
    log_graph,
    log_mapper_info,
    log_orbital_config,
    log_snapshot_info,
)
from scripts.dataset import (  # noqa: E402
    build_datasets_from_yaml,
    build_silicon_datasets,
    build_siox_datasets,
    build_zncusnses_small_datasets,
    build_zncusnses_datasets,
)
from utils.run_name import resolve_run_name  # noqa: E402


def run_training(
    run_args: argparse.Namespace | SimpleNamespace | dict[str, Any],
    *,
    parsed_yaml: dict[str, Any] | None = None,
    extra_callbacks: list[Any] | None = None,
    objective_metric: str | None = None,
) -> dict[str, float]:
    args = _as_namespace(run_args)
    print("=== Mandala training run starting ===")
    cfg = _populate_config_from_args(args)
    requested_run_name = getattr(args, "run_name", None)
    cfg.save_dir = str(getattr(args, "checkpoint_dir", cfg.save_dir))
    resume_checkpoint = _resolve_resume_checkpoint(
        getattr(args, "resume_from_checkpoint", None),
        getattr(args, "resume_mode", "latest"),
    )
    accelerator, devices = _resolve_accelerator_and_device(cfg)
    logger = _build_logger(args, cfg, requested_run_name)
    run_name = resolve_run_name(requested_run_name, logger)
    setattr(args, "run_name", run_name)
    cfg.run_name = run_name
    _print_run_summary(args, cfg, run_name, resume_checkpoint, accelerator, devices)

    dataset_bundle = _build_dataset_bundle(args, cfg, parsed_yaml)
    train_ds, val_ds, mapper = dataset_bundle
    run_dir = Path(getattr(args, "checkpoint_dir", cfg.save_dir)) / run_name
    if run_dir.exists() and resume_checkpoint is None:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a unique run name or resume explicitly."
        )
    _log_dataset_and_model_context(cfg, run_dir, train_ds, mapper)

    train_ds, val_ds = _maybe_move_datasets_to_device(train_ds, val_ds, cfg)
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, accelerator, cfg)

    model = E3GNN(mapper=mapper, cfg=cfg)
    callbacks = _build_callbacks(args, cfg, run_dir, extra_callbacks)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.grad_clip_val,
        gradient_clip_algorithm="value",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        precision=getattr(args, "precision", "32-true"),
    )

    print("--- Starting training ---")
    torch.set_float32_matmul_precision("high")
    interrupted = False
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
        )
    except KeyboardInterrupt:
        interrupted = True
        setattr(args, "interrupted", True)
        print("--- Training interrupted by Ctrl+C; finishing shutdown ---", flush=True)
    metrics = _extract_metrics(trainer)
    if objective_metric is not None and objective_metric not in metrics:
        if interrupted:
            print(
                f"--- Objective metric {objective_metric!r} was not available after interrupt; continuing ---"
            )
        else:
            raise RuntimeError(
                f"Objective metric {objective_metric!r} was not found in trainer.callback_metrics."
            )
    elif objective_metric is not None:
        print(
            f"Objective metric {objective_metric} = {metrics.get(objective_metric, 'MISSING')}"
        )
    print("--- Training finished ---")
    if interrupted:
        print(
            "--- Training stopped after interrupt; finalization completed normally ---"
        )
    return metrics


def _as_namespace(
    run_args: argparse.Namespace | SimpleNamespace | dict[str, Any],
) -> argparse.Namespace:
    if isinstance(run_args, argparse.Namespace):
        return run_args
    if isinstance(run_args, SimpleNamespace):
        return argparse.Namespace(**vars(run_args))
    if isinstance(run_args, dict):
        return argparse.Namespace(**run_args)
    raise TypeError(f"Unsupported run_args type: {type(run_args)!r}")


def _populate_config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    print("--- Populating Config from args ---")
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if isinstance(cfg.matrix_targets, str):
        cfg.matrix_targets = _coerce_list_value(cfg.matrix_targets)
    if isinstance(cfg.radial_layers, str):
        cfg.radial_layers = _coerce_list_value(cfg.radial_layers)
    if isinstance(cfg.dtype, str):
        cfg.dtype = _resolve_torch_dtype(cfg.dtype)
    return cfg


def _coerce_list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") or raw.startswith("("):
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = None
            else:
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
        if "," in raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [raw]
    return [value]


def _resolve_torch_dtype(value: str) -> torch.dtype:
    candidate = value.strip()
    if candidate.startswith("torch."):
        candidate = candidate.split(".", 1)[1]
    dtype = getattr(torch, candidate, None)
    if not isinstance(dtype, torch.dtype):
        raise AttributeError(f"module 'torch' has no dtype named {value!r}")
    return dtype


def _resolve_resume_checkpoint(path: str | None, resume_mode: str) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        ckpt_name = {
            "latest": "latest_checkpoint.pt",
            "best": "best_model.pt",
            "final": "final_model.pt",
        }[resume_mode]
        ckpt = candidate / ckpt_name
        if ckpt.exists():
            print(f"--- Resuming from checkpoint: {ckpt.resolve()} ---")
            return ckpt.resolve()
        raise FileNotFoundError(
            f"No checkpoint matching mode={resume_mode!r} found in {candidate}"
        )
    raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")


def _resolve_accelerator_and_device(cfg: Config) -> tuple[str, int | str]:
    if cfg.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices: int | str = cfg.gpus
        cfg.device = torch.device("cuda:0")
        print(f"--- Using {devices} GPU(s) ---")
        return accelerator, devices
    accelerator = "cpu"
    devices = "auto"
    cfg.device = torch.device("cpu")
    if cfg.gpus > 0:
        print("--- Warning: --gpus was > 0 but CUDA is not available. Using CPU. ---")
    else:
        print("--- Using CPU ---")
    return accelerator, devices


def _build_logger(
    args: argparse.Namespace, cfg: Config, run_name: str | None
) -> WandbLogger | None:
    wandb_mode = getattr(args, "wandb_mode", None)
    if wandb_mode is None:
        wandb_mode = os.getenv("WANDB_MODE", "online")
    if wandb_mode == "disabled":
        print("--- WandB disabled ---")
        return None
    os.environ["WANDB_MODE"] = wandb_mode
    wandb_project = (
        getattr(args, "wandb_project", None)
        or cfg.wandb_project
        or os.getenv("WANDB_PROJECT")
        or "mandala-silicon-main-study-port"
    )
    if wandb_project is None:
        print("--- No WandB project set; logger disabled ---")
        return None
    logger_config = dataclasses.asdict(cfg)
    if run_name is None:
        logger_config.pop("run_name", None)
    print(
        f"--- WandB logger: mode={wandb_mode}, project={wandb_project}, run_name={run_name or '<wandb-assigned>'} ---"
    )
    return WandbLogger(
        project=wandb_project,
        name=run_name,
        config=logger_config,
        save_dir=str(Path(getattr(args, "checkpoint_dir", cfg.save_dir))),
    )


def _build_dataset_bundle(
    args: argparse.Namespace,
    cfg: Config,
    parsed_yaml: dict[str, Any] | None,
):
    convention = getattr(args, "convention", "e3nn")
    dataset_kind = getattr(args, "dataset_kind", "silicon")
    print(
        f"--- Preparing datasets: dataset_kind={dataset_kind}, convention={convention} ---"
    )
    if parsed_yaml is not None:
        print("--- Dataset configuration sourced from parsed YAML ---")
        return build_datasets_from_yaml(
            parsed_yaml, cfg, overrides=vars(args), convention=convention
        )

    if dataset_kind == "silicon":
        return build_silicon_datasets(
            data_path=getattr(args, "data_path"),
            cfg=cfg,
            min_temp=getattr(args, "min_temp", 300),
            max_temp=getattr(args, "max_temp", 3000),
            temp_step=getattr(args, "temp_step", 300),
            n_snapshots_per_temp=getattr(args, "n_snapshots_per_temp", 50),
            val_temp=getattr(args, "val_temp", 1500),
            val_n_snapshots=getattr(args, "val_n_snapshots", None),
            num_train=getattr(args, "num_train", None),
            num_val=getattr(args, "num_val", None),
            seed=cfg.seed,
            convention=convention,
        )
    if dataset_kind == "siox":
        return build_siox_datasets(
            data_path=getattr(args, "data_path"),
            cfg=cfg,
            num_train=getattr(args, "num_train", None),
            num_val=getattr(args, "num_val", None),
            val_fraction=getattr(args, "val_fraction", 0.2),
            seed=cfg.seed,
            convention=convention,
        )
    if dataset_kind == "ZnCuSnSeS_small":
        return build_zncusnses_small_datasets(
            data_path=getattr(args, "data_path"),
            cfg=cfg,
            num_train=getattr(args, "num_train", None),
            num_val=getattr(args, "num_val", None),
            val_fraction=getattr(args, "val_fraction", 0.2),
            seed=cfg.seed,
            convention=convention,
        )
    if dataset_kind == "ZnCuSnSeS":
        scales = getattr(args, "scales", None)
        if scales is None:
            scales = [1, 2, 3]
        return build_zncusnses_datasets(
            data_path=getattr(args, "data_path"),
            cfg=cfg,
            scales=[int(x) for x in scales],
            num_train_per_scale=int(getattr(args, "num_train_per_scale", 40)),
            num_val_per_scale=int(getattr(args, "num_val_per_scale", 10)),
            seed=cfg.seed,
            convention=convention,
        )
    raise ValueError(f"Unsupported dataset_kind: {dataset_kind!r}")


def _build_dataloaders(
    train_ds: Any, val_ds: Any, accelerator: str, cfg: Config
) -> tuple[DataLoader, DataLoader]:
    dataset_on_device = _dataset_is_on_device(train_ds) or _dataset_is_on_device(val_ds)
    if dataset_on_device:
        print(
            "--- Dataset already moved to its target device; disabling DataLoader workers and pin_memory ---"
        )
        total_workers = 0
    else:
        total_workers = _resolve_total_workers(cfg.num_workers)
    train_workers, val_workers = _split_worker_budget(
        total_workers, len(train_ds or []), len(val_ds or [])
    )
    print(
        f"--- DataLoader workers: total={total_workers}, train={train_workers}, val={val_workers} ---"
    )

    def _dl(ds, shuffle: bool = False, num_workers: int = 0):
        use_pin_memory = accelerator == "gpu" and not dataset_on_device
        use_persistent_workers = num_workers > 0
        return DataLoader(
            ds or [],
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
            collate_fn=lambda b: b[0],
        )

    train_loader = _dl(train_ds, shuffle=True, num_workers=train_workers)
    val_loader = _dl(val_ds, shuffle=False, num_workers=val_workers)
    print(
        f"Created dataloaders: train batches={len(train_loader)}, val batches={len(val_loader)}"
    )
    return train_loader, val_loader


def _maybe_move_datasets_to_device(
    train_ds: Any, val_ds: Any, cfg: Config
) -> tuple[Any, Any]:
    dataset_device = getattr(cfg, "dataset_device", None)
    if dataset_device in (None, "", "cpu"):
        return train_ds, val_ds
    device = torch.device(dataset_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"dataset_device={dataset_device!r} was requested but CUDA is not available"
        )
    print(f"--- Moving processed datasets to {device} ---")
    if hasattr(train_ds, "to"):
        train_ds = train_ds.to(device)
    if val_ds is not None and hasattr(val_ds, "to"):
        val_ds = val_ds.to(device)
    return train_ds, val_ds


def _dataset_is_on_device(ds: Any) -> bool:
    device = getattr(ds, "device", None)
    if device is None:
        return False
    try:
        return torch.device(device).type != "cpu"
    except Exception:
        return False


def _resolve_total_workers(num_workers: int | None) -> int:
    if num_workers is not None:
        return max(0, int(num_workers))
    try:
        affinity = os.sched_getaffinity(0)
        available_cores = len(affinity)
    except (AttributeError, OSError):
        available_cores = os.cpu_count() or 1
    return max(0, int(available_cores) - 1)


def _split_worker_budget(
    total_workers: int, train_count: int, val_count: int
) -> tuple[int, int]:
    if total_workers <= 0:
        return 0, 0
    total_count = train_count + val_count
    if total_count <= 0:
        return 0, 0
    train_workers = math.floor(total_workers * train_count / total_count)
    train_workers = max(0, min(total_workers, train_workers))
    val_workers = total_workers - train_workers
    return train_workers, val_workers


def _log_dataset_and_model_context(
    cfg: Config,
    run_dir: Path,
    train_ds: Any,
    mapper: Any,
) -> None:
    frames_dir = run_dir / "frames"
    if not (cfg.log_model or cfg.log_data):
        return
    log_config(
        {
            **dataclasses.asdict(cfg),
            "hidden_irreps": cfg.hidden_irreps,
            "device": str(cfg.device),
        },
        run_dir,
        frames_dir,
    )
    if len(train_ds) == 0:
        return
    x0, y0 = train_ds[0]
    if cfg.log_data:
        log_snapshot_info(x0, y0)
        if cfg.apply_cutoff_to_targets and "target_edges_before_cutoff" in x0:
            log_cutoff_application(
                int(x0["target_edges_before_cutoff"]),
                int(x0["target_edges_after_cutoff"]),
                float(cfg.cutoff_radius),
            )
        log_graph(x0)
    if cfg.log_model:
        log_orbital_config(mapper.orbital_cfg)
        log_mapper_info(mapper)


def _build_callbacks(
    args: argparse.Namespace,
    cfg: Config,
    run_dir: Path,
    extra_callbacks: list[Any] | None,
) -> list[Any]:
    callbacks: list[Any] = []
    if cfg.benchmark:
        callbacks.append(
            BenchmarkCallback(
                verbosity=cfg.bench_verbosity, log_activation_mag=cfg.log_activation_mag
            )
        )
    callbacks.append(_build_progress_bar())
    if getattr(args, "log_artifacts", True):
        callbacks.append(
            ArtifactCheckpointCallback(
                output_dir=run_dir,
                generate_video=getattr(args, "generate_video", True),
                log_per_irrep_images=cfg.log_per_irrep_images,
            )
        )
    if cfg.revert_on_spike:
        callbacks.append(
            RevertOnSpikeCallback(
                output_dir=run_dir,
                monitor=cfg.revert_monitor or cfg.lr_scheduler_target,
                patience=cfg.revert_decay_patience,
                decay_rate=cfg.revert_decay_rate,
                spike_factor=cfg.revert_spike_factor,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    print(f"--- Built callbacks: {[type(cb).__name__ for cb in callbacks]} ---")
    return callbacks


def _build_progress_bar() -> TQDMProgressBar:
    class _FilteredProgressBar(TQDMProgressBar):
        _allowed_keys = {"train/loss_total", "val/loss_total"}

        def get_metrics(self, trainer, pl_module):
            metrics = super().get_metrics(trainer, pl_module)
            return {
                key: value
                for key, value in metrics.items()
                if key in self._allowed_keys or key in {"epoch", "step", "v_num"}
            }

    return _FilteredProgressBar()


def _extract_metrics(trainer: pl.Trainer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                metrics[str(key)] = float(value.detach().cpu().item())
        else:
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def _print_run_summary(
    args: argparse.Namespace,
    cfg: Config,
    run_name: str,
    resume_checkpoint: Path | None,
    accelerator: str,
    devices: int | str,
) -> None:
    checkpoint_dir = Path(getattr(args, "checkpoint_dir", cfg.save_dir))
    print("--- Run summary ---")
    print(f"run_name={run_name}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"resume_checkpoint={resume_checkpoint}")
    print(f"dataset_kind={getattr(args, 'dataset_kind', 'silicon')}")
    print(f"data_path={getattr(args, 'data_path', None)}")
    print(f"max_epochs={cfg.max_epochs}")
    print(f"matrix_targets={cfg.matrix_targets}")
    print(f"accelerator={accelerator}")
    print(f"devices={devices}")
    print(f"precompute_edge_features={cfg.precompute_edge_features}")
