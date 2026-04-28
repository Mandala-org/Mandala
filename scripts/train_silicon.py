import argparse
import dataclasses
import glob
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, get_type_hints, Union
from types import NoneType, UnionType
import typing
import ast

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from data.factory import DatasetFactory  # noqa: E402
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from net.benchmark import BenchmarkCallback  # noqa: E402
from net.artifacts import (
    ArtifactCheckpointCallback,
    RevertOnSpikeCallback,
)  # noqa: E402
from net.silicon_study_logging import (  # noqa: E402
    log_config,
    log_cutoff_application,
    log_graph,
    log_mapper_info,
    log_orbital_config,
    log_snapshot_info,
)
from utils.run_name import resolve_run_name  # noqa: E402


def str_to_bool(value):
    """Helper function to handle boolean command-line arguments."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str_to_list(value):
    """Helper function to parse string representation of a list."""
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if "," in raw:
                return [part.strip() for part in raw.split(",") if part.strip()]
            return [raw]
        raise argparse.ArgumentTypeError(f"Failed to parse '{value}' as a list.")


def _arg_names(name: str) -> tuple[str, ...]:
    primary = f"--{name}"
    alias = f"--{name.replace('_', '-')}"
    if alias == primary:
        return (primary,)
    return (primary, alias)


def discover_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for matrix_path in sorted(root.rglob("Si_DM")):
        if not matrix_path.is_file():
            continue
        info_path = matrix_path.parent / "info.dat"
        if not info_path.exists():
            continue
        pairs.append((matrix_path.resolve(), info_path.resolve()))
    return pairs


def setup_argparse():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train E3GNN for Silicon.")
    default_config = Config()  # Create an instance to get actual default values

    # --- Dataset Arguments ---
    parser.add_argument(
        *_arg_names("data_path"),
        type=str,
        default="/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A",
    )
    parser.add_argument(*_arg_names("min_temp"), type=int, default=300)
    parser.add_argument(*_arg_names("max_temp"), type=int, default=3000)
    parser.add_argument(*_arg_names("temp_step"), type=int, default=300)
    parser.add_argument(
        *_arg_names("n_snapshots_per_temp"),
        type=int,
        default=50,
        help="Number of snapshots to use for training per temperature.",
    )
    parser.add_argument(
        *_arg_names("val_temp"),
        type=int,
        default=1500,
        help="Temperature to use for the validation set.",
    )
    parser.add_argument(
        *_arg_names("val_n_snapshots"),
        type=int,
        default=None,
        help="Number of snapshots to use for validation. If None, uses the same as training.",
    )
    parser.add_argument(
        *_arg_names("num_train"),
        type=int,
        default=None,
        help="If set together with --num-val, use a global train/val split over all snapshots.",
    )
    parser.add_argument(
        *_arg_names("num_val"),
        type=int,
        default=None,
        help="Validation count for global split mode.",
    )
    parser.add_argument(
        *_arg_names("precision"),
        type=str,
        default="32-true",
        help="PyTorch Lightning precision setting (e.g., '32-true', '16-mixed').",
    )
    parser.add_argument(
        *_arg_names("checkpoint_dir"),
        type=str,
        default="checkpoints/silicon",
        help="Directory where run checkpoints and artifacts are stored.",
    )
    parser.add_argument(
        *_arg_names("resume_from_checkpoint"),
        type=str,
        default=None,
        help="Resume from a checkpoint file or run directory (latest/best/final).",
    )
    parser.add_argument(
        *_arg_names("resume_mode"),
        type=str,
        default="latest",
        choices=["latest", "best", "final"],
        help="When resuming from a directory, which checkpoint to load.",
    )
    parser.add_argument(
        *_arg_names("generate_video"),
        type=str_to_bool,
        default=True,
        help="Generate and upload training-progress videos.",
    )
    parser.add_argument(
        *_arg_names("log_artifacts"),
        type=str_to_bool,
        default=True,
        help="Enable DOS, distance-curve, and per-irrep artifact generation.",
    )

    # --- Dynamically add Config fields as arguments ---
    config_fields = get_type_hints(Config)
    for name, field_type in config_fields.items():
        # Get default value from the instance, not the class
        default_value = getattr(default_config, name)
        if name == "run_name":
            default_value = None

        # Use the new boolean handling for bool types
        if field_type is bool:
            parser.add_argument(
                *_arg_names(name), type=str_to_bool, default=default_value
            )
            continue

        arg_type_callable = None
        origin = typing.get_origin(field_type)

        if origin is Union or origin is typing.Union or origin is UnionType:
            union_args = typing.get_args(field_type)
            non_none_args = [
                t for t in union_args if t is not type(None) and t is not NoneType
            ]
            if len(non_none_args) == 1:
                arg_type_callable = non_none_args[0]
            else:
                arg_type_callable = str
        else:
            arg_type_callable = field_type

        if arg_type_callable is torch.device or arg_type_callable is torch.dtype:
            arg_type_callable = str

        is_sequence = False
        try:
            if (
                "Sequence" in str(field_type)
                or "list" in str(field_type)
                or (origin and issubclass(origin, typing.Sequence))
            ):
                is_sequence = True
        except TypeError:
            pass

        if is_sequence:
            parser.add_argument(
                *_arg_names(name), type=str_to_list, default=default_value
            )
        else:
            if not callable(arg_type_callable):
                arg_type_callable = str
            parser.add_argument(
                *_arg_names(name), type=arg_type_callable, default=default_value
            )

    return parser.parse_args()


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
            return ckpt.resolve()
        raise FileNotFoundError(
            f"No checkpoint matching mode={resume_mode!r} found in {candidate}"
        )
    raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")


def _populate_config_from_args(args: argparse.Namespace) -> Config:
    # Create Config object and update it from the parsed arguments
    cfg = Config()
    print("--- Populating Config from args ---")
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if isinstance(cfg.dtype, str):
        cfg.dtype = getattr(torch, cfg.dtype)
    return cfg


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


def _build_wandb_logger(
    args: argparse.Namespace, cfg: Config, run_name: str | None
) -> WandbLogger:
    wandb_project = (
        args.wandb_project
        or cfg.wandb_project
        or os.getenv("WANDB_PROJECT")
        or "mandala-silicon-main-study-port"
    )
    logger_config = dataclasses.asdict(cfg)
    if run_name is None:
        logger_config.pop("run_name", None)
    return WandbLogger(
        project=wandb_project,
        name=run_name,
        config=logger_config,
        save_dir=str(Path(args.checkpoint_dir)),
    )


def _build_train_val_pairs(
    args: argparse.Namespace, cfg: Config
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    print("--- Setting up datasets ---")
    if args.num_train is not None or args.num_val is not None:
        if args.num_train is None or args.num_val is None:
            raise ValueError(
                "Global split mode requires both --num_train and --num_val."
            )
        if args.num_train <= 0:
            raise ValueError("--num_train must be > 0")
        if args.num_val < 0:
            raise ValueError("--num_val must be >= 0")
        all_pairs = discover_snapshot_pairs(Path(args.data_path))
        random.Random(cfg.seed).shuffle(all_pairs)
        if len(all_pairs) < args.num_train + args.num_val:
            raise ValueError(
                f"Requested train+val={args.num_train + args.num_val} but found only {len(all_pairs)} snapshots under {args.data_path}"
            )
        train_pairs = all_pairs[: args.num_train]
        val_pairs = all_pairs[args.num_train : args.num_train + args.num_val]
        return train_pairs, val_pairs

    all_temps = range(args.min_temp, args.max_temp + 1, args.temp_step)
    train_temps = [
        t
        for t in all_temps
        if t != args.val_temp or args.min_temp == args.max_temp == args.val_temp
    ]

    train_pairs: list[tuple[Path, Path]] = []
    for temp in train_temps:
        temp_path = Path(args.data_path) / f"{temp}K"
        snapshot_paths = sorted(glob.glob(str(temp_path / "*/Si_DM")))
        num_to_sample = min(len(snapshot_paths), args.n_snapshots_per_temp)
        selected_paths = random.sample(snapshot_paths, num_to_sample)
        for matrix_path in selected_paths:
            info_path = Path(matrix_path).parent / "info.dat"
            if info_path.exists():
                train_pairs.append((Path(matrix_path), info_path))

    val_pairs: list[tuple[Path, Path]] = []
    val_path = Path(args.data_path) / f"{args.val_temp}K"
    val_snapshot_paths = sorted(glob.glob(str(val_path / "*/Si_DM")))
    num_val_to_sample = min(len(val_snapshot_paths), args.n_snapshots_per_temp)
    if args.val_n_snapshots is not None:
        num_val_to_sample = min(num_val_to_sample, args.val_n_snapshots)
    selected_val_paths = random.sample(val_snapshot_paths, num_val_to_sample)
    for matrix_path in selected_val_paths:
        info_path = Path(matrix_path).parent / "info.dat"
        if info_path.exists():
            val_pairs.append((Path(matrix_path), info_path))
    return train_pairs, val_pairs


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


def run_single_training(
    args: argparse.Namespace,
) -> None:
    cfg = _populate_config_from_args(args)

    requested_run_name = getattr(args, "run_name", None)
    cfg.save_dir = args.checkpoint_dir
    resume_checkpoint = _resolve_resume_checkpoint(
        args.resume_from_checkpoint, args.resume_mode
    )
    accelerator, devices = _resolve_accelerator_and_device(cfg)
    wandb_logger = _build_wandb_logger(args, cfg, requested_run_name)
    run_name = resolve_run_name(requested_run_name, wandb_logger)
    setattr(args, "run_name", run_name)
    cfg.run_name = run_name
    train_pairs, val_pairs = _build_train_val_pairs(args, cfg)

    print(
        f"Found {len(train_pairs)} training snapshots and {len(val_pairs)} validation snapshots."
    )

    cfg_ds = dataclasses.replace(cfg)
    if not cfg.apply_cutoff_to_targets:
        cfg_ds.cutoff_radius = None
    fac = DatasetFactory(cfg_ds)

    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    for m, i in val_pairs:
        fac.add_snapshot(m, i, purpose="val")

    train_ds, val_ds, mapper = fac.create()
    run_dir = Path(args.checkpoint_dir) / run_name
    if run_dir.exists() and resume_checkpoint is None:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a unique run name or resume explicitly."
        )
    _log_dataset_and_model_context(cfg, run_dir, train_ds, mapper)
    train_ds, val_ds = _maybe_move_datasets_to_device(train_ds, val_ds, cfg)
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, accelerator, cfg)

    # --- Model and Trainer Setup ---
    print("--- Setting up model and trainer ---")
    model = E3GNN(mapper=mapper, cfg=cfg)

    callbacks = []
    if cfg.benchmark:
        callbacks.append(
            BenchmarkCallback(
                verbosity=cfg.bench_verbosity, log_activation_mag=cfg.log_activation_mag
            )
        )
    if args.log_artifacts:
        callbacks.append(
            ArtifactCheckpointCallback(
                output_dir=Path(args.checkpoint_dir) / run_name,
                generate_video=args.generate_video,
                log_per_irrep_images=args.log_per_irrep_images,
            )
        )
    if cfg.revert_on_spike:
        callbacks.append(
            RevertOnSpikeCallback(
                output_dir=Path(args.checkpoint_dir) / run_name,
                monitor=cfg.revert_monitor or cfg.lr_scheduler_target,
                patience=cfg.revert_decay_patience,
                decay_rate=cfg.revert_decay_rate,
                spike_factor=cfg.revert_spike_factor,
            )
        )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.grad_clip_val,
        gradient_clip_algorithm="value",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        precision=args.precision,
    )

    # --- Start Training ---
    print("--- Starting training ---")
    torch.set_float32_matmul_precision("high")
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
        )
    except KeyboardInterrupt:
        setattr(args, "interrupted", True)
        print("--- Training interrupted by Ctrl+C; finishing shutdown ---", flush=True)


def main():
    """Main training loop."""
    args = setup_argparse()
    run_single_training(args)


if __name__ == "__main__":
    main()
