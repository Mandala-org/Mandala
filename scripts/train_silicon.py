import argparse
import dataclasses
import glob
import os
import random
import sys
from pathlib import Path
from typing import get_type_hints, Union
from types import NoneType
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
        raise argparse.ArgumentTypeError(f"Failed to parse '{value}' as a list.")


def setup_argparse():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train E3GNN for Silicon.")
    default_config = Config()  # Create an instance to get actual default values

    # --- Dataset Arguments ---
    parser.add_argument(
        "--data_path",
        type=str,
        default="/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A",
    )
    parser.add_argument("--min_temp", type=int, default=300)
    parser.add_argument("--max_temp", type=int, default=3000)
    parser.add_argument("--temp_step", type=int, default=300)
    parser.add_argument(
        "--n_snapshots_per_temp",
        type=int,
        default=50,
        help="Number of snapshots to use for training per temperature.",
    )
    parser.add_argument(
        "--val_temp",
        type=int,
        default=1500,
        help="Temperature to use for the validation set.",
    )
    parser.add_argument(
        "--val_n_snapshots",
        type=int,
        default=None,
        help="Number of snapshots to use for validation. If None, uses the same as training.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="PyTorch Lightning precision setting (e.g., '32-true', '16-mixed').",
    )

    # --- Dynamically add Config fields as arguments ---
    config_fields = get_type_hints(Config)
    for name, field_type in config_fields.items():
        # Get default value from the instance, not the class
        default_value = getattr(default_config, name)

        # Use the new boolean handling for bool types
        if field_type is bool:
            parser.add_argument(f"--{name}", type=str_to_bool, default=default_value)
            continue

        arg_type_callable = None
        origin = typing.get_origin(field_type)

        if origin is Union or origin is typing.Union:
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
            parser.add_argument(f"--{name}", type=str_to_list, default=default_value)
        else:
            if not callable(arg_type_callable):
                arg_type_callable = str
            parser.add_argument(
                f"--{name}", type=arg_type_callable, default=default_value
            )

    return parser.parse_args()


def main():
    """Main training loop."""
    args = setup_argparse()

    # Create Config object and update it from the parsed arguments
    cfg = Config()
    print("--- Populating Config from args ---")
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # --- Initialize W&B ---
    # Use environment variables for W&B project if available, otherwise use default
    wandb_project = os.getenv("WANDB_PROJECT", "mandala-silicon-sweep")
    # Pass the final, correct config to W&B for logging
    wandb_logger = WandbLogger(project=wandb_project, config=dataclasses.asdict(cfg))

    # Post-process special types from argparse/wandb
    if isinstance(cfg.dtype, str):
        cfg.dtype = getattr(torch, cfg.dtype)

    # --- Determine accelerator and devices ---
    if cfg.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = cfg.gpus
        cfg.device = torch.device("cuda:0")
        print(f"--- Using {devices} GPU(s) ---")
    else:
        accelerator = "cpu"
        devices = "auto"
        cfg.device = torch.device("cpu")
        if cfg.gpus > 0:
            print(
                "--- Warning: --gpus was > 0 but CUDA is not available. Using CPU. ---"
            )
        else:
            print("--- Using CPU ---")

    # --- Data Loading ---
    print("--- Setting up datasets ---")
    all_temps = range(args.min_temp, args.max_temp + 1, args.temp_step)
    train_temps = [
        t
        for t in all_temps
        if t != args.val_temp or args.min_temp == args.max_temp == args.val_temp
    ]

    train_pairs = []
    for temp in train_temps:
        temp_path = Path(args.data_path) / f"{temp}K"
        snapshot_paths = sorted(glob.glob(str(temp_path / "*/Si_DM")))
        # Ensure we don't request more samples than available
        num_to_sample = min(len(snapshot_paths), args.n_snapshots_per_temp)
        selected_paths = random.sample(snapshot_paths, num_to_sample)
        for matrix_path in selected_paths:
            info_path = Path(matrix_path).parent / "info.dat"
            if info_path.exists():
                train_pairs.append((matrix_path, info_path))

    val_pairs = []
    val_path = Path(args.data_path) / f"{args.val_temp}K"
    val_snapshot_paths = sorted(glob.glob(str(val_path / "*/Si_DM")))
    # Limit validation snapshots as well
    num_val_to_sample = min(len(val_snapshot_paths), args.n_snapshots_per_temp)
    if args.val_n_snapshots is not None:
        num_val_to_sample = min(num_val_to_sample, args.val_n_snapshots)
    selected_val_paths = random.sample(val_snapshot_paths, num_val_to_sample)
    for matrix_path in selected_val_paths:
        info_path = Path(matrix_path).parent / "info.dat"
        if info_path.exists():
            val_pairs.append((matrix_path, info_path))

    print(
        f"Found {len(train_pairs)} training snapshots and {len(val_pairs)} validation snapshots."
    )

    fac = DatasetFactory(cfg)

    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    for m, i in val_pairs:
        fac.add_snapshot(m, i, purpose="val")

    train_ds, val_ds, mapper = fac.create()

    def _dl(ds, shuffle=False):
        return DataLoader(
            ds or [],
            batch_size=1,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.gpus == 0,  # Pin memory only if not using GPU
            collate_fn=lambda b: b[0],
        )

    train_loader = _dl(train_ds, shuffle=True)
    val_loader = _dl(val_ds, shuffle=False)
    print(
        f"Created dataloaders: train batches={len(train_loader)}, val batches={len(val_loader)}"
    )

    # --- Model and Trainer Setup ---
    print("--- Setting up model and trainer ---")
    model = E3GNN(mapper=mapper, cfg=cfg)

    callbacks = [
        BenchmarkCallback(
            verbosity=cfg.bench_verbosity, log_activation_mag=cfg.log_activation_mag
        )
    ]

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
        terminate_on_nan=True,
    )

    # --- Start Training ---
    print("--- Starting training ---")
    torch.set_float32_matmul_precision("high")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
