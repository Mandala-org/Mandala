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

import torch
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


def setup_argparse():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train E3GNN for Silicon.")

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

    # --- Dynamically add Config fields as arguments ---
    config_fields = get_type_hints(Config)
    for name, field_type in config_fields.items():
        default_value = getattr(Config, name, dataclasses.MISSING)
        if isinstance(default_value, dataclasses.Field):
            default_value = default_value.default

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
            inner_type = str
            try:
                inner_type = typing.get_args(field_type)[0]
            except (IndexError, TypeError):
                pass
            parser.add_argument(
                f"--{name}", type=inner_type, nargs="+", default=default_value
            )
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

    # --- Initialize W&B ---
    # Use environment variables for W&B project if available, otherwise use default
    wandb_project = os.getenv("WANDB_PROJECT", "mandala-silicon-sweep")
    wandb_logger = WandbLogger(project=wandb_project, config=vars(args))

    # Create Config object and update it from wandb
    cfg = Config()
    for key, value in wandb_logger.experiment.config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # Post-process special types from argparse/wandb
    if isinstance(cfg.dtype, str):
        cfg.dtype = getattr(torch, cfg.dtype)

    # Set device
    cfg.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and cfg.gpus else "cpu"
    )

    # --- Data Loading ---
    print("--- Setting up datasets ---")
    all_temps = range(args.min_temp, args.max_temp + 1, args.temp_step)
    train_temps = [t for t in all_temps if t != args.val_temp]

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
    # Use all available snapshots for validation
    for matrix_path in val_snapshot_paths:
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

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
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
        devices=[cfg.device.index] if cfg.device.type == "cuda" else "auto",
        accelerator="gpu" if cfg.device.type == "cuda" else "cpu",
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.grad_clip_val,
    )

    # --- Start Training ---
    print("--- Starting training ---")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
