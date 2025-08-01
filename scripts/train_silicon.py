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

        arg_type = field_type
        origin = typing.get_origin(field_type)

        # Handle Union types like str | None, which are not callable
        if origin is Union or origin is typing.Union:
            union_args = typing.get_args(field_type)
            non_none_args = [
                t for t in union_args if t is not type(None) and t is not NoneType
            ]
            if len(non_none_args) == 1:
                arg_type = non_none_args[0]
            else:  # Fallback for more complex unions
                arg_type = str

        # Handle other special types that argparse can't call directly
        if arg_type is torch.dtype:
            arg_type = str  # Accept a string, convert to torch.dtype later

        if field_type is bool:
            parser.add_argument(f"--{name}", action="store_true", default=default_value)
        else:
            # Handle Sequence types
            if "Sequence" in str(field_type) or "list" in str(field_type):
                parser.add_argument(
                    f"--{name}", type=int, nargs="+", default=default_value
                )
            else:
                parser.add_argument(f"--{name}", type=arg_type, default=default_value)

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
