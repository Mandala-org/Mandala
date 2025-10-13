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

from data.factory import DatasetFactory
from net.common import Config
from net.benchmark import BenchmarkCallback
from data.residual_dataset import ResidualDataset
from net.residual_e3gnn import ResidualE3GNN


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
    parser = argparse.ArgumentParser(description="Train a residual E3GNN for Silicon.")
    default_config = Config()  # Create an instance to get actual default values

    # --- Path for the pretrained model ---
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to the checkpoint of the pretrained model.",
    )

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

    # --- Dynamically add Config fields for the RESIDUAL model ---
    config_fields = get_type_hints(Config)
    for name, field_type in config_fields.items():
        default_value = getattr(default_config, name)
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
    """Main training loop for the residual model."""
    args = setup_argparse()

    # --- CONFIGS ---
    # 1. Hardcode the config for the pretrained model as requested
    pretrained_cfg = Config(
        activation_gate="tanh",
        activation_scalar="leakyrelu",
        cutoff_gnn=7,
        cutoff_matrix=7,
        dropout=0,
        edge_update="concat",
        edge_update_linear="pre",
        edge_update_residual=True,
        head_depth=1,
        head_use_mlp_log_scale=False,
        hidden_base_dim=64,
        l_max_gnn=5,
        loss_coef_observables=6.563183096188056e-09,
        lr=0.002177634574679404,
        matrix_targets=["hamiltonian", "density", "overlap"],
        n_radial=128,
        neck_depth=1,
        nonlin_kind="normact",
        num_layers_gnn=2,
        num_layers_matrix=1,
        train_target="matrix",
    )

    # 2. Create the config for the new residual model from command-line args
    residual_cfg = Config()
    print("--- Populating Residual Config from args ---")
    for key, value in vars(args).items():
        if hasattr(residual_cfg, key):
            setattr(residual_cfg, key, value)

    # --- Initialize W&B ---
    wandb_project = os.getenv("WANDB_PROJECT", "mandala-residual-learning")
    wandb_logger = WandbLogger(
        project=wandb_project, config=dataclasses.asdict(residual_cfg)
    )

    # Post-process special types from argparse/wandb
    if isinstance(residual_cfg.dtype, str):
        residual_cfg.dtype = getattr(torch, residual_cfg.dtype)

    # --- Determine accelerator and devices ---
    if residual_cfg.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = residual_cfg.gpus
        device = torch.device("cuda:0")
        print(f"--- Using {devices} GPU(s) ---")
    else:
        accelerator = "cpu"
        devices = "auto"
        device = torch.device("cpu")
        print("--- Using CPU ---")

    # --- Data Loading ---
    print("--- Setting up original datasets ---")
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
        num_to_sample = min(len(snapshot_paths), args.n_snapshots_per_temp)
        selected_paths = random.sample(snapshot_paths, num_to_sample)
        for matrix_path in selected_paths:
            info_path = Path(matrix_path).parent / "info.dat"
            if info_path.exists():
                train_pairs.append((matrix_path, info_path))

    val_pairs = []
    val_path = Path(args.data_path) / f"{args.val_temp}K"
    val_snapshot_paths = sorted(glob.glob(str(val_path / "*/Si_DM")))
    num_val_to_sample = min(len(val_snapshot_paths), args.n_snapshots_per_temp)
    if args.val_n_snapshots is not None:
        num_val_to_sample = min(num_val_to_sample, args.val_n_snapshots)
    selected_val_paths = random.sample(val_snapshot_paths, num_val_to_sample)
    for matrix_path in selected_val_paths:
        info_path = Path(matrix_path).parent / "info.dat"
        if info_path.exists():
            val_pairs.append((matrix_path, info_path))

    print(
        f"Found {len(train_pairs)} training and {len(val_pairs)} validation snapshots."
    )

    # Use residual_cfg for data factory, as cutoffs etc. should match
    fac = DatasetFactory(residual_cfg)
    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    for m, i in val_pairs:
        fac.add_snapshot(m, i, purpose="val")
    original_train_ds, original_val_ds, mapper = fac.create()

    # --- Create Residual Datasets ---
    print("--- Creating residual datasets ---")
    train_ds = ResidualDataset(
        original_train_ds, args.pretrained_model_path, pretrained_cfg, mapper, device
    )
    val_ds = (
        ResidualDataset(
            original_val_ds, args.pretrained_model_path, pretrained_cfg, mapper, device
        )
        if original_val_ds
        else None
    )

    def _dl(ds, shuffle=False):
        return DataLoader(
            ds or [],
            batch_size=1,
            shuffle=shuffle,
            num_workers=residual_cfg.num_workers,
            pin_memory=residual_cfg.gpus == 0,
            collate_fn=lambda b: b[0],
        )

    train_loader = _dl(train_ds, shuffle=True)
    val_loader = _dl(val_ds, shuffle=False)
    print(
        f"Created dataloaders: train batches={len(train_loader)}, val batches={len(val_loader)}"
    )

    # --- Model and Trainer Setup ---
    print("--- Setting up model and trainer ---")
    model = ResidualE3GNN(
        pretrained_model_path=args.pretrained_model_path,
        pretrained_cfg=pretrained_cfg,
        residual_cfg=residual_cfg,
        mapper=mapper,
    )

    callbacks = [
        BenchmarkCallback(
            verbosity=residual_cfg.bench_verbosity,
            log_activation_mag=residual_cfg.log_activation_mag,
        )
    ]

    trainer = pl.Trainer(
        max_epochs=residual_cfg.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
        log_every_n_steps=residual_cfg.log_every_n_steps,
        gradient_clip_val=residual_cfg.grad_clip_val,
        gradient_clip_algorithm="value",
        accumulate_grad_batches=residual_cfg.accumulate_grad_batches,
        precision=args.precision,
    )

    # --- Start Training ---
    print("--- Starting training ---")
    torch.set_float32_matmul_precision("high")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
