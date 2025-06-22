# src/scripts/train.py
# ---------------------------------------------------------------------
#  Train an E3GNN on OpenMX snapshots with a single shared BlockIrrepMapper.
#
#  $ python -m scripts.train --config configs/hydrogen.yaml
#  $ python -m scripts.train matrix_paths.txt info_paths.txt  # quick ad-hoc run
#
#  Logging:   WandB by default   (WANDB_API_KEY must be in the env)
#  Sweeps:    --tune wandb      → WandB Sweep agent
#             --tune ray        → Ray-Tune HPO
# ---------------------------------------------------------------------

from __future__ import annotations
import sys
import yaml
import json
import argparse
from typing import Any, Dict
import datetime as dt
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from net.benchmark import BenchmarkCallback
from data.factory import DatasetFactory
from net.common import HyperParams
from net.e3gnn import E3GNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
def _parse_cfg(path: Path | None) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file into a dict."""
    if path is None:
        return {}
    with open(path) as fh:
        if path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(fh) or {}
        elif path.suffix == ".json":
            return json.load(fh)
        else:
            raise ValueError("config file must be .yaml/.yml or .json")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train.py", description="E3GNN trainer")

    p.add_argument("--config", type=Path, help="YAML/JSON with hyper-params")
    p.add_argument(
        "--tune",
        choices=[None, "wandb", "ray"],
        default=None,
        help="run inside a WandB sweep or Ray-Tune session",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="cpu",
        help="'cpu'  or a comma-separated list of CUDA device ids, e.g. '0,1'",
    )
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--accum", type=int, default=1, help="grad-accum steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bench-verbosity",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Benchmark verbosity: 0=off,1=run summary,2=+epoch,3=+profiler",
    )
    p.add_argument(
        "--log-activation-mag",
        action="store_true",
        help="Enable logging of activation/feature magnitudes by irrep to WandB and benchmark report",
    )
    p.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity of script output: 0=none, 1=basic events, 2=+detailed timings",
    )
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# Main entry
# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = _cli()

    # helper for timestamped printing
    def vprint(msg: str) -> None:
        if args.verbosity >= 1:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} {msg}")

    def vprint_detail(msg: str) -> None:
        if args.verbosity >= 2:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} {msg}")

    vprint("Starting training script")
    torch.manual_seed(args.seed)
    vprint(f"Loaded config from {args.config}")

    cfg = _parse_cfg(args.config)

    # ------------------------------- 1. Dataset factory ----------------
    # Dataset factory with explicit type casting of numeric hyperparams
    fact = DatasetFactory(
        cutoff_gnn=float(cfg.get("cutoff_gnn", 5.0)),
        cutoff_matrix=float(cfg.get("cutoff_matrix", 7.5)),
        l_max_sh=int(cfg.get("l_max_sh", 3)),
        n_radial=int(cfg.get("n_radial", 64)),
        device="cpu",
    )

    # From config: explicit snapshot listings
    for entry in cfg.get("snapshots", []):
        fact.add_snapshot(
            Path(entry["matrix"]),
            Path(entry["info"]),
            entry.get("purpose", "train"),
        )
    # Auto-generate snapshots by temperature and index if configured
    base = cfg.get("dataset_base_path")
    temps_train = cfg.get("train_temps")
    temps_val = cfg.get("val_temps")
    n_snaps = cfg.get("n_snapshots_per_temp")
    if base and n_snaps and (temps_train or temps_val):
        base_p = Path(base)
        # Training splits
        for t in temps_train or []:
            for i in range(n_snaps):
                fact.add_snapshot(
                    base_p / f"{t}K" / str(i) / "Si_DM",
                    base_p / f"{t}K" / str(i) / "info.dat",
                    purpose="train",
                )
        # Validation splits
        for t in temps_val or []:
            for i in range(n_snaps):
                fact.add_snapshot(
                    base_p / f"{t}K" / str(i) / "Si_DM",
                    base_p / f"{t}K" / str(i) / "info.dat",
                    purpose="val",
                )

    # Optional extra CLI lists
    if not cfg.get("snapshots") and len(sys.argv) >= 3:
        # quick mode: python … matrix.txt info.txt
        mat_path, info_path = map(Path, sys.argv[-2:])
        fact.add_snapshot(mat_path, info_path, "train")

    ds_train, ds_val, mapper = fact.create()
    vprint(f"Created datasets: train={len(ds_train)}, val={len(ds_val)}")

    # ------------------------------- 2. DataLoaders --------------------
    def _dl(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=1,  # ALWAYS 1
            shuffle=shuffle,
            num_workers=int(cfg.get("num_workers", 7)),
            pin_memory=True,
            collate_fn=lambda b: b[0],  # <- avoid default_collate on custom objects
        )

    dl_train = _dl(ds_train, shuffle=True)
    dl_val = _dl(ds_val, shuffle=False)
    vprint(
        f"Created dataloaders: train batches={len(dl_train)}, val batches={len(dl_val)}"
    )

    # ------------------------------- 3. Hyper-parameters ---------------
    hp = HyperParams(**cfg.get("model", {}))
    if args.gpus.lower() == "cpu":
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "gpu"
        devices = [int(i) for i in args.gpus.split(",") if i.strip()]

    # ------------------------------- 4. Model --------------------------
    # Model instantiation with explicit casting for learning rate
    model = E3GNN(
        mapper=mapper,
        edge_types=ds_train.edge_types,
        hp=hp,
        lr=float(cfg.get("lr", 3e-4)),
        device="cuda" if accelerator == "gpu" else "cpu",
    )
    vprint(
        f"Built model: l_max={hp.l_max}, hidden_base_dim={hp.hidden_base_dim}, "
        f"layers_gnn={hp.num_layers_gnn}, layers_matrix={hp.num_layers_matrix}"
    )

    # ------------------------------- 5. Logging ------------------------
    run_name = cfg.get("run_name") or f"e3gnn_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if cfg.get("wandb_project", None) is not None:
        logger = WandbLogger(
            project=cfg.get("wandb_project"),
            name=run_name,
            log_model=True,
            save_dir=str(PROJECT_ROOT / "wandb"),
        )
        logger.experiment.config.update(cfg, allow_val_change=True)
        vprint(f"Initialized WandB logger with run name '{run_name}'")
    else:
        logger = None
        vprint("Not using a logger")

    # ------------------------------- 6. Trainer ------------------------
    # build callbacks
    bench_cb = None
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            dirpath=PROJECT_ROOT / "checkpoints" / run_name,
            filename="{epoch:03d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    # add benchmark callback if enabled
    if args.bench_verbosity > 0:
        bench_cb = BenchmarkCallback(
            verbosity=args.bench_verbosity,
            log_activation_mag=args.log_activation_mag,
        )
        callbacks.append(bench_cb)
    vprint("Configured callbacks")

    # Trainer with explicit casting for epochs and precision
    trainer = pl.Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(args.max_epochs or cfg.get("max_epochs", 100)),
        accumulate_grad_batches=int(args.accum),
        precision=int(cfg.get("precision", 16)),
        callbacks=callbacks,
        deterministic=True,
        log_every_n_steps=int(cfg.get("log_every_n_steps", 4)),
    )

    # ------------------------------- 7. HPO integration ----------------
    if args.tune == "wandb":
        import wandb

        wandb.finish()  # lightning starts its own run inside Trainer
    elif args.tune == "ray":
        from ray import tune

        tune.report(loss=0.0)  # Lightning will handle metrics; this is placeholder

    # ------------------------------- 8. Train --------------------------
    vprint("Starting training")
    trainer.fit(model, dl_train, dl_val)
    vprint("Training complete")
    # detailed timing summary
    if args.verbosity >= 2 and bench_cb is not None:
        # loader times summary
        lt = np.array(bench_cb.loader_times) if bench_cb.loader_times else np.array([])
        if lt.size:
            vprint_detail(
                f"Loader times (s): mean={lt.mean():.4f}, median={np.median(lt):.4f}, "
                f"std={lt.std():.4f}, min={lt.min():.4f}, max={lt.max():.4f}"
            )
        # train batch timings
        train_sum = bench_cb._summarize(bench_cb.train_stats)
        for k, s in train_sum.items():
            vprint_detail(
                f"Train {k} (s): mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}"
            )
        # val batch timings
        val_sum = bench_cb._summarize(bench_cb.val_stats)
        for k, s in val_sum.items():
            vprint_detail(
                f"Val {k} (s): mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}"
            )


if __name__ == "__main__":
    main()
