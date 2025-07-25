# src/scripts/train.py
# ---------------------------------------------------------------------
#  Train an E3GNN on OpenMX snapshots with a single shared BlockIrrepMapper.
#
#  Usage examples:
#    python scripts/train.py --config-name debug_cpu
#    python scripts/train.py --config-name medium_gpu config.verbosity 2 config.verbosity 0
#
#  Logging:   WandB by default   (WANDB_API_KEY must be in the env)
#  Sweeps:    tune: 'wandb' → WandB Sweep Agent
#             tune: 'ray'   → Ray Tune HPO
# ---------------------------------------------------------------------

from __future__ import annotations
import datetime as dt
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from net.benchmark import BenchmarkCallback
from data.factory import DatasetFactory
from net.common import Config, get_torch_dtype
from net.e3gnn import E3GNN


@hydra.main(config_path="../conf", version_base="1.1")
def main(omega_cfg: DictConfig) -> None:
    # Extract grouped config settings
    cfg = Config(**omega_cfg.config)
    cfg.dtype = get_torch_dtype(cfg.dtype)

    def vprint(msg: str) -> None:
        if cfg.verbosity >= 1:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} {msg}")

    def vprint_detail(msg: str) -> None:
        if cfg.verbosity >= 2:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} {msg}")

    vprint("Starting training script")
    torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------
    # 1. Dataset construction
    # ------------------------------------------------------------------
    fact = DatasetFactory(cfg)
    for entry in omega_cfg.dataset.snapshots:
        matrix_path = to_absolute_path(entry.matrix)
        info_path = to_absolute_path(entry.info)
        fact.add_snapshot(Path(matrix_path), Path(info_path), entry.purpose)
    ds_train, ds_val, mapper = fact.create()
    vprint(f"Created datasets: train={len(ds_train)}, val={len(ds_val or [])}")

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    def _dl(ds, shuffle=False):
        return DataLoader(
            ds or [],
            batch_size=1,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=lambda b: b[0],
        )

    dl_train = _dl(ds_train, shuffle=True)
    dl_val = _dl(ds_val, shuffle=False)
    vprint(
        f"Created dataloaders: train batches={len(dl_train)}, val batches={len(dl_val)}"
    )

    # ------------------------------------------------------------------
    # 3. Model instantiation
    # ------------------------------------------------------------------
    # Hardware setup: interpret training.gpus as "cpu" or a GPU count
    if isinstance(cfg.gpus, str) and cfg.gpus.lower() == "cpu":
        accelerator = "cpu"
        devices = 1
    else:
        # parse GPU count
        try:
            count = int(cfg.gpus) if isinstance(cfg.gpus, str) else cfg.gpus
        except Exception:
            raise ValueError(f"Invalid training.gpus value: {cfg.gpus}")
        if count <= 0:
            accelerator = "cpu"
            devices = 1
        else:
            accelerator = "gpu"
            devices = list(range(count))
    model = E3GNN(
        mapper=mapper,
        cfg=cfg,
    )
    vprint(
        f"Built model: l_max={cfg.l_max}, hidden_base_dim={cfg.hidden_base_dim}, "
        f"layers_gnn={cfg.num_layers_gnn}, layers_matrix={cfg.num_layers_matrix}"
    )

    # ------------------------------------------------------------------
    # 4. Logger setup
    # ------------------------------------------------------------------
    run_name = cfg.run_name or f"e3gnn_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if cfg.wandb_project:
        logger = WandbLogger(
            project=cfg.wandb_project,
            name=run_name,
            log_model=cfg.log_model,
            save_dir=to_absolute_path(cfg.save_dir),
        )
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
        )
        vprint(f"Initialized WandB logger '{run_name}'")
    else:
        logger = None
        vprint("Not using a logger")

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    bench_cb = None
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            dirpath=to_absolute_path(f"checkpoints/{run_name}"),
            filename="{epoch:03d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.verbosity > 0:
        bench_cb = BenchmarkCallback(
            verbosity=cfg.verbosity,
            log_activation_mag=cfg.log_activation_mag,
        )
        callbacks.append(bench_cb)
    vprint("Configured callbacks")

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    if cfg.smoke_test:
        vprint("Smoke test mode enabled: running one batch of train and val.")
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            fast_dev_run=True,
            deterministic=True,
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=cfg.max_epochs,
            # precision=cfg.precision,
            callbacks=callbacks,
            deterministic=True,
            log_every_n_steps=cfg.log_every_n_steps,
            gradient_clip_val=cfg.grad_clip_val,
        )

    # ------------------------------------------------------------------
    # 7. HPO integration
    # ------------------------------------------------------------------
    if cfg.tune == "wandb":
        import wandb

        wandb.finish()
    elif cfg.tune == "ray":
        from ray import tune

        tune.report(loss=0.0)

    # ------------------------------------------------------------------
    # 8. Training
    # ------------------------------------------------------------------
    vprint("Starting training")
    trainer.fit(model, dl_train, dl_val)
    vprint("Training complete")

    # 9. Detailed benchmark summary
    if cfg.verbosity >= 2 and bench_cb is not None:
        lt = np.array(bench_cb.loader_times) if bench_cb.loader_times else np.array([])
        if lt.size:
            vprint_detail(
                f"Loader times (s): mean={lt.mean():.4f}, median={np.median(lt):.4f}, "
                f"std={lt.std():.4f}, min={lt.min():.4f}, max={lt.max():.4f}"
            )
        train_sum = bench_cb._summarize(bench_cb.train_stats)
        for k, s in train_sum.items():
            vprint_detail(
                f"Train {k} (s): mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}"
            )
        val_sum = bench_cb._summarize(bench_cb.val_stats)
        for k, s in val_sum.items():
            vprint_detail(
                f"Val {k} (s): mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}"
            )


if __name__ == "__main__":
    main()
