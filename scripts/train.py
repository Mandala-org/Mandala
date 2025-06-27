# src/scripts/train.py
# ---------------------------------------------------------------------
#  Train an E3GNN on OpenMX snapshots with a single shared BlockIrrepMapper.
#
#  Usage examples:
#    python scripts/train.py --config-name debug_cpu
#    python scripts/train.py --config-name medium_gpu --verbosity 2 --bench_verbosity 0
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


@hydra.main(config_path="../conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Extract grouped config settings
    verbosity = cfg.logging.verbosity
    bench_verbosity = cfg.logging.bench_verbosity
    log_activation_mag = cfg.logging.log_activation_mag
    gpus_cfg = cfg.training.gpus
    tune_cfg = cfg.training.tune

    def vprint(msg: str) -> None:
        if verbosity >= 1:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} {msg}")

    def vprint_detail(msg: str) -> None:
        if verbosity >= 2:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} {msg}")

    vprint("Starting training script")
    torch.manual_seed(cfg.training.seed)

    # ------------------------------------------------------------------
    # 1. Dataset construction
    # ------------------------------------------------------------------
    fact = DatasetFactory(
        cutoff_gnn=cfg.data.cutoff_gnn,
        cutoff_matrix=cfg.data.cutoff_matrix,
        l_max_sh=cfg.model.l_max,
        n_radial=cfg.data.n_radial,
        device="cpu",
    )
    for entry in cfg.data.snapshots:
        fact.add_snapshot(Path(entry.matrix), Path(entry.info), entry.purpose)
    ds_train, ds_val, mapper = fact.create()
    vprint(f"Created datasets: train={len(ds_train)}, val={len(ds_val or [])}")

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    def _dl(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
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
    hp = HyperParams(**cfg.model)
    # Hardware setup: interpret training.gpus as "cpu" or a GPU count
    if isinstance(gpus_cfg, str) and gpus_cfg.lower() == "cpu":
        accelerator = "cpu"
        devices = 1
    else:
        # parse GPU count
        try:
            count = int(gpus_cfg) if isinstance(gpus_cfg, str) else gpus_cfg
        except Exception:
            raise ValueError(f"Invalid training.gpus value: {gpus_cfg}")
        if count <= 0:
            accelerator = "cpu"
            devices = 1
        else:
            accelerator = "gpu"
            devices = list(range(count))
    model = E3GNN(
        mapper=mapper,
        edge_types=ds_train.edge_types,
        hp=hp,
        lr=cfg.get("lr", 3e-4),
        device="cuda" if accelerator == "gpu" else "cpu",
    )
    vprint(
        f"Built model: l_max={hp.l_max}, hidden_base_dim={hp.hidden_base_dim}, "
        f"layers_gnn={hp.num_layers_gnn}, layers_matrix={hp.num_layers_matrix}"
    )

    # ------------------------------------------------------------------
    # 4. Logger setup
    # ------------------------------------------------------------------
    run_name = cfg.logging.run_name or f"e3gnn_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if cfg.logging.wandb_project:
        logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=run_name,
            log_model=cfg.logging.log_model,
            save_dir=str(PROJECT_ROOT / cfg.logging.save_dir),
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
            dirpath=PROJECT_ROOT / "checkpoints" / run_name,
            filename="{epoch:03d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if bench_verbosity > 0:
        bench_cb = BenchmarkCallback(
            verbosity=bench_verbosity,
            log_activation_mag=log_activation_mag,
        )
        callbacks.append(bench_cb)
    vprint("Configured callbacks")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        deterministic=True,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    # ------------------------------------------------------------------
    # 7. HPO integration
    # ------------------------------------------------------------------
    if tune_cfg == "wandb":
        import wandb

        wandb.finish()
    elif tune_cfg == "ray":
        from ray import tune

        tune.report(loss=0.0)

    # ------------------------------------------------------------------
    # 8. Training
    # ------------------------------------------------------------------
    vprint("Starting training")
    trainer.fit(model, dl_train, dl_val)
    vprint("Training complete")

    # 9. Detailed benchmark summary
    if verbosity >= 2 and bench_cb is not None:
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
