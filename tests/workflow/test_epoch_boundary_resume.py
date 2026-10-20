"""Scientific continuation contract: splitting/replaying epochs preserves learning."""

import random
import dataclasses
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from net.restart import EpochBoundaryCheckpoint, EpochShuffleSampler


class StochasticModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        factor = 1 + 0.1 * random.random() + 0.1 * np.random.random()
        return factor * (self.net(x) - y).square().mean()

    def validation_step(self, batch, batch_idx):
        # Force scheduler reductions to detect pre-scheduler checkpoint bugs.
        self.log("val/loss", 1.0, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=0
                ),
                "monitor": "val/loss",
            },
        }


class StopOrCrash(pl.Callback):
    def __init__(self, crash):
        self.crash = crash

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.crash and trainer.current_epoch == 2 and batch_idx == 1:
            raise RuntimeError("simulated interruption")

    def on_validation_epoch_end(self, trainer, module):
        if not self.crash and trainer.current_epoch == 1:
            trainer.should_stop = True


def fit(path, *, checkpoint=None, stop=None, device="cpu"):
    pl.seed_everything(731, workers=True)
    ds = TensorDataset(
        torch.arange(18, dtype=torch.float32).reshape(9, 2) / 18,
        torch.arange(9, dtype=torch.float32).reshape(9, 1) / 9,
    )
    train = DataLoader(
        ds,
        batch_size=1,
        sampler=EpochShuffleSampler(ds, 731),
        generator=torch.Generator().manual_seed(731),
    )
    val = DataLoader(ds, batch_size=1, generator=torch.Generator().manual_seed(731))
    callbacks = ([stop] if stop else []) + [EpochBoundaryCheckpoint(path, "val/loss")]
    trainer = pl.Trainer(
        max_epochs=4,
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    model = StochasticModel()
    trainer.fit(model, train, val, ckpt_path=checkpoint, weights_only=False)
    return torch.load(path / "latest_checkpoint.pt", weights_only=False)


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("crash", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path, crash, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA continuation check requires a GPU.")
    full = fit(tmp_path / "full", device=device)
    split_path = tmp_path / "split"
    if crash:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            fit(split_path, stop=StopOrCrash(True), device=device)
    else:
        fit(split_path, stop=StopOrCrash(False), device=device)
    # The selection file may have advanced beyond the safe boundary before
    # interruption. Recovery must reinstate its own matching best checkpoint.
    torch.save({"abandoned_selection": True}, split_path / "best_model.pt")
    resumed = fit(
        split_path, checkpoint=split_path / "latest_checkpoint.pt", device=device
    )
    for field in [
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "global_step",
        "epoch",
        "callbacks",
    ]:
        assert_nested_equal(full[field], resumed[field])
    assert full["lr_schedulers"][0]["last_epoch"] == 4
    best = torch.load(split_path / "best_model.pt", weights_only=False)
    assert best["epoch"] == 0  # restart must not reset the all-time best ranking


def test_reject_best_checkpoint_for_continuation(tmp_path):
    fit(tmp_path)
    with pytest.raises(ValueError, match="epoch-boundary"):
        fit(tmp_path, checkpoint=tmp_path / "best_model.pt")


def test_e3gnn_training_entrypoint_continues_exactly(
    tmp_path, monkeypatch, small_angular_snapshot_e3nn
):
    from data.factory import DatasetFactory
    from data.snapshot import Snapshot
    from net.common import Config
    from scripts import train

    cfg = Config(
        cutoff_radius=8,
        l_max=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        dropout=0.1,
        verbosity=0,
        device="cpu",
        gpus=0,
        num_workers=0,
        max_epochs=4,
        epoch_boundary_resume=True,
        checkpoint_monitor="val/loss_total",
        lr_scheduler_target="val/loss_total",
        revert_monitor="val/loss_total",
        lr_scheduler_patience=0,
        log_per_irrep_metrics=False,
        log_hamiltonian_irrep_contrib_metrics=False,
        log_hamiltonian_pair_contrib_metrics=False,
    )
    monkeypatch.setattr(
        Snapshot, "from_openmx", lambda *a, **k: small_angular_snapshot_e3nn
    )
    monkeypatch.setattr(
        DatasetFactory,
        "_load_info",
        lambda self, *a: SimpleNamespace(orbital_set={"H": "1s1p"}),
    )
    factory = DatasetFactory(cfg)
    factory.add_snapshot("synthetic.matrix", "synthetic.out")
    ds, _, mapper = factory.create()
    val_ds = copy.copy(ds)
    val_ds.snapshot_paths = [(Path("validation.matrix"), Path("validation.out"))]
    monkeypatch.setattr(train, "_build_dataset_bundle", lambda *a: (ds, val_ds, mapper))
    monkeypatch.setattr(train, "_log_dataset_and_model_context", lambda *a: None)

    def run(name, checkpoint=None, stop=None):
        values = dataclasses.asdict(cfg) | dict(
            checkpoint_dir=str(tmp_path),
            run_name=name,
            wandb_mode="disabled",
            resume_from_checkpoint=checkpoint,
            resume_mode="latest",
            evaluate_test_after_fit=False,
        )
        values["_explicit_args"] = set(values)
        train.run_training(values, extra_callbacks=[stop] if stop else [])
        return torch.load(tmp_path / name / "latest_checkpoint.pt", weights_only=False)

    full = run("full")
    run("split", stop=StopOrCrash(False))
    resumed = run("split", checkpoint=str(tmp_path / "split/latest_checkpoint.pt"))
    for field in [
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "global_step",
        "epoch",
    ]:
        assert_nested_equal(full[field], resumed[field])
    for prefix in ["RevertOnSpikeCallback", "EpochBoundaryCheckpoint"]:
        assert_nested_equal(
            [
                value
                for key, value in full["callbacks"].items()
                if key.startswith(prefix)
            ],
            [
                value
                for key, value in resumed["callbacks"].items()
                if key.startswith(prefix)
            ],
        )
