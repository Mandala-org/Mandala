from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def _load_module():
    path = Path("scripts/train.py").resolve()
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_populate_config_from_args_normalizes_string_list_fields():
    mod = _load_module()
    args = mod.argparse.Namespace(
        matrix_targets="density",
        radial_layers="[128, 64]",
    )

    cfg = mod._populate_config_from_args(args)

    assert cfg.matrix_targets == ["density"]
    assert cfg.radial_layers == [128, 64]


def test_build_progress_bar_filters_metrics(monkeypatch):
    mod = _load_module()

    class DummyProgressBar(mod.TQDMProgressBar):
        def get_metrics(self, trainer, pl_module):
            return super().get_metrics(trainer, pl_module)

    monkeypatch.setattr(mod, "TQDMProgressBar", DummyProgressBar)

    class FakeProgressBar(DummyProgressBar):
        def get_metrics(self, trainer, pl_module):
            return {
                "epoch": 1,
                "step": 2,
                "v_num": 3,
                "train/loss_total": 0.1,
                "val/loss_total": 0.2,
                "train/density_irrep_1e_l2_elem": 99.0,
            }

    monkeypatch.setattr(mod, "TQDMProgressBar", FakeProgressBar)
    progress_bar = mod._build_progress_bar()
    metrics = progress_bar.get_metrics(object(), object())

    assert metrics == {
        "epoch": 1,
        "step": 2,
        "v_num": 3,
        "train/loss_total": 0.1,
        "val/loss_total": 0.2,
    }


def test_build_dataloaders_splits_workers_by_sample_ratio(monkeypatch):
    mod = _load_module()

    class DummyDataset(list):
        pass

    train_ds = DummyDataset([1, 2, 3])
    val_ds = DummyDataset([4])
    captured = []

    class DummyLoader:
        def __init__(self, ds, *, num_workers=0, **kwargs):
            captured.append(num_workers)
            self.num_workers = num_workers
            self._len = len(ds)
            self.persistent_workers = kwargs.get("persistent_workers", False)
            self.pin_memory = kwargs.get("pin_memory", False)

        def __len__(self):
            return self._len

    monkeypatch.setattr(mod, "DataLoader", DummyLoader)
    monkeypatch.setattr(mod, "_resolve_total_workers", lambda num_workers: 8)

    train_loader, val_loader = mod._build_dataloaders(
        train_ds, val_ds, accelerator="cpu", cfg=mod.Config(num_workers=None)
    )

    assert captured == [6, 2]
    assert train_loader.num_workers == 6
    assert val_loader.num_workers == 2
    assert train_loader.persistent_workers is True
    assert val_loader.persistent_workers is True


def test_build_dataloaders_disables_workers_for_gpu_dataset(monkeypatch):
    mod = _load_module()

    class DummyDataset(list):
        def __init__(self, *args):
            super().__init__(*args)
            self.device = "cuda"

    train_ds = DummyDataset([1, 2, 3])
    val_ds = DummyDataset([4])
    captured = []

    class DummyLoader:
        def __init__(self, ds, *, num_workers=0, **kwargs):
            captured.append((num_workers, kwargs.get("pin_memory", False)))
            self.num_workers = num_workers
            self._len = len(ds)
            self.persistent_workers = kwargs.get("persistent_workers", False)
            self.pin_memory = kwargs.get("pin_memory", False)

        def __len__(self):
            return self._len

    monkeypatch.setattr(mod, "DataLoader", DummyLoader)
    monkeypatch.setattr(mod, "_resolve_total_workers", lambda num_workers: 8)

    train_loader, val_loader = mod._build_dataloaders(
        train_ds, val_ds, accelerator="gpu", cfg=mod.Config(num_workers=None)
    )

    assert captured == [(0, False), (0, False)]
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False


def test_build_callbacks_adds_revert_on_spike(monkeypatch, tmp_path):
    mod = _load_module()

    monkeypatch.setattr(mod, "_build_progress_bar", lambda: "progress")
    monkeypatch.setattr(
        mod,
        "ArtifactCheckpointCallback",
        lambda **kwargs: ("artifact", kwargs),
    )
    monkeypatch.setattr(mod, "RunBookkeepingCallback", lambda: "bookkeeping")
    monkeypatch.setattr(
        mod,
        "RevertOnSpikeCallback",
        lambda **kwargs: ("revert", kwargs),
    )

    args = mod.argparse.Namespace(log_artifacts=True, generate_video=False)
    cfg = mod.Config(benchmark=False, revert_on_spike=True, revert_monitor=None)

    callbacks = mod._build_callbacks(args, cfg, tmp_path, extra_callbacks=None)

    assert callbacks[0] == "progress"
    assert callbacks[1] == "bookkeeping"
    assert callbacks[2][0] == "artifact"
    assert callbacks[3][0] == "revert"
    assert callbacks[3][1]["monitor"] == cfg.lr_scheduler_target


def test_build_callbacks_adds_wall_clock_budget(monkeypatch, tmp_path):
    mod = _load_module()

    monkeypatch.setattr(mod, "_build_progress_bar", lambda: "progress")
    monkeypatch.setattr(
        mod,
        "ArtifactCheckpointCallback",
        lambda **kwargs: ("artifact", kwargs),
    )
    monkeypatch.setattr(mod, "RunBookkeepingCallback", lambda: "bookkeeping")
    monkeypatch.setattr(
        mod,
        "WallClockBudgetCallback",
        lambda **kwargs: ("wall_clock", kwargs),
    )

    args = mod.argparse.Namespace(log_artifacts=True, generate_video=False)
    cfg = mod.Config(
        benchmark=False,
        revert_on_spike=False,
        max_wall_clock_seconds=43200.0,
    )

    callbacks = mod._build_callbacks(args, cfg, tmp_path, extra_callbacks=None)

    assert callbacks[0] == "progress"
    assert callbacks[1] == "bookkeeping"
    assert callbacks[2][0] == "artifact"
    assert callbacks[3] == ("wall_clock", {"budget_seconds": 43200.0})


def test_run_training_uses_wandb_run_name_for_run_dir(monkeypatch, tmp_path):
    mod = _load_module()

    captured = {}

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.experiment = type(
                "Experiment",
                (),
                {"name": "vibran-sweep-14", "id": "abc123"},
            )()

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.callback_metrics = {"val/loss_total": torch.tensor(1.0)}
            self.optimizers = [type("Opt", (), {"param_groups": [{"lr": 1e-3}]})()]

        def fit(self, *args, **kwargs):
            captured["fit_kwargs"] = kwargs

    def fake_build_dataset_bundle(args, cfg, parsed_yaml):
        return [1], [2], object()

    def fake_build_dataloaders(train_ds, val_ds, accelerator, cfg):
        return [1], [2]

    def fake_build_callbacks(args, cfg, run_dir, extra_callbacks):
        captured["run_dir"] = run_dir
        captured["cfg_run_name"] = cfg.run_name
        return []

    monkeypatch.setattr(mod, "WandbLogger", DummyLogger)
    monkeypatch.setattr(mod.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(mod, "_build_dataset_bundle", fake_build_dataset_bundle)
    monkeypatch.setattr(mod, "_maybe_move_datasets_to_device", lambda a, b, c: (a, b))
    monkeypatch.setattr(mod, "_build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(mod, "_build_callbacks", fake_build_callbacks)
    monkeypatch.setattr(mod, "E3GNN", lambda mapper, cfg: object())
    monkeypatch.setattr(mod, "_log_dataset_and_model_context", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_resolve_resume_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "_resolve_accelerator_and_device", lambda cfg: ("cpu", "auto")
    )
    monkeypatch.setattr(
        mod,
        "_build_logger",
        lambda args, cfg, run_name: DummyLogger(),
    )

    args = mod.argparse.Namespace(
        checkpoint_dir=str(tmp_path),
        run_name=None,
        wandb_mode="offline",
        wandb_project="test-project",
        resume_from_checkpoint=None,
        resume_mode="latest",
        precision="32-true",
    )

    mod.run_training(args)

    assert captured["run_dir"] == tmp_path / "vibran-sweep-14"
    assert captured["cfg_run_name"] == "vibran-sweep-14"
    assert args.run_name == "vibran-sweep-14"


def test_populate_config_from_args_accepts_torch_prefixed_dtype():
    mod = _load_module()
    args = mod.argparse.Namespace(
        dtype="torch.float32",
        matrix_targets=["hamiltonian"],
        radial_layers=[64, 64],
    )

    cfg = mod._populate_config_from_args(args)

    assert cfg.dtype == torch.float32


def test_split_manifest_is_disjoint_and_stable(tmp_path):
    mod = _load_module()

    class Dataset:
        def __init__(self, pairs):
            self.snapshot_paths = pairs

    train = Dataset([(tmp_path / "h0", tmp_path / "i0")])
    val = Dataset([(tmp_path / "h1", tmp_path / "i1")])
    test = Dataset([(tmp_path / "h2", tmp_path / "i2")])

    first = mod._write_split_manifest(
        tmp_path / "run-a",
        train_ds=train,
        val_ds=val,
        test_ds=test,
        data_split_seed=42,
    )
    second = mod._write_split_manifest(
        tmp_path / "run-b",
        train_ds=train,
        val_ds=val,
        test_ds=test,
        data_split_seed=42,
    )

    assert first["split_hash_sha256"] == second["split_hash_sha256"]
    assert first["counts"] == {"train": 1, "val": 1, "test": 1}
    on_disk = json.loads((tmp_path / "run-a" / "split_manifest.json").read_text())
    assert on_disk["split_hash_sha256"] == first["split_hash_sha256"]


def test_split_manifest_rejects_leakage(tmp_path):
    mod = _load_module()

    class Dataset:
        snapshot_paths = [(tmp_path / "h", tmp_path / "i")]

    with pytest.raises(ValueError, match="leakage"):
        mod._write_split_manifest(
            tmp_path / "run",
            train_ds=Dataset(),
            val_ds=Dataset(),
            test_ds=None,
            data_split_seed=42,
        )


def test_paper_run_gate_rejects_resume_and_missing_test():
    mod = _load_module()
    cfg = mod.Config(
        paper_run=True,
        experiment_id="paper-v1",
        ablation_name="envelope",
        ablation_setting="on",
        max_wall_clock_seconds=10,
        allow_incomplete_dataset=False,
        checkpoint_monitor="val/hamiltonian_mae",
    )
    args = mod.argparse.Namespace(evaluate_test_after_fit=True, log_artifacts=True)

    with pytest.raises(ValueError, match="Paper-run engineering gate failed"):
        mod._validate_paper_run_gate(
            args,
            cfg,
            train_ds=[1],
            val_ds=[2],
            test_ds=None,
            resume_checkpoint=Path("checkpoint.pt"),
            compatibility_mode=False,
        )


def test_paper_run_gate_accepts_controlled_from_scratch_run():
    mod = _load_module()
    cfg = mod.Config(
        paper_run=True,
        experiment_id="paper-v1",
        ablation_name="envelope",
        ablation_setting="on",
        max_wall_clock_seconds=10,
        allow_incomplete_dataset=False,
        checkpoint_monitor="val/hamiltonian_mae",
    )
    args = mod.argparse.Namespace(evaluate_test_after_fit=True, log_artifacts=True)

    mod._validate_paper_run_gate(
        args,
        cfg,
        train_ds=[1],
        val_ds=[2],
        test_ds=[3],
        resume_checkpoint=None,
        compatibility_mode=False,
    )
