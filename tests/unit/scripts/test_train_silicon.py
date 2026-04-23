from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_train_silicon_module():
    path = Path("scripts/train_silicon.py").resolve()
    spec = importlib.util.spec_from_file_location("train_silicon_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_resume_checkpoint_prefers_named_file(tmp_path):
    mod = _load_train_silicon_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    latest = run_dir / "latest_checkpoint.pt"
    latest.write_text("x")
    assert mod._resolve_resume_checkpoint(str(run_dir), "latest") == latest.resolve()
    assert mod._resolve_resume_checkpoint(str(latest), "best") == latest.resolve()


def test_setup_argparse_accepts_hyphen_aliases_and_scalar_matrix_target(monkeypatch):
    mod = _load_train_silicon_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_silicon.py",
            "--matrix-targets",
            "density",
            "--num-train",
            "20",
            "--num-val",
            "5",
            "--wandb-project",
            "silicon-test",
        ],
    )

    args = mod.setup_argparse()

    assert args.matrix_targets == ["density"]
    assert args.num_train == 20
    assert args.num_val == 5
    assert args.wandb_project == "silicon-test"


def test_setup_argparse_parses_optional_float_union(monkeypatch):
    mod = _load_train_silicon_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_silicon.py",
            "--grad-clip-val",
            "1.0",
        ],
    )

    args = mod.setup_argparse()

    assert isinstance(args.grad_clip_val, float)
    assert args.grad_clip_val == pytest.approx(1.0)


@pytest.mark.integration
def test_train_silicon_wires_checkpoints_and_artifacts(monkeypatch, tmp_path):
    mod = _load_train_silicon_module()

    data_root = tmp_path / "data"
    for idx in range(2):
        snap_root = data_root / "300K" / f"sample{idx}"
        snap_root.mkdir(parents=True)
        (snap_root / "Si_DM").write_text("matrix")
        (snap_root / "info.dat").write_text("info")
    ckpt_dir = tmp_path / "checkpoints"
    run_name = "silicon_test_run"
    run_dir = ckpt_dir / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "latest_checkpoint.pt").write_text("ckpt")

    class DummyDatasetFactory:
        def __init__(self, cfg):
            self.cfg = cfg
            self.snapshots = []

        def add_snapshot(self, matrix_path, info_path, purpose="train"):
            self.snapshots.append((matrix_path, info_path, purpose))

        def create(self):
            return [1], [2], object()

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.experiment = type("E", (), {"config": type("C", (), {})()})()

    captured = {}

    class DummyArtifactCallback:
        def __init__(self, *args, **kwargs):
            captured["artifact_kwargs"] = kwargs

    class DummyBenchmarkCallback:
        def __init__(self, *args, **kwargs):
            captured["benchmark_kwargs"] = kwargs

    class DummyModel:
        def __init__(self, mapper, cfg):
            self.mapper = mapper
            self.cfg = cfg

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.logger = kwargs.get("logger")

        def fit(self, *args, **kwargs):
            captured["fit_kwargs"] = kwargs

    monkeypatch.setattr(mod, "DatasetFactory", DummyDatasetFactory)
    monkeypatch.setattr(mod, "WandbLogger", DummyLogger)
    monkeypatch.setattr(mod, "ArtifactCheckpointCallback", DummyArtifactCallback)
    monkeypatch.setattr(mod, "BenchmarkCallback", DummyBenchmarkCallback)
    monkeypatch.setattr(mod, "E3GNN", DummyModel)
    monkeypatch.setattr(mod.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [str(snap_root / "Si_DM")])
    monkeypatch.setattr(mod, "_resolve_total_workers", lambda num_workers: 8)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_silicon.py",
            "--data_path",
            str(data_root),
            "--min_temp",
            "300",
            "--max_temp",
            "300",
            "--temp_step",
            "300",
            "--n_snapshots_per_temp",
            "1",
            "--val_temp",
            "300",
            "--val_n_snapshots",
            "1",
            "--checkpoint_dir",
            str(ckpt_dir),
            "--run_name",
            run_name,
            "--resume_from_checkpoint",
            str(run_dir),
            "--generate_video",
            "false",
            "--log_per_irrep_images",
            "true",
            "--log_artifacts",
            "true",
        ],
    )

    mod.main()

    assert captured["artifact_kwargs"]["output_dir"] == run_dir
    assert captured["artifact_kwargs"]["generate_video"] is False
    assert captured["artifact_kwargs"]["log_per_irrep_images"] is True
    train_loader = captured["fit_kwargs"]["train_dataloaders"]
    assert train_loader.persistent_workers is True
    assert train_loader.pin_memory is False
    assert train_loader.num_workers == 4
    assert captured["fit_kwargs"]["ckpt_path"] == str(
        (run_dir / "latest_checkpoint.pt").resolve()
    )


@pytest.mark.integration
def test_train_silicon_global_split_mode(monkeypatch, tmp_path):
    mod = _load_train_silicon_module()

    data_root = tmp_path / "data"
    for idx in range(3):
        snap_root = data_root / f"300K/sample{idx}"
        snap_root.mkdir(parents=True)
        (snap_root / "Si_DM").write_text("matrix")
        (snap_root / "info.dat").write_text("info")

    captured = {}

    class DummyDatasetFactory:
        def __init__(self, cfg):
            captured["dataset_cfg"] = cfg
            self.snapshots = []

        def add_snapshot(self, matrix_path, info_path, purpose="train"):
            self.snapshots.append((matrix_path, info_path, purpose))

        def create(self):
            captured["snapshots"] = list(self.snapshots)
            return [1], [2], object()

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.experiment = type("E", (), {"config": type("C", (), {})()})()

    class DummyModel:
        def __init__(self, mapper, cfg):
            self.mapper = mapper
            self.cfg = cfg

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            captured["fit_called"] = True

    monkeypatch.setattr(mod, "DatasetFactory", DummyDatasetFactory)
    monkeypatch.setattr(mod, "WandbLogger", DummyLogger)
    monkeypatch.setattr(mod, "ArtifactCheckpointCallback", lambda *a, **k: object())
    monkeypatch.setattr(mod, "BenchmarkCallback", lambda *a, **k: object())
    monkeypatch.setattr(mod, "E3GNN", DummyModel)
    monkeypatch.setattr(mod.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_silicon.py",
            "--data-path",
            str(data_root),
            "--num-train",
            "2",
            "--num-val",
            "1",
            "--apply-cutoff-to-targets",
            "false",
            "--benchmark",
            "false",
        ],
    )

    mod.main()

    assert captured["fit_called"] is True
    assert len(captured["snapshots"]) == 3
    assert sum(1 for _, _, purpose in captured["snapshots"] if purpose == "train") == 2
    assert sum(1 for _, _, purpose in captured["snapshots"] if purpose == "val") == 1
    assert captured["dataset_cfg"].cutoff_radius is None


def test_train_silicon_splits_workers_by_sample_ratio(monkeypatch, tmp_path):
    mod = _load_train_silicon_module()

    class DummyDatasetFactory:
        def __init__(self, cfg):
            self.cfg = cfg

        def add_snapshot(self, matrix_path, info_path, purpose="train"):
            pass

        def create(self):
            return [1, 2, 3], [4], object()

    class DummyLoader:
        def __init__(self, ds, *, num_workers=0, **kwargs):
            self.num_workers = num_workers
            self.persistent_workers = kwargs.get("persistent_workers", False)
            self.pin_memory = kwargs.get("pin_memory", False)
            self._len = len(ds)

        def __len__(self):
            return self._len

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.experiment = type("E", (), {"config": type("C", (), {})()})()

    class DummyModel:
        def __init__(self, mapper, cfg):
            self.mapper = mapper
            self.cfg = cfg

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def fit(self, *args, **kwargs):
            captured["fit_kwargs"] = kwargs

    captured = {}
    monkeypatch.setattr(mod, "DatasetFactory", DummyDatasetFactory)
    monkeypatch.setattr(mod, "DataLoader", DummyLoader)
    monkeypatch.setattr(mod, "WandbLogger", DummyLogger)
    monkeypatch.setattr(mod, "ArtifactCheckpointCallback", lambda *a, **k: object())
    monkeypatch.setattr(mod, "BenchmarkCallback", lambda *a, **k: object())
    monkeypatch.setattr(mod, "E3GNN", DummyModel)
    monkeypatch.setattr(mod.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        mod,
        "discover_snapshot_pairs",
        lambda root: [
            (tmp_path / f"matrix{i}", tmp_path / f"info{i}") for i in range(4)
        ],
    )
    monkeypatch.setattr(mod, "_resolve_total_workers", lambda num_workers: 8)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_silicon.py",
            "--data-path",
            str(tmp_path / "data"),
            "--num-train",
            "3",
            "--num-val",
            "1",
            "--benchmark",
            "false",
        ],
    )

    mod.main()

    train_loader = captured["fit_kwargs"]["train_dataloaders"]
    val_loader = captured["fit_kwargs"]["val_dataloaders"]
    assert train_loader.num_workers == 6
    assert val_loader.num_workers == 2


@pytest.mark.integration
def test_train_silicon_moves_dataset_to_gpu_and_disables_workers(monkeypatch, tmp_path):
    mod = _load_train_silicon_module()

    data_root = tmp_path / "data"
    for idx in range(2):
        snap_root = data_root / "300K" / f"sample{idx}"
        snap_root.mkdir(parents=True)
        (snap_root / "Si_DM").write_text("matrix")
        (snap_root / "info.dat").write_text("info")
    ckpt_dir = tmp_path / "checkpoints"
    run_name = "silicon_gpu_dataset_run"

    class DummyDataset:
        def __init__(self):
            self.device = "cpu"

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"x": idx}, {"y": idx}

        def to(self, device):
            self.device = torch.device(device)
            return self

    class DummyDatasetFactory:
        def __init__(self, cfg):
            self.cfg = cfg

        def add_snapshot(self, matrix_path, info_path, purpose="train"):
            pass

        def create(self):
            return DummyDataset(), DummyDataset(), object()

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.experiment = type("E", (), {"config": type("C", (), {})()})()

    class DummyModel:
        def __init__(self, mapper, cfg):
            self.mapper = mapper
            self.cfg = cfg

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def fit(self, *args, **kwargs):
            captured["fit_kwargs"] = kwargs

    class DummyLoader:
        def __init__(self, ds, *, num_workers=0, **kwargs):
            captured.setdefault("loader_kwargs", []).append(
                (num_workers, kwargs.get("pin_memory", False))
            )
            self.num_workers = num_workers
            self.persistent_workers = kwargs.get("persistent_workers", False)
            self.pin_memory = kwargs.get("pin_memory", False)
            self._len = len(ds)

        def __len__(self):
            return self._len

    captured = {}
    monkeypatch.setattr(mod, "DatasetFactory", DummyDatasetFactory)
    monkeypatch.setattr(mod, "WandbLogger", DummyLogger)
    monkeypatch.setattr(mod, "ArtifactCheckpointCallback", lambda *a, **k: object())
    monkeypatch.setattr(mod, "BenchmarkCallback", lambda *a, **k: object())
    monkeypatch.setattr(mod, "E3GNN", DummyModel)
    monkeypatch.setattr(mod.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(mod, "DataLoader", DummyLoader)
    monkeypatch.setattr(mod, "_log_dataset_and_model_context", lambda *a, **k: None)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_silicon.py",
            "--data-path",
            str(data_root),
            "--num-train",
            "1",
            "--num-val",
            "1",
            "--dataset-device",
            "cuda",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--run-name",
            run_name,
            "--benchmark",
            "false",
            "--log-artifacts",
            "false",
        ],
    )

    mod.main()

    train_loader = captured["fit_kwargs"]["train_dataloaders"]
    val_loader = captured["fit_kwargs"]["val_dataloaders"]
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
    assert train_loader.pin_memory is False
    assert val_loader.pin_memory is False
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False
