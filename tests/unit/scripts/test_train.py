from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
