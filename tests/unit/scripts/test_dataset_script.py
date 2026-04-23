from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/dataset.py").resolve()
    spec = importlib.util.spec_from_file_location("dataset_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discover_siox_snapshot_pairs_prefers_hs_and_sio2_out(tmp_path):
    mod = _load_module()
    sample = tmp_path / "siox" / "0_123"
    sample.mkdir(parents=True)
    (sample / "HS.out").write_text("matrix")
    (sample / "SiO2.out").write_text("info")
    (sample / "log.out").write_text("log")

    pairs = mod.discover_siox_snapshot_pairs(tmp_path / "siox")

    assert pairs == [((sample / "HS.out").resolve(), (sample / "SiO2.out").resolve())]


def test_build_datasets_from_yaml_dispatches_to_siox(monkeypatch):
    mod = _load_module()
    captured = {}

    def fake_build_siox_datasets(**kwargs):
        captured["kwargs"] = kwargs
        return "train", "val", "mapper"

    monkeypatch.setattr(mod, "build_siox_datasets", fake_build_siox_datasets)
    cfg = mod.Config()
    parsed_yaml = {
        "parameters": {
            "dataset-kind": {"value": "siox"},
            "data-path": {"value": "/tmp/siox"},
            "num-train": {"value": 4},
            "num-val": {"value": 1},
        }
    }

    result = mod.build_datasets_from_yaml(parsed_yaml, cfg)

    assert result == ("train", "val", "mapper")
    assert captured["kwargs"]["data_path"] == "/tmp/siox"
    assert captured["kwargs"]["num_train"] == 4
    assert captured["kwargs"]["num_val"] == 1


def test_build_silicon_datasets_single_temp_global_split_is_disjoint(
    monkeypatch, tmp_path
):
    mod = _load_module()
    captured = {}

    def fake_create_datasets_from_pairs(
        train_pairs, val_pairs, cfg, *, convention="e3nn"
    ):
        captured["train_pairs"] = train_pairs
        captured["val_pairs"] = val_pairs
        captured["convention"] = convention
        return "train", "val", "mapper"

    monkeypatch.setattr(
        mod, "_create_datasets_from_pairs", fake_create_datasets_from_pairs
    )
    temp_root = tmp_path / "3000K"
    for idx in range(12):
        sample = temp_root / f"{idx}"
        sample.mkdir(parents=True)
        (sample / "Si_DM").write_text("matrix")
        (sample / "info.dat").write_text("info")

    result = mod.build_silicon_datasets(
        data_path=tmp_path,
        cfg=mod.Config(),
        min_temp=3000,
        max_temp=3000,
        val_temp=3000,
        num_train=10,
        num_val=2,
        seed=0,
    )

    assert result == ("train", "val", "mapper")
    assert len(captured["train_pairs"]) == 10
    assert len(captured["val_pairs"]) == 2
    assert set(captured["train_pairs"]).isdisjoint(set(captured["val_pairs"]))
    assert all(
        "3000K" in str(matrix_path) for matrix_path, _ in captured["train_pairs"]
    )
    assert all("3000K" in str(matrix_path) for matrix_path, _ in captured["val_pairs"])
