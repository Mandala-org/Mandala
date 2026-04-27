from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/wandb_run.py").resolve()
    spec = importlib.util.spec_from_file_location("wandb_run_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_setup_argparse_accepts_dataset_kind_and_aliases(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_run.py",
            "--dataset-kind",
            "siox",
            "--data-path",
            "/tmp/siox",
            "--num-train",
            "4",
            "--num-val",
            "1",
            "--matrix-targets",
            "density",
        ],
    )

    args = mod.setup_argparse()

    assert args.dataset_kind == "siox"
    assert args.data_path == "/tmp/siox"
    assert args.num_train == 4
    assert args.num_val == 1
    assert args.matrix_targets == ["density"]


def test_setup_argparse_leaves_run_name_unset_by_default(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_run.py",
            "--data-path",
            "/tmp/data",
        ],
    )

    args = mod.setup_argparse()

    assert args.run_name is None


def test_main_passes_parsed_yaml_to_run_training(monkeypatch, tmp_path):
    mod = _load_module()
    sweep_yaml = tmp_path / "sweep.yaml"
    sweep_yaml.write_text(
        """
parameters:
  dataset-kind:
    value: silicon
  data-path:
    value: /tmp/data
"""
    )
    captured = {}

    def fake_run_training(
        args, *, parsed_yaml=None, extra_callbacks=None, objective_metric=None
    ):
        captured["args"] = args
        captured["parsed_yaml"] = parsed_yaml
        return {}

    monkeypatch.setattr(mod, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_run.py",
            "--data-path",
            "/tmp/data",
            "--sweep-yaml",
            str(sweep_yaml),
        ],
    )

    mod.main()

    assert captured["args"].data_path == "/tmp/data"
    assert captured["parsed_yaml"]["parameters"]["dataset-kind"]["value"] == "silicon"
