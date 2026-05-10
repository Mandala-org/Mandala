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


def test_setup_argparse_accepts_resume_from_wandb(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_run.py",
            "--resume-from-wandb",
            "https://wandb.ai/acme/project/runs/abc123",
        ],
    )

    args = mod.setup_argparse()

    assert args.resume_from_wandb == "https://wandb.ai/acme/project/runs/abc123"


def test_setup_argparse_accepts_fork_run(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_run.py",
            "--fork-run",
            "true",
        ],
    )

    args = mod.setup_argparse()

    assert args.fork_run is True


def test_setup_argparse_accepts_max_wall_clock_hours(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_run.py",
            "--max-wall-clock-hours",
            "12",
        ],
    )

    args = mod.setup_argparse()
    mod._normalize_wall_clock_args(args)

    assert args.max_wall_clock_hours == 12.0
    assert args.max_wall_clock_seconds == 43200.0


def test_parse_wandb_run_url_accepts_sweep_run_url():
    mod = _load_module()

    entity, project, run_id = mod._parse_wandb_run_url(
        "https://wandb.ai/b-brzoza/mandala-matrices/sweeps/xqqtet2h/runs/e9yo3gkj"
    )

    assert entity == "b-brzoza"
    assert project == "mandala-matrices"
    assert run_id == "e9yo3gkj"


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


def test_normalize_wall_clock_args_rejects_both_units():
    mod = _load_module()
    args = mod.argparse.Namespace(
        max_wall_clock_seconds=60.0,
        max_wall_clock_hours=1.0,
        _explicit_args={"max_wall_clock_seconds", "max_wall_clock_hours"},
    )

    try:
        mod._normalize_wall_clock_args(args)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when both time-budget units are set")


def test_resolve_wandb_resume_uses_summary_checkpoint_paths(monkeypatch):
    mod = _load_module()

    class DummyRun:
        name = "blooming-oath-3"
        config = {
            "data_path": "/tmp/siox",
            "lr": 0.01,
            "matrix-targets": "hamiltonian,density,overlap",
        }
        summary = {
            "checkpoint/latest_path": "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt",
            "checkpoint/best_path": "checkpoints/siox/blooming-oath-3/best_model.pt",
        }

    class DummyApi:
        def run(self, path):
            assert (
                path
                == "b-brzoza/mandala-minimal-siox-hamiltonian-energy-halfgt-rosi/9onx8a88"
            )
            return DummyRun()

    class DummyWandb:
        Api = DummyApi

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)

    resolved = mod._resolve_wandb_resume(
        "https://wandb.ai/b-brzoza/mandala-minimal-siox-hamiltonian-energy-halfgt-rosi/runs/9onx8a88",
        "latest",
    )

    assert (
        resolved["checkpoint_path"]
        == "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt"
    )
    assert resolved["run_name"] == "blooming-oath-3"
    assert resolved["checkpoint_dir"] == "checkpoints/siox"
    assert resolved["config"]["data_path"] == "/tmp/siox"
    assert resolved["config"]["lr"] == 0.01
    assert resolved["config"]["matrix_targets"] == "hamiltonian,density,overlap"


def test_apply_wandb_resume_metadata_sets_checkpoint_and_run_dir(monkeypatch):
    mod = _load_module()
    args = mod.argparse.Namespace(
        resume_from_wandb="https://wandb.ai/acme/project/runs/abc123",
        resume_from_checkpoint=None,
        fork_run=False,
        resume_mode="latest",
        run_name=None,
        checkpoint_dir="checkpoints/main",
        wandb_project=None,
        data_path=None,
        lr=3e-4,
        matrix_targets=["density"],
        _explicit_args=set(),
    )

    monkeypatch.setattr(
        mod,
        "_resolve_wandb_resume",
        lambda url, mode: {
            "checkpoint_path": "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt",
            "run_name": "blooming-oath-3",
            "checkpoint_dir": "checkpoints/siox",
            "project": "project",
            "config": {
                "data_path": "/tmp/siox",
                "lr": 0.01,
                "matrix_targets": "hamiltonian,density,overlap",
            },
        },
    )

    mod._apply_wandb_resume_metadata(args)

    assert (
        args.resume_from_checkpoint
        == "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt"
    )
    assert args.run_name == "blooming-oath-3"
    assert args.checkpoint_dir == "checkpoints/siox"
    assert args.wandb_project == "project"
    assert args.data_path == "/tmp/siox"
    assert args.lr == 0.01
    assert args.matrix_targets == "hamiltonian,density,overlap"


def test_apply_wandb_resume_metadata_keeps_explicit_overrides(monkeypatch):
    mod = _load_module()
    args = mod.argparse.Namespace(
        resume_from_wandb="https://wandb.ai/acme/project/runs/abc123",
        resume_from_checkpoint=None,
        fork_run=False,
        resume_mode="latest",
        run_name="manual-name",
        checkpoint_dir="manual/checkpoints",
        wandb_project="manual-project",
        data_path="/manual/data",
        lr=0.002,
        _explicit_args={
            "run_name",
            "checkpoint_dir",
            "wandb_project",
            "data_path",
            "lr",
        },
    )

    monkeypatch.setattr(
        mod,
        "_resolve_wandb_resume",
        lambda url, mode: {
            "checkpoint_path": "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt",
            "run_name": "blooming-oath-3",
            "checkpoint_dir": "checkpoints/siox",
            "project": "project",
            "config": {
                "data_path": "/tmp/siox",
                "lr": 0.01,
            },
        },
    )

    mod._apply_wandb_resume_metadata(args)

    assert (
        args.resume_from_checkpoint
        == "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt"
    )
    assert args.run_name == "manual-name"
    assert args.checkpoint_dir == "manual/checkpoints"
    assert args.wandb_project == "manual-project"
    assert args.data_path == "/manual/data"
    assert args.lr == 0.002


def test_apply_wandb_resume_metadata_fork_run_keeps_new_run_name_unset(monkeypatch):
    mod = _load_module()
    args = mod.argparse.Namespace(
        resume_from_wandb="https://wandb.ai/acme/project/runs/abc123",
        resume_from_checkpoint=None,
        fork_run=True,
        resume_mode="latest",
        run_name=None,
        checkpoint_dir="checkpoints/main",
        wandb_project=None,
        data_path=None,
        lr=3e-4,
        _explicit_args=set(),
    )

    monkeypatch.setattr(
        mod,
        "_resolve_wandb_resume",
        lambda url, mode: {
            "checkpoint_path": "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt",
            "run_name": "blooming-oath-3",
            "checkpoint_dir": "checkpoints/siox",
            "project": "project",
            "config": {
                "data_path": "/tmp/siox",
                "lr": 0.01,
            },
        },
    )

    mod._apply_wandb_resume_metadata(args)

    assert (
        args.resume_from_checkpoint
        == "checkpoints/siox/blooming-oath-3/latest_checkpoint.pt"
    )
    assert args.run_name is None
    assert args.checkpoint_dir == "checkpoints/siox"
    assert args.wandb_project == "project"
    assert args.data_path == "/tmp/siox"
    assert args.lr == 0.01


def test_validate_required_args_accepts_wandb_filled_data_path():
    mod = _load_module()
    args = mod.argparse.Namespace(data_path="/tmp/data")

    mod._validate_required_args(args)
