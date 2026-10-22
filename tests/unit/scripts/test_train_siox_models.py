"""SiOx split parity, paper recipe, direct source integrity and safe continuation."""

import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from scripts import dataset
from scripts.train_siox_models import run


def synthetic_source(root, count=105):
    for index in range(count):
        path = root / f"snapshot_{index}"
        path.mkdir(parents=True)
        (path / "HS.out").write_text(f"matrix {index}")
        (path / "SiO2.out").write_text("info")
    return root


def test_split_exactly_matches_paper_builder(tmp_path, monkeypatch):
    source = synthetic_source(tmp_path / "raw")
    selection = run.select_inputs(source)
    captured = {}

    def capture(train, val, test, cfg, *, convention):
        captured.update(train=train, val=val, test=test)

    monkeypatch.setattr(dataset, "_create_datasets_from_pairs", capture)
    dataset.build_siox_datasets(
        data_path=source,
        cfg=dataset.Config(),
        num_train=90,
        num_val=10,
        num_test=0,
        data_split_seed=42,
    )
    for split in ("train", "val", "test"):
        assert [
            (r["matrix_path"], r["info_path"]) for r in selection["splits"][split]
        ] == [(str(matrix), str(info)) for matrix, info in captured[split]]
    manifest = tmp_path / "manifest.json"
    run.write_json(manifest, selection)
    # The legacy trainer's explicit-manifest path preserves the same order.
    dataset.build_datasets_from_manifest(manifest, dataset.Config())
    assert len(captured["train"]) == 90
    assert len(captured["val"]) == 10
    settings = [
        run.model_settings(model, selection, manifest, tmp_path / "checkpoints")
        for model in run.MODELS
    ]
    native_splits = settings[0]["data"]["splits"]
    for model, setting in zip(run.MODELS[:3], settings[:3]):
        assert setting["data"]["splits"] == native_splits
        source_config = OmegaConf.to_container(
            OmegaConf.load(run.ROOT / f"configs/reference/{model}.yaml"), resolve=False
        )
        assert setting["pipeline"] == source_config["pipeline"]
        assert setting["optimization"] == source_config["optimization"]
        assert setting["objective"] == source_config["objective"]
    paper = json.loads((run.ROOT / "scripts/reproduce_examples/sio2.json").read_text())
    ignored = {
        "run_name",
        "save_dir",
        "wandb_project",
        "snapshot_cache_dir",
        "hamiltonian_envelope_path",
    }
    for key, value in paper["config"].items():
        if key not in ignored:
            assert settings[3][key] == value, key
    assert settings[3]["data"]["splits"] == native_splits
    assert settings[3]["lr"] == 0.0027
    assert settings[3]["compatibility"] is False
    assert settings[3].get("resume_from_checkpoint") is None


def test_frozen_raw_manifest_can_resume_and_detects_tampering(tmp_path):
    source = tmp_path / "raw"
    fixture = run.ROOT / "data/small/Si"
    for index in range(2):
        snapshot = source / str(index)
        snapshot.mkdir(parents=True)
        for name in ("HS.out", "Si.out", "Si.cif"):
            shutil.copyfile(fixture / name, snapshot / name)
    args = SimpleNamespace(
        data_root=source,
        native_root=tmp_path / "native",
        records=tmp_path / "records",
        num_train=1,
        num_val=1,
        workers=2,
    )
    run.prepare(args)
    path = args.records / "dataset.json"
    first = path.read_bytes()
    run.prepare(args)
    assert path.read_bytes() == first
    manifest = run.validate_manifest(path)
    assert "conversions" not in manifest
    assert not args.native_root.exists()
    raw = Path(manifest["splits"]["train"][0]["matrix_path"])
    original = raw.read_bytes()
    raw.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        run.validate_manifest(path)
    raw.write_bytes(original)
    extra = raw.parent / "added.cif"
    extra.write_text("new geometry")
    with pytest.raises(ValueError, match="inventory changed"):
        run.validate_manifest(path)
    extra.unlink()


def test_safe_recovery_and_duplicate_writer_protection(tmp_path):
    directory = tmp_path / "checkpoints"
    assert run.recovery_checkpoint(directory, action="auto") is None
    with pytest.raises(FileNotFoundError):
        run.recovery_checkpoint(directory, action="continue")
    directory.mkdir()
    with pytest.raises(RuntimeError, match="no safe checkpoint"):
        run.recovery_checkpoint(directory, action="auto")
    checkpoint = directory / "latest_checkpoint.pt"
    torch.save({"epoch_boundary_resume": {"safe_to_resume": False}}, checkpoint)
    with pytest.raises(ValueError, match="not a safe epoch boundary"):
        run.recovery_checkpoint(directory, action="auto")
    torch.save({"epoch_boundary_resume": {"safe_to_resume": True}}, checkpoint)
    assert run.recovery_checkpoint(directory, action="auto") == checkpoint
    with pytest.raises(FileExistsError):
        run.recovery_checkpoint(directory, action="start")
    with run.exclusive(tmp_path / ".model.lock"):
        with pytest.raises(BlockingIOError):
            with run.exclusive(tmp_path / ".model.lock"):
                pytest.fail("A second writer acquired the lock")


def test_train_resumes_its_own_run_and_stops_at_completion(tmp_path, monkeypatch):
    from scripts import train as shared_training

    args = SimpleNamespace(
        records=tmp_path / "records",
        model="deeph_e3",
        checkpoint_root=tmp_path / "models",
        plan=False,
        action="auto",
    )
    args.records.mkdir()
    run.write_json(args.records / "dataset.json", {})
    monkeypatch.setattr(run, "validate_manifest", lambda path: {})
    monkeypatch.setattr(
        run,
        "model_settings",
        lambda *args: {"training": {"max_epochs": 3, "min_lr": 0.00003}},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def fit(cli):
        calls.append(cli)
        output = Path(cli[cli.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": 1 if len(calls) == 1 else 3,
                "epoch_boundary_resume": {"safe_to_resume": True},
                "optimizer_states": [{"param_groups": [{"lr": 0.002}]}],
            },
            output / "latest_checkpoint.pt",
        )

    monkeypatch.setattr(shared_training, "main", fit)
    run.train(args)
    assert "--resume" not in calls[0]
    assert (args.records / "deeph_e3.last_segment.json").is_file()
    run.train(args)
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--resume") + 1] == str(
        args.checkpoint_root / args.model / "latest_checkpoint.pt"
    )
    assert (args.records / "deeph_e3.complete.json").is_file()
    run.train(args)
    assert len(calls) == 2


def test_preparation_does_not_change_a_frozen_selection(tmp_path):
    source = synthetic_source(tmp_path / "raw", count=3)
    first = run.select_inputs(source, num_train=2, num_val=1)
    path = tmp_path / "selection.json"
    run.immutable_json(path, first)
    random.Random(1).shuffle(first["splits"]["train"])
    first["seed"] = 5
    with pytest.raises(ValueError, match="Frozen specification changed"):
        run.immutable_json(path, first)


def test_legacy_trainer_dispatches_explicit_manifest_before_discovery(monkeypatch):
    from scripts import train

    received = {}

    def build(path, cfg, *, convention):
        received.update(path=path, convention=convention)
        return "train", "val", "mapper"

    monkeypatch.setattr(train, "build_datasets_from_manifest", build)
    args = SimpleNamespace(dataset_manifest="/frozen/dataset.json", dataset_kind="siox")
    assert train._build_dataset_bundle(args, dataset.Config(), None) == (
        "train",
        "val",
        "mapper",
    )
    assert received == {"path": "/frozen/dataset.json", "convention": "e3nn"}
    with pytest.raises(ValueError, match="either an explicit"):
        train._build_dataset_bundle(args, dataset.Config(), {"parameters": {}})


@pytest.mark.parametrize("model", run.MODELS)
def test_every_preset_uses_same_training_command_and_full_checkpoint(
    tmp_path, monkeypatch, model
):
    from scripts import train

    args = SimpleNamespace(
        records=tmp_path / "records",
        model=model,
        checkpoint_root=tmp_path / "checkpoints",
        plan=False,
        action="auto",
    )
    args.records.mkdir()
    manifest = {"splits": {key: [] for key in ("train", "val", "test")}}
    run.write_json(args.records / "dataset.json", manifest)
    monkeypatch.setattr(run, "validate_manifest", lambda path: manifest)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def fit(cli):
        calls.append(cli)
        document = OmegaConf.to_container(
            OmegaConf.load(cli[cli.index("--config") + 1]), resolve=False
        )
        if model == "mandala_paper":
            cfg = train.Config.from_mapping(document)
            assert cfg.lr == 0.0027
            assert cfg.loss_l1_fraction == 1
            assert cfg.hamiltonian_envelope_mode == "multiply_prediction"
        output = Path(cli[cli.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": 1 if len(calls) == 1 else 10000,
                "epoch_boundary_resume": {"safe_to_resume": True},
                "optimizer_states": [{"param_groups": [{"lr": 0.0001}]}],
            },
            output / "latest_checkpoint.pt",
        )

    monkeypatch.setattr(train, "main", fit)
    run.train(args)
    assert "--resume" not in calls[0]
    run.train(args)
    assert "--resume" in calls[1]
    run.train(args)
    assert len(calls) == 2
