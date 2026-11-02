"""Small frozen-input audit; no benchmark or test-label evaluation."""

import json
from pathlib import Path

import h5py
from omegaconf import OmegaConf
import pytest
import torch

from scripts.audit_operator_accuracy import (
    aggregate_errors,
    audit,
    file_hash,
    predictor_signature,
    snapshot_checkpoint,
)
from scripts.dataset import dataset_paths, input_manifest
from scripts.train import build_components
from test_operator_pilots import fixture_config
from utils.units import HARTREE_TO_EV


def frozen_fixture(root):
    config = fixture_config(root)
    document = OmegaConf.load(config)
    document.data.splits.test = []
    document.apply_cutoff_to_targets = False
    OmegaConf.save(document, config)
    # Both shifted-self directions remain supervised beyond the radial range.
    for split in ("train", "val"):
        with h5py.File(root / split / "hamiltonians.h5", "a") as stream:
            for shift in (-1, 1):
                stream[str([shift, 0, 0, 1, 1])] = [[0.25]]
    _, _, model, _, inputs = build_components(document, resume=True)
    manifest = root / "pilot_manifest.json"
    manifest.write_text(
        json.dumps(
            {"config": model.cfg.to_mapping(), "specification": {"inputs": inputs}}
        )
    )
    return config, manifest, model


def test_tiny_audit_certifies_units_directions_and_checkpoint_partitions(
    tmp_path, monkeypatch
):
    config, manifest, model = frozen_fixture(tmp_path)
    source = tmp_path / "pilot.pt"
    copied = tmp_path / "snapshots" / "pilot.pt"
    torch.save({"state_dict": model.state_dict(), "epoch": 3}, source)
    original_hash = file_hash(source)
    original_stat = source.stat()

    original_open = Path.open
    test_directory = tmp_path / "test"

    def reject_test_access(path, *args, **kwargs):
        if path.resolve().is_relative_to(test_directory):
            raise AssertionError("Audit attempted to access held-out test files")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_test_access)
    result = audit(config, manifest, checkpoint=source, checkpoint_copy=copied)
    assert result["passed"]
    assert result["test_inputs"] == 0
    assert result["input_hashes_match_pilot"]
    assert result["predictor_configuration_matches_pilot"]
    source_root = Path(__file__).resolve().parents[2] / "src"
    assert result["module_origins"] == {
        name: str(source_root.joinpath(*name.split(".")).with_suffix(".py"))
        for name in ("data.sample", "data.factory", "data.snapshot", "net.e3gnn")
    }
    for name, value in result["checks"].items():
        assert value <= result["check_thresholds"][name]
    validation = result["splits"]["val"]
    assert validation["entries_by_pair"] == {"C-C": 16, "C-H": 4, "H-C": 4, "H-H": 3}
    assert validation["max_stored_edge_length_angstrom"] == pytest.approx(8.0)
    # The full 5x5 target has mean 0.022 Ha; periodic H-H blocks are 0.25 eV each.
    assert validation["zero_predictor"]["dataset_entry_mae_mev"] == pytest.approx(
        (25 * 0.022 * HARTREE_TO_EV + 0.5) / 27 * 1000, rel=2e-6
    )
    measured = result["validation_checkpoint"]
    assert measured["epoch"] == 3
    assert measured["sha256"] == file_hash(copied) == original_hash
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert file_hash(source) == original_hash
    metrics = measured["metrics"]
    assert metrics["raw"]["entries"] == metrics["projected"]["entries"] == 27
    for kind in ("raw", "projected"):
        assert metrics[kind + "_onsite"]["entries"] == 17
        assert metrics[kind + "_offsite"]["entries"] == 10
        assert metrics[kind]["entries"] == (
            metrics[kind + "_onsite"]["entries"] + metrics[kind + "_offsite"]["entries"]
        )
    assert (
        metrics["projected"]["dataset_entry_mae_mev"]
        <= metrics["raw"]["dataset_entry_mae_mev"] + 1e-5
    )


def test_corrupt_input_hash_is_rejected_before_materializing_data(
    tmp_path, monkeypatch
):
    config = fixture_config(tmp_path)
    document = OmegaConf.load(config)
    document.data.splits.test = []
    OmegaConf.save(document, config)
    inputs = input_manifest(dataset_paths(document))
    inputs["train"][0]["sha256"]["hamiltonians.h5"] = "0" * 64
    manifest = tmp_path / "pilot_manifest.json"
    manifest.write_text(json.dumps({"specification": {"inputs": inputs}}))

    def refuse_materialization(*args, **kwargs):
        raise AssertionError(
            "Corrupt hashes should be rejected before factory creation"
        )

    monkeypatch.setattr("scripts.train.build_components", refuse_materialization)
    with pytest.raises(ValueError, match="content hashes differ"):
        audit(config, manifest)


def test_stale_import_is_rejected_before_inputs_or_factory(tmp_path, monkeypatch):
    import data.sample

    monkeypatch.setattr(data.sample, "__file__", str(tmp_path / "stale/data/sample.py"))

    def refuse_scientific_access(*args, **kwargs):
        raise AssertionError(
            "Import-origin rejection must precede input hashing/factory"
        )

    monkeypatch.setattr("scripts.dataset.input_manifest", refuse_scientific_access)
    monkeypatch.setattr("scripts.train.build_components", refuse_scientific_access)
    with pytest.raises(ValueError, match="data.sample is outside frozen source"):
        audit(tmp_path / "never_opened.yaml", tmp_path / "never_opened_manifest.json")


def test_nonempty_test_inputs_are_refused_before_factory_or_hashing(
    tmp_path, monkeypatch
):
    config = fixture_config(tmp_path)

    def refuse_access(*args, **kwargs):
        raise AssertionError(
            "Test refusal must precede materialization and input hashing"
        )

    monkeypatch.setattr("scripts.train.build_components", refuse_access)
    monkeypatch.setattr("scripts.dataset.input_manifest", refuse_access)
    with pytest.raises(ValueError, match="refuses test inputs"):
        audit(config, tmp_path / "unused_manifest.json")


def test_shape_compatible_predictor_change_is_rejected(tmp_path):
    config, manifest, _ = frozen_fixture(tmp_path)
    document = OmegaConf.load(config)
    document.radial_weight_granularity = "copy"
    OmegaConf.save(document, config)
    with pytest.raises(ValueError, match="Prediction configuration differs"):
        audit(config, manifest)


def test_old_defaults_and_explicit_optimizer_do_not_change_predictor_identity():
    from net.common import Config

    old = Config().to_mapping()
    old.pop("radial_weight_granularity")
    old.pop("matrix_loss_reduction")
    old["dtype"] = "torch.float32"
    explicit = Config().to_mapping()
    explicit["optimization"] = {"optimizer": {"kind": "adamw", "weight_decay": 0.01}}
    assert predictor_signature(old) == predictor_signature(explicit)
    explicit["radial_weight_granularity"] = "copy"
    assert predictor_signature(old) != predictor_signature(explicit)


def test_checkpoint_snapshot_is_exclusive_and_rejects_changing_source(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "pilot.pt", tmp_path / "copy.pt"
    source.write_bytes(b"original checkpoint")
    destination.write_bytes(b"preserve existing snapshot")
    with pytest.raises(FileExistsError):
        snapshot_checkpoint(source, destination)
    assert destination.read_bytes() == b"preserve existing snapshot"
    with pytest.raises(ValueError, match="differ"):
        snapshot_checkpoint(source, source)
    destination.unlink()

    import scripts.audit_operator_accuracy as module

    original_copy = module.shutil.copyfileobj

    def changing_copy(original, copied):
        original_copy(original, copied)
        source.write_bytes(b"new checkpoint generation")

    monkeypatch.setattr(module.shutil, "copyfileobj", changing_copy)
    with pytest.raises(RuntimeError, match="changed during snapshot"):
        snapshot_checkpoint(source, destination)
    assert not destination.exists()


def test_structure_and_dataset_entry_reductions_remain_distinct():
    rows = [
        {"absolute_sum_ev": 1.0, "squared_sum_ev2": 1.0, "entries": 1, "pairs": {}},
        {"absolute_sum_ev": 0.0, "squared_sum_ev2": 0.0, "entries": 9, "pairs": {}},
    ]
    result = aggregate_errors(rows)
    assert result["structure_mean_mae_mev"] == 500
    assert result["dataset_entry_mae_mev"] == 100
