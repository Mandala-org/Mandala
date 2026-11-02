"""Compatibility/memory probing guards, with synthetic CPU model data only."""

from copy import deepcopy
import json

from omegaconf import OmegaConf
import pytest
import torch

from scripts import probe_operator_runtime as probe


def document():
    return {
        "data": {
            "splits": {
                "train": ["train1", "train2"],
                "val": ["val1", "val2"],
                "test": [],
            }
        }
    }


def test_technical_subset_preserves_original_and_refuses_test_membership():
    original = document()
    preserved = deepcopy(original)
    selected = probe.miniature_document(original)
    assert original == preserved
    assert selected["data"]["splits"] == {
        "train": ["train1"],
        "val": ["val1"],
        "test": [],
    }
    original["data"]["splits"]["test"] = ["heldout"]
    with pytest.raises(ValueError, match="refuses test"):
        probe.miniature_document(original)


def test_probe_refuses_test_before_runtime_initialization(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    values = document()
    values["data"]["splits"]["test"] = ["heldout"]
    OmegaConf.save(OmegaConf.create(values), config)
    monkeypatch.setattr(
        probe,
        "runtime_metadata",
        lambda **kwargs: pytest.fail("GPU must not initialize"),
    )
    with pytest.raises(ValueError, match="refuses test"):
        probe.probe(config)


def test_requires_slurm_before_imports_or_gpu(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        probe, "module_origins", lambda: pytest.fail("No runtime on login")
    )
    with pytest.raises(RuntimeError, match="allocated Slurm"):
        probe.runtime_metadata()


def test_legacy_versions_are_explicit():
    probe.validate_versions(
        {"torch": "2.5.1+cu121", "cuda": "12.1"},
        expected_torch_prefix="2.5.1",
        expected_cuda_prefix="12.1",
    )
    with pytest.raises(ValueError, match="Expected cuda"):
        probe.validate_versions(
            {"torch": "2.7.1+cu128", "cuda": "12.8"}, expected_cuda_prefix="12.1"
        )


def test_exclusive_json_is_atomic_and_preserves_existing(tmp_path):
    output = tmp_path / "probe.json"
    probe.write_exclusive_json(output, {"passed": True})
    with pytest.raises(FileExistsError):
        probe.write_exclusive_json(output, {"passed": False})
    assert json.loads(output.read_text()) == {"passed": True}
    assert list(tmp_path.iterdir()) == [output]


def test_shared_probe_backward_on_synthetic_cpu_fixture(tmp_path):
    from test_operator_pilots import fixture_config

    config = fixture_config(tmp_path)
    values = OmegaConf.load(config)
    values.data.splits.test = []
    result = probe.exercise_model(
        probe.miniature_document(values), device=torch.device("cpu")
    )
    assert result["structures"] == {"train": 1, "val": 1, "test": 0}
    assert result["training_backward_finite"] is True
    assert result["raw_validation_outputs_finite"] is True
    assert result["optimizer_steps"] == 0
