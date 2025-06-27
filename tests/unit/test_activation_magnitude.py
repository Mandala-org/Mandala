import pytest
import yaml
import torch
from types import SimpleNamespace

from net.benchmark import BenchmarkCallback


@pytest.mark.unit
def test_activation_magnitude_logging(tmp_path, monkeypatch):
    # Ensure a clean workspace for benchmarks
    monkeypatch.chdir(tmp_path)

    # Initialize callback with activation magnitude logging enabled
    callback = BenchmarkCallback(verbosity=1, log_activation_mag=True)
    # Create a fake trainer with minimal attributes
    trainer = SimpleNamespace()
    trainer.train_dataloader = []
    trainer.val_dataloaders = []
    trainer.logger = SimpleNamespace(
        name="TESTRUN",
        experiment=SimpleNamespace(config={}),
        log_metrics=lambda m: None,
    )
    trainer.devices = ["cpu"]
    trainer.precision = 32

    # Simulate on_fit_start to set environment and timestamp
    callback.on_fit_start(trainer, pl_module=None)
    # Override timestamp for deterministic output path
    callback.environment["timestamp"] = "TEST_TS"

    # Simulate one training batch with synthetic activation magnitudes
    pl_mod_train = SimpleNamespace()
    pl_mod_train._last_activation_mags = {"mag_test_l_0": torch.tensor([3.0, 4.0])}
    callback.on_train_batch_start(trainer, pl_mod_train, batch=None, batch_idx=0)
    callback.on_train_batch_end(
        trainer, pl_mod_train, outputs=None, batch=None, batch_idx=0
    )

    # Simulate one validation batch
    pl_mod_val = SimpleNamespace()
    pl_mod_val._last_activation_mags = {"mag_test_l_1": torch.tensor([5.0])}
    callback.on_validation_batch_start(trainer, pl_mod_val, batch=None, batch_idx=0)
    callback.on_validation_batch_end(
        trainer, pl_mod_val, outputs=None, batch=None, batch_idx=0
    )

    # Finalize and write the benchmark report
    callback.on_fit_end(trainer, pl_mod_val)

    # Verify the report file exists
    report_path = tmp_path / "benchmarks" / "TESTRUN_TEST_TS_bench.yaml"
    assert report_path.exists(), f"Expected report at {report_path}"

    # Load and inspect the report
    report = yaml.safe_load(report_path.read_text())
    assert "activation_magnitudes" in report
    act = report["activation_magnitudes"]
    # Should have both train and val entries
    assert "train" in act and "val" in act
    # Check that our synthetic tags are present
    train_act = act["train"]
    val_act = act["val"]
    assert "mag_test_l_0" in train_act
    assert "mean" in train_act["mag_test_l_0"]
    assert "mag_test_l_1" in val_act
    assert "mean" in val_act["mag_test_l_1"]
