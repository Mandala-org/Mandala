import pytest
import yaml
import torch
import numpy as np
from types import SimpleNamespace
from collections import OrderedDict

from net.benchmark import BenchmarkCallback


@pytest.mark.unit
def test_activation_magnitude_logging_rewrite(tmp_path, monkeypatch):
    """
    Verify that the BenchmarkCallback correctly logs activation magnitudes
    with a nested structure in the final YAML report.
    """
    # 1. Arrange: Set up the test environment and mock objects.
    monkeypatch.chdir(tmp_path)

    callback = BenchmarkCallback(verbosity=1, log_activation_mag=True)
    logged_metrics = {}
    trainer = SimpleNamespace(
        train_dataloader=[],
        val_dataloaders=[],
        logger=SimpleNamespace(
            name="TESTRUN",
            experiment=SimpleNamespace(config={}),
            log_metrics=lambda m: logged_metrics.update(m),
        ),
        devices=["cpu"],
        precision=32,
    )

    callback.on_fit_start(trainer, pl_module=None)
    callback.environment["timestamp"] = "REWRITE_TS"

    # Mock pl_module with realistic activation magnitudes
    pl_module = SimpleNamespace()
    pl_module._last_activation_mags = OrderedDict(
        [
            ("mag_node_encoding_32x0e", torch.tensor([1.0, 2.0])),
            ("mag_node_encoding_32x0o", torch.tensor([3.0, 4.0])),
            ("mag_edge_encoding_16x1e", torch.tensor([5.0, 6.0])),
        ]
    )

    # 2. Act: Simulate the training and validation steps.
    callback.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    callback.on_train_batch_end(
        trainer, pl_module, outputs=None, batch=None, batch_idx=0
    )

    # Use different magnitudes for the validation step
    pl_module_val = SimpleNamespace()
    pl_module_val._last_activation_mags = OrderedDict(
        [
            ("mag_node_encoding_32x0e", torch.tensor([7.0, 8.0])),
        ]
    )
    callback.on_validation_batch_start(trainer, pl_module_val, batch=None, batch_idx=0)
    callback.on_validation_batch_end(
        trainer, pl_module_val, outputs=None, batch=None, batch_idx=0
    )

    callback.on_fit_end(trainer, pl_module)

    # 3. Assert: Verify the output.
    report_path = tmp_path / "benchmarks" / "TESTRUN_REWRITE_TS_bench.yaml"
    assert report_path.exists(), f"Expected report at {report_path}"

    report = yaml.safe_load(report_path.read_text())
    assert "activation_magnitudes" in report

    act_mags = report["activation_magnitudes"]
    assert "train" in act_mags
    assert "val" in act_mags

    # Verify training stats
    train_mags = act_mags["train"]
    assert "node_encoding" in train_mags
    assert "edge_encoding" in train_mags
    assert "32x0e" in train_mags["node_encoding"]
    np.testing.assert_allclose(train_mags["node_encoding"]["32x0e"]["mean"], 1.5)
    np.testing.assert_allclose(train_mags["node_encoding"]["32x0o"]["mean"], 3.5)
    np.testing.assert_allclose(train_mags["edge_encoding"]["16x1e"]["mean"], 5.5)

    # Verify validation stats
    val_mags = act_mags["val"]
    assert "node_encoding" in val_mags
    assert "32x0e" in val_mags["node_encoding"]
    np.testing.assert_allclose(val_mags["node_encoding"]["32x0e"]["mean"], 7.5)
    assert logged_metrics["bench_train_node_encoding_32x0e_mean"] == pytest.approx(1.5)
