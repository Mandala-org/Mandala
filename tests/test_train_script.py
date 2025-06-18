import os
import sys
import importlib


def test_train_script_setup(tmp_path, monkeypatch):
    """
    Test that scripts/train.py can set up the full training pipeline without errors
    by mocking out the Trainer.fit call and WandBLogger.
    """
    # Create minimal config YAML
    cfg_path = tmp_path / "train_smoke.yaml"
    cfg_path.write_text(
        """
run_name: test-smoke
snapshots:
  - matrix: data/small/H2O/original/H2O.matrix
    info:   data/small/H2O/original/H2O.info.out
    purpose: train
  - matrix: data/small/H2O/original/H2O.matrix
    info:   data/small/H2O/original/H2O.info.out
    purpose: val
cutoff_gnn: 4.0
cutoff_matrix: 6.0
l_max_sh: 2
n_radial: 16
num_workers: 0
model:
  num_layers_gnn: 1
  num_layers_matrix: 1
  hidden_base_dim: 16
  head_depth: 1
lr: 0.001
max_epochs: 1
precision: 32
accum: 1
"""
    )
    # Import train script as module
    spec = importlib.util.spec_from_file_location(
        "train_script", os.path.join(os.getcwd(), "scripts", "train.py")
    )
    train_script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train_script
    spec.loader.exec_module(train_script)

    # Monkey-patch WandbLogger to a dummy logger
    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.experiment = type(
                "E",
                (),
                {
                    "config": type(
                        "C", (), {"update": staticmethod(lambda cfg, **kw: None)}
                    )()
                },
            )

        def log_metrics(self, *args, **kwargs):
            pass

    monkeypatch.setattr(train_script, "WandbLogger", DummyLogger)
    # Monkey-patch Trainer.fit to no-op
    monkeypatch.setattr(
        train_script.pl.Trainer, "fit", lambda self, *args, **kwargs: None
    )
    # Set args and environment
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path)])
    # Run main without raising
    train_script.main()
