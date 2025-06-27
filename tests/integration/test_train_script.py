import pytest
import os
import sys
import importlib
from hydra.core.global_hydra import GlobalHydra


@pytest.mark.integration
@pytest.mark.integration
def test_train_script_setup(monkeypatch):
    """
    Test that scripts/train.py can set up the full training pipeline without errors
    by mocking out the Trainer.fit call and WandBLogger.
    """
    # Get the absolute path to the config directory
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))

    # Dynamically rewrite the hydra.main decorator to use an absolute path
    with open("scripts/train.py", "r") as f:
        train_script_code = f.read()
    train_script_code = train_script_code.replace(
        '@hydra.main(config_path="../conf", version_base="1.1")',
        f'@hydra.main(config_path="{config_path}", version_base="1.1")',
    )
    exec(train_script_code, globals())

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
    # Use grouped debug_cpu config
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config-name", "debug_cpu"],
    )
    # Run main without raising, but with the correct config path
    GlobalHydra.instance().clear()
    main()  # noqa: F821
