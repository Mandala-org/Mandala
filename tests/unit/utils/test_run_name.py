from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("src/utils/run_name.py").resolve()
    spec = importlib.util.spec_from_file_location("utils_run_name", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_run_name_ignores_placeholder_wandb_name():
    mod = _load_module()

    class DummyLogger:
        def __init__(self):
            self.experiment = type(
                "Experiment",
                (),
                {"name": "mandala-run", "id": "vibran-sweep-14"},
            )()

    resolved = mod.resolve_run_name(None, DummyLogger())

    assert resolved == "vibran-sweep-14"
