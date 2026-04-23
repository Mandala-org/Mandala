from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/train.py").resolve()
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_populate_config_from_args_normalizes_string_list_fields():
    mod = _load_module()
    args = mod.argparse.Namespace(
        matrix_targets="density",
        radial_layers="[128, 64]",
    )

    cfg = mod._populate_config_from_args(args)

    assert cfg.matrix_targets == ["density"]
    assert cfg.radial_layers == [128, 64]
