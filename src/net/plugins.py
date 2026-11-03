"""Load transformation plugins: Python files that register transforms.

A recipe lists them under ``plugins``; paths are relative to the working
directory. Each file is imported once per process (by resolved path), so its
``register_transform`` calls run exactly once.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

_LOADED: dict[Path, object] = {}


def load_plugin(path):
    path = Path(path).expanduser().resolve()
    if path in _LOADED:
        return _LOADED[path]
    if not path.is_file():
        raise FileNotFoundError(f"Plugin file not found: {path}")
    name = (
        f"mandala_plugin_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _LOADED[path] = module
    return module


def load_plugins(paths):
    return [load_plugin(path) for path in paths or ()]
