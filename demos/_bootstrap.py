from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root_to_path() -> Path:
    """Make the repo importable when a demo is run as a plain script."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root
