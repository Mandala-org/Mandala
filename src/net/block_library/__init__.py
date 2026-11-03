"""Reusable composite blocks shipped with MANDALA.

Each YAML file of this package is a library of block definitions (see
``net.blocks``). A recipe imports libraries by name::

    pipeline:
      block_libraries: [deeph_e3]
      stages:
        - {id: layer0, transform: e3_node_update, ...}

Blocks the recipe defines itself take precedence over library blocks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

LIBRARY = Path(__file__).resolve().parent


def available_libraries():
    return sorted(path.stem for path in LIBRARY.glob("*.yaml"))


@lru_cache(None)
def _load(name):
    path = LIBRARY / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(
            f"Unknown block library {name!r}; available: {available_libraries()}"
        )
    blocks = yaml.safe_load(path.read_text()) or {}
    for block in blocks.values():
        block.setdefault("library", name)
    return blocks


def library_blocks(names):
    """``{block name: definition}`` of the named libraries (later ones win)."""
    import copy

    blocks = {}
    for name in names or ():
        blocks.update(copy.deepcopy(_load(name)))
    return blocks


def merged_blocks(specification):
    """The library blocks a pipeline imports, overridden by its own blocks."""
    return {
        **library_blocks(specification.get("block_libraries")),
        **dict(specification.get("blocks") or {}),
    }
