"""Summarize base-versus-modular timings written by ``forward_timing.py``.

Pairs ``<case>_base.json`` with ``<case>_modular.json`` (and, when present,
``<case>_modular_nocache.json``) in a directory and writes a Markdown table of
medians and ratios (uncached modular / base, or modular / base).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def milliseconds(result, key):
    timing = result[key]
    if timing.get("out_of_memory"):
        return "OOM"
    return timing["median_s"] * 1e3


def text(value):
    if value is None:
        return "-"
    return value if isinstance(value, str) else f"{value:.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [
        "| case | parameters (base / modular) | edges | forward ms (base / modular / uncached / ratio) "
        "| train step ms (base / modular / uncached / ratio) | peak memory MB (base / modular) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for base_path in sorted(args.directory.glob("*_base.json")):
        case = base_path.name.removesuffix("_base.json")
        modular_path = args.directory / f"{case}_modular.json"
        if not modular_path.exists():
            continue
        base, modular = json.loads(base_path.read_text()), json.loads(
            modular_path.read_text()
        )
        uncached_path = args.directory / f"{case}_modular_nocache.json"
        uncached = (
            json.loads(uncached_path.read_text()) if uncached_path.exists() else None
        )
        same = base["parameters"] == modular["parameters"]
        flag = "" if same else " **(differ)**"
        cells = [
            case,
            f"{base['parameters']} / {modular['parameters']}{flag}",
            str(modular["edges"]),
        ]
        # The ratio uses the uncached arm when present: like the base tree it
        # recomputes structure-only stages, isolating the decomposition's cost.
        for key in ("forward", "train_step"):
            b, m = milliseconds(base, key), milliseconds(modular, key)
            u = milliseconds(uncached, key) if uncached else None
            reference = u if uncached else m
            ratio = (
                f"{reference / b:.3f}"
                if isinstance(reference, float) and isinstance(b, float)
                else "-"
            )
            cells.append(f"{text(b)} / {text(m)} / {text(u)} / {ratio}")
        memory = [result.get("peak_memory_bytes") for result in (base, modular)]
        cells.append(
            " / ".join(f"{value / 2**20:.0f}" if value else "-" for value in memory)
        )
        rows.append("| " + " | ".join(cells) + " |")
    args.output.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
