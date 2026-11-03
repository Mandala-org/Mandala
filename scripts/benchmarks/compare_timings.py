"""Summarize base-versus-modular timings written by ``forward_timing.py``.

Pairs ``<case>_base.json`` with ``<case>_modular.json`` (and, when present,
``<case>_modular_nocache.json``) in a directory and writes a Markdown table of
medians and ratios (uncached modular / base, or modular / base).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
            b, m = base[key]["median_s"] * 1e3, modular[key]["median_s"] * 1e3
            u = uncached[key]["median_s"] * 1e3 if uncached else None
            ratio = (u if u is not None else m) / b
            cells.append(
                f"{b:.2f} / {m:.2f} / {'-' if u is None else f'{u:.2f}'} / {ratio:.3f}"
            )
        memory = [result.get("peak_memory_bytes") for result in (base, modular)]
        cells.append(
            " / ".join(f"{value / 2**20:.0f}" if value else "-" for value in memory)
        )
        rows.append("| " + " | ".join(cells) + " |")
    args.output.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
