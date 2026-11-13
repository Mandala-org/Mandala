"""Tabulate optimize_reference_inference.py results.

Usage: python summarize_optimization.py RESULT_DIR [RESULT_DIR ...]
Prints, per material and e3nn mode, end-to-end and forward medians (and
minimums) by stage, then output differences and compile times.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ORDER = (
    "deeph",
    "deeph_in_mandala",
    "deeph_e3",
    "deeph_e3_in_mandala",
    "mace_h",
    "mace_h_in_mandala",
    "mandala",
)
STAGES = (
    "baseline",
    "host",
    "tf32",
    "safe_scatter",
    "compile_stages",
    "compile",
    "cudagraphs",
)


def load(roots):
    results = {}
    for root in roots:
        for path in sorted(Path(root).glob("*.json")):
            d = json.loads(path.read_text())
            if "stages" in d:
                results[(d["material"], d["jit_mode"], d["variant"])] = d
    return results


def cell(stage, key, stat):
    if not stage or "timing" not in stage:
        return "err" if stage and "error" in stage else "-"
    block = stage["timing"][key]
    value = block.get(stat)
    if value is None and stat == "min_ms":
        value = min(block["samples_ms"])
    return f"{value:.0f}"


def main():
    results = load(sys.argv[1:])
    groups = sorted({(m, j) for m, j, _ in results})
    for material, jit in groups:
        rows = [v for v in ORDER if (material, jit, v) in results]
        for key in ("end_to_end", "forward"):
            for stat in ("median_ms", "min_ms"):
                print(f"\n## {material}, e3nn {jit}: {key} {stat.split('_')[0]} (ms)")
                print("| variant | " + " | ".join(STAGES) + " |")
                print("|---|" + "---:|" * len(STAGES))
                for v in rows:
                    st = results[(material, jit, v)]["stages"]
                    print(
                        f"| {v} | "
                        + " | ".join(cell(st.get(s), key, stat) for s in STAGES)
                        + " |"
                    )
        print(
            f"\n## {material}, e3nn {jit}: max |dH| / max |H| vs baseline (noise first)"
        )
        print(
            "| variant | noise | "
            + " | ".join(STAGES[1:])
            + " | compile s (first pass, per stage) |"
        )
        print("|---|" + "---:|" * (len(STAGES) + 1) + "---|")
        for v in rows:
            d = results[(material, jit, v)]
            st = d["stages"]

            def rel(s):
                diff = (st.get(s) or {}).get("output_difference") or {}
                value = diff.get("max_abs_relative_to_max")
                return f"{value:.1e}" if value is not None else "-"

            compile_s = ", ".join(
                f"{s}={st[s]['first_pass_seconds']:.0f}"
                for s in ("compile_stages", "compile", "cudagraphs")
                if s in st and "first_pass_seconds" in st[s]
            )
            noise = d["baseline_noise"].get("max_abs_relative_to_max")
            print(
                f"| {v} | {noise:.1e} | "
                + " | ".join(rel(s) for s in STAGES[1:])
                + f" | {compile_s} |"
            )
        for v in rows:
            host = results[(material, jit, v)]["stages"].get("host", {})
            patches = host.get("patches", {})
            used = {
                k: p["identical"]
                for k, p in patches.items()
                if p["identical"] is not None
            }
            errors = {
                s: x["error"][:120]
                for s, x in results[(material, jit, v)]["stages"].items()
                if "error" in x
            }
            print(
                f"- {v}: host patches identical {used}"
                + (f"; errors {errors}" if errors else "")
            )


if __name__ == "__main__":
    main()
