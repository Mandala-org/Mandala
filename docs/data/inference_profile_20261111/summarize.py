"""Print compact tables from profile_reference_inference.py result directories."""

import json
import sys
from pathlib import Path

ORDER = (
    "mandala",
    "deeph",
    "deeph_in_mandala",
    "deeph_e3",
    "deeph_e3_in_mandala",
    "mace_h",
    "mace_h_in_mandala",
)


def footer_gpu_ms(path):
    """True GPU time from the profiler table footer (the JSON field double counts)."""
    import re

    try:
        text = Path(path).read_text()
    except OSError:
        return float("nan")
    m = re.search(r"Self CUDA time total: ([0-9.]+)(us|ms|s)", text)
    if not m:
        return float("nan")
    return float(m[1]) * {"us": 1e-3, "ms": 1.0, "s": 1e3}[m[2]]


def load(root):
    out = {}
    for path in sorted(Path(root).glob("*/result.json")):
        d = json.loads(path.read_text())
        for phase in ("forward", "decode"):
            if phase in d.get("profiler", {}):
                prof = d["profiler"][phase]
                prof["gpu_kernel_ms"] = footer_gpu_ms(
                    path.parent / f"{phase}_by_cpu.txt"
                )
                # Relative to the uninstrumented median; profiling inflates wall time.
                base = d.get("baseline", {}).get(phase, {}).get("median_ms")
                prof["gpu_busy_fraction"] = (
                    prof["gpu_kernel_ms"] / base if base else float("nan")
                )
        out[(d["material"], d["variant"])] = d
    return out


def main():
    results = {}
    for root in sys.argv[1:]:
        results.update(load(root))
    for material in ("graphene", "mos2"):
        rows = [(v, results[(material, v)]) for v in ORDER if (material, v) in results]
        if not rows:
            continue
        print(
            f"\n## {material}  ({rows[0][1]['gpu']}, e3nn {rows[0][1]['e3nn_jit_mode']})"
        )
        print(
            f"{'variant':22s} {'fwd':>7s} {'e2e':>7s} | {'FP fwd':>7s} {'FP e2e':>7s} | {'TA fwd':>7s} {'TA e2e':>7s} | {'both f':>7s} {'both e':>7s} | "
            f"{'fwd GPU ms':>10s} {'busy':>5s} {'launch':>6s} {'D2H sync':>8s} | {'dec GPU':>7s} {'dec sync':>8s}"
        )
        for v, d in rows:
            b = d.get("baseline", {})
            vf = d.get("vectorized_from_packed", {})
            vt = d.get("vectorized_trace_alignment", {})
            vb = d.get("vectorized_both", {})
            pf = d.get("profiler", {}).get("forward", {})
            pd = d.get("profiler", {}).get("decode", {})

            def med(block, key):
                return f"{block[key]['median_ms']:7.1f}" if key in block else "      -"

            def syncs(p):
                s = p.get("sync_like_events", {})
                return s.get("aten::_local_scalar_dense", 0) + s.get(
                    "cudaStreamSynchronize", 0
                )

            print(
                f"{v:22s} {med(b, 'forward')} {med(b, 'end_to_end')} | {med(vf, 'forward')} {med(vf, 'end_to_end')} | {med(vt, 'forward')} {med(vt, 'end_to_end')} | {med(vb, 'forward')} {med(vb, 'end_to_end')} | "
                f"{pf.get('gpu_kernel_ms', float('nan')):10.1f} {pf.get('gpu_busy_fraction', float('nan')):5.2f} "
                f"{pf.get('kernel_launches', 0):6d} {syncs(pf):8d} | {pd.get('gpu_kernel_ms', float('nan')):7.1f} {syncs(pd):8d}"
            )
        for v, d in rows:
            print(
                f"  {v}: identical={ {k: all(x) if x else None for k, x in d.get('counterfactual_identical', {}).items()} } calls/e2e={d.get('counterfactual_calls_per_end_to_end')}"
            )
        for v, d in rows:
            stages = d.get("stages")
            if not stages:
                continue
            print(
                f"\n### {material} / {v}: stage timings, ms per structure (inclusive, synchronised)"
            )
            for phase in ("prep", "forward", "decode"):
                entries = stages.get(phase, {})
                for label, value in sorted(
                    entries.items(), key=lambda kv: -kv[1]["ms_per_structure"]
                ):
                    if value["ms_per_structure"] < 0.5 and label != "<phase total>":
                        continue
                    print(
                        f"  {phase:8s} {label:48s} {value['ms_per_structure']:9.1f}  x{value['calls_per_structure']:.0f}"
                    )


if __name__ == "__main__":
    main()
