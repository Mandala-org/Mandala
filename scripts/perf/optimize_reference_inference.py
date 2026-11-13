"""Measure inference optimisations for native and in-MANDALA reference models.

One variant per process. The variant is built exactly as in
``scripts/profile_reference_inference.py`` (same configs, structures, seed and
support). Optimisations are applied cumulatively in this order and each is
gated by an output check before it is timed:

``baseline``
    Unmodified code.
``host``
    Vectorised replacements for per-edge Python loops (``fast_paths.py``).
    Every replaced call made during one end-to-end pass is re-run through
    both implementations; the results must be bitwise identical.
``tf32``
    TF32 matmuls (numerical change; reported as relative error, not gated).
    Turned off again before the next stage.
``safe_scatter``
    Plain-torch replacement for ``torch_scatter.scatter`` (eager; checked to
    noise level). Required for compilation: torch 2.7.1 miscompiles
    torch_scatter's ``reduce="mean"``. Kept on for all compiled stages.
``compile_stages`` / ``compile``
    ``torch.compile`` of the network with static shapes: each pipeline
    transform separately (MANDALA only; whole model for native), or the
    whole network as one function (``pipeline.forward`` / upstream model).
``cudagraphs``
    ``torch.compile(mode="reduce-overhead")``: compiled kernels replayed
    through CUDA graphs. Needs static shapes; MoS2 structures differ in edge
    count, so one graph is recorded per shape.

Numerical stages report the maximum absolute difference of the decoded
Hamiltonian against the baseline output, next to the run-to-run noise of the
baseline itself (scatter atomics are not deterministic on GPU).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import e3nn
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_paths as F  # noqa: E402

import scripts.benchmark_reference_inference as B  # noqa: E402
import scripts.profile_reference_inference as P  # noqa: E402
import data.bond_geometry as BG  # noqa: E402
import data.edge_alignment as EA  # noqa: E402
import data.operator_features as OF  # noqa: E402
from data.block_matrix import IrrepsBlockData  # noqa: E402

STAGES = (
    "baseline",
    "host",
    "tf32",
    "safe_scatter",
    "compile_stages",
    "compile",
    "cudagraphs",
)


# ---------------------------------------------------------------------------
# Host-side patches: (label, owner, attribute, replacement factory, comparator)
# ---------------------------------------------------------------------------
def _same_blocks(a, b):
    if list(a.lookup.items()) != list(b.lookup.items()):
        return False
    if a.pair_blocks.keys() != b.pair_blocks.keys():
        return False
    for key in a.pair_blocks:
        if not torch.equal(a.pair_blocks[key], b.pair_blocks[key]):
            return False
        if not torch.equal(a.pair_edges[key], b.pair_edges[key]):
            return False
    ma, mb = getattr(a, "_coverage_masks", {}), getattr(b, "_coverage_masks", {})
    return ma.keys() == mb.keys() and all(torch.equal(ma[k], mb[k]) for k in ma)


def _same_dict(a, b):
    return a.keys() == b.keys() and all(
        torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a
    )


HOST_PATCHES = (
    (
        "from_packed",
        IrrepsBlockData,
        "from_packed",
        "classmethod",
        F.from_packed,
        P.same_container,
    ),
    (
        "trace_alignment",
        EA,
        "build_trace_alignment_from_pair_edges",
        "function",
        F.trace_alignment,
        P.same_alignment,
    ),
    ("to_blocks", IrrepsBlockData, "to_blocks", "method", F.to_blocks, _same_blocks),
    ("bond_frames", BG, "bond_frames", "function", F.bond_frames, torch.equal),
    (
        "bond_frames@operator_features",
        OF,
        "bond_frames",
        "function",
        F.bond_frames,
        torch.equal,
    ),
    (
        "bond_neighborhood_relations",
        BG,
        "bond_neighborhood_relations",
        "function",
        F.bond_neighborhood_relations,
        _same_dict,
    ),
)


def _original(owner, attribute):
    return (
        owner.__dict__[attribute]
        if isinstance(owner, type)
        else getattr(owner, attribute)
    )


def _callable(value):
    return value.__func__ if isinstance(value, (classmethod, staticmethod)) else value


def _install(owner, attribute, kind, function):
    setattr(
        owner, attribute, classmethod(function) if kind == "classmethod" else function
    )


def check_host_patches(variant, structures):
    """Capture every patched call in end-to-end passes, compare both versions."""
    captured = {label: [] for label, *_ in HOST_PATCHES}
    originals = {}
    for label, owner, attribute, kind, _fast, _cmp in HOST_PATCHES:
        original = _original(owner, attribute)
        originals[label] = original
        function = _callable(original)

        def capture(*args, _label=label, _function=function, **kwargs):
            captured[_label].append((args, kwargs))
            return _function(*args, **kwargs)

        _install(owner, attribute, kind, capture)
    try:
        with torch.no_grad():
            for i in range(structures):
                variant["end_to_end"](i)
    finally:
        for label, owner, attribute, _kind, _fast, _cmp in HOST_PATCHES:
            setattr(owner, attribute, originals[label])
    report = {}
    with torch.no_grad():
        for label, _owner, _attribute, _kind, fast, compare in HOST_PATCHES:
            original = _callable(originals[label])
            results = []
            # Captured classmethod calls already include cls as the first argument.
            for args, kwargs in captured[label]:
                results.append(
                    bool(compare(original(*args, **kwargs), fast(*args, **kwargs)))
                )
            report[label] = {
                "calls_per_end_to_end": len(results) / structures,
                "identical": all(results) if results else None,
            }
    return report, originals


@contextlib.contextmanager
def host_patches(enabled_labels):
    saved = []
    try:
        for label, owner, attribute, kind, fast, _cmp in HOST_PATCHES:
            if label in enabled_labels:
                saved.append((owner, attribute, _original(owner, attribute)))
                _install(owner, attribute, kind, fast)
        yield
    finally:
        for owner, attribute, value in reversed(saved):
            setattr(owner, attribute, value)


@contextlib.contextmanager
def safe_scatter():
    """Rebind every module-level torch_scatter.scatter alias (MANDALA and upstream)."""
    aliases = F.scatter_aliases()
    try:
        for module, attribute in aliases:
            setattr(module, attribute, F.compile_safe_scatter)
        yield [m.__name__ for m, _ in aliases]
    finally:
        for module, attribute in aliases:
            setattr(module, attribute, F._TORCH_SCATTER)


# ---------------------------------------------------------------------------
# Network compilation.
# ---------------------------------------------------------------------------
def network_modules(variant, granularity):
    """'whole': the full network as one function; 'stages': each MANDALA stage."""
    model = variant["model"]
    if variant["native"]:
        return [model]
    if granularity == "whole":
        return [model.pipeline]
    return list(model.pipeline.transforms.values())


@contextlib.contextmanager
def compiled_network(variant, mode, granularity="whole"):
    modules = network_modules(variant, granularity)
    saved = [(m, m.__dict__.get("forward")) for m in modules]
    try:
        for module in modules:
            compiled = torch.compile(
                module.forward, mode=mode, dynamic=False, fullgraph=False
            )
            if mode == "reduce-overhead":
                # Each call is a new CUDA-graph step, so a replay never
                # overwrites outputs of the previous call still in use.
                def forward(*args, _compiled=compiled, **kwargs):
                    torch.compiler.cudagraph_mark_step_begin()
                    return _compiled(*args, **kwargs)

                module.forward = forward
            else:
                module.forward = compiled
        yield
    finally:
        for module, previous in saved:
            if previous is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous
        torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Output comparison and timing.
# ---------------------------------------------------------------------------
def decoded_outputs(variant, structures):
    with torch.no_grad():
        return [variant["end_to_end"](i) for i in range(structures)]


def max_difference(reference, candidate):
    worst_abs, scale = 0.0, 0.0
    for ref, cand in zip(reference, candidate):
        if ref.pair_blocks.keys() != cand.pair_blocks.keys():
            return {"error": "different pair keys"}
        for key in ref.pair_blocks:
            a, b = ref.pair_blocks[key], cand.pair_blocks[key]
            if a.shape != b.shape:
                return {"error": f"shape mismatch for {key}"}
            worst_abs = max(worst_abs, float((a - b).abs().max()))
            scale = max(scale, float(a.abs().max()))
    return {
        "max_abs": worst_abs,
        "max_abs_relative_to_max": worst_abs / scale if scale else None,
    }


def time_variant(variant, cases, structures, repeats):
    with torch.no_grad():
        start = time.perf_counter()
        forward = B.timing(variant["forward"], cases, 5, repeats)
        end_to_end = B.timing(
            variant["end_to_end"], list(range(structures)), 2, repeats
        )
        total = time.perf_counter() - start
    for block in (forward, end_to_end):
        block["samples_ms"] = [round(v, 3) for v in block["samples_ms"]]
    for block in (forward, end_to_end):
        block["min_ms"] = min(block["samples_ms"])
    return {
        "forward": forward,
        "end_to_end": end_to_end,
        "stage_wall_seconds": total,
        "loadavg": os.getloadavg(),
    }


def first_call_seconds(variant, cases):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for case in cases:
            variant["forward"](case)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--material", choices=["graphene", "mos2"], required=True)
    p.add_argument("--variant", choices=P.VARIANTS, required=True)
    p.add_argument("--jit-mode", choices=["eager", "script"], default="eager")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Needs a CUDA device")

    e3nn.set_optimization_defaults(jit_mode=args.jit_mode)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    paths = list(
        OmegaConf.load(
            args.root / "configs" / f"mandala_{args.material}.yaml"
        ).data.splits.val
    )[:3]
    result = dict(
        material=args.material,
        variant=args.variant,
        jit_mode=args.jit_mode,
        torch=str(torch.__version__),
        e3nn=str(e3nn.__version__),
        gpu=torch.cuda.get_device_name(),
        host=os.uname().nodename,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        stages={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)

    start = time.perf_counter()
    variant = P.build_variant(args.root, args.material, args.variant, paths)
    result["build_seconds"] = time.perf_counter() - start
    structures = len(paths)
    with torch.no_grad():
        cases = [variant["prep"](i) for i in range(structures)]
    result["first_forward_seconds"] = first_call_seconds(variant, cases)
    print("BUILT", args.variant, f"{result['build_seconds']:.1f}s", flush=True)

    reference = decoded_outputs(variant, structures)
    result["baseline_noise"] = max_difference(
        reference, decoded_outputs(variant, structures)
    )

    def stage(name, fn):
        if name not in args.stages:
            return
        try:
            value = fn()
        except Exception as error:  # record and continue with the next stage
            value = {"error": repr(error), "traceback": traceback.format_exc()[-4000:]}
        result["stages"][name] = value
        timing = value.get("timing", {}) if isinstance(value, dict) else {}
        summary = {
            k: round(v["median_ms"], 1)
            for k, v in timing.items()
            if isinstance(v, dict) and "median_ms" in v
        }
        print("STAGE", name, summary or value.get("error", ""), flush=True)
        save()

    stage(
        "baseline",
        lambda: {"timing": time_variant(variant, cases, structures, args.repeats)},
    )

    host_labels = set()

    def host():
        report, _ = check_host_patches(variant, structures)
        host_labels.update(k for k, v in report.items() if v["identical"])
        with host_patches(host_labels):
            with torch.no_grad():
                patched_cases = [variant["prep"](i) for i in range(structures)]
            diff = max_difference(reference, decoded_outputs(variant, structures))
            timing = time_variant(variant, patched_cases, structures, args.repeats)
        return {
            "patches": report,
            "applied": sorted(host_labels),
            "output_difference": diff,
            "timing": timing,
        }

    stage("host", host)

    def tf32():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            with host_patches(host_labels):
                diff = max_difference(reference, decoded_outputs(variant, structures))
                timing = time_variant(variant, cases, structures, args.repeats)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return {"output_difference": diff, "timing": timing}

    stage("tf32", tf32)

    def scatter_stage():
        with host_patches(host_labels), safe_scatter() as modules:
            diff = max_difference(reference, decoded_outputs(variant, structures))
            timing = time_variant(variant, cases, structures, args.repeats)
        return {"patched_modules": modules, "output_difference": diff, "timing": timing}

    stage("safe_scatter", scatter_stage)

    def compiled(mode, granularity):
        def run():
            # torch_scatter's reduce="mean" is miscompiled by torch 2.7.1, so
            # every compiled stage uses the compile-safe scatter.
            with (
                host_patches(host_labels),
                safe_scatter(),
                compiled_network(variant, mode, granularity),
            ):
                compile_seconds = first_call_seconds(variant, cases)
                # A second pass separates compilation from CUDA-graph recording.
                second_seconds = first_call_seconds(variant, cases)
                diff = max_difference(reference, decoded_outputs(variant, structures))
                timing = time_variant(variant, cases, structures, args.repeats)
            return {
                "mode": mode,
                "granularity": granularity,
                "first_pass_seconds": compile_seconds,
                "second_pass_seconds": second_seconds,
                "output_difference": diff,
                "timing": timing,
            }

        return run

    stage("compile_stages", compiled("default", "stages"))
    stage("compile", compiled("default", "whole"))
    stage("cudagraphs", compiled("reduce-overhead", "whole"))
    print("DONE", args.material, args.variant, flush=True)


if __name__ == "__main__":
    main()
