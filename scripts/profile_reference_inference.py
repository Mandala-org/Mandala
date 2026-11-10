"""Decompose native and in-MANDALA inference latency into measured stages.

Companion to ``scripts/benchmark_reference_inference.py``. It rebuilds one
variant exactly as that benchmark does (same configs, structures, seed, dtype,
e3nn backend and stored support), then measures:

1. ``baseline``: the benchmark's own forward and end-to-end medians, with no
   instrumentation active, to show the run reproduces the published numbers.
2. ``stages``: end-to-end passes split into prep / forward / decode phases, with
   CUDA-synchronised inclusive timers around framework functions and around
   every pipeline stage (in-MANDALA) or top-level upstream submodule (native).
3. ``profiler``: one torch.profiler trace of forward and of decode, counting
   kernel launches, device-to-host synchronisations and total GPU kernel time.
4. ``vectorized_from_packed``: the same baseline timings after replacing
   ``IrrepsBlockData.from_packed`` with a version that synchronises once per
   call instead of once per edge. Its output is checked to be identical first.

Run one variant per process, on a Slurm GPU allocation, with the benchmark's
framework on PYTHONPATH.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import functools
import gc
import json
import os
from pathlib import Path
import time

import e3nn
from omegaconf import OmegaConf
import torch

import scripts.benchmark_reference_inference as B
from core.basis_converter import OpenMXE3NNConverter
import data.block_matrix as BM
import data.edge_alignment as EA
import data.graph_features as GF
import data.graph_topology as GT
from data.block_matrix import IrrepsBlockData
from data.operator_features import build_operator_sample
from data.sample import move_tree
from scripts.train import build_components

VARIANTS = (
    "mandala",
    "deeph",
    "deeph_in_mandala",
    "deeph_e3",
    "deeph_e3_in_mandala",
    "mace_h",
    "mace_h_in_mandala",
)


# --------------------------------------------------------------------------
# Variant construction: mirrors benchmark_reference_inference.benchmark().
# --------------------------------------------------------------------------
def build_variant(root, material, name, paths):
    kind = name.removesuffix("_in_mandala")
    native = name != "mandala" and not name.endswith("_in_mandala")
    config = OmegaConf.load(root / "configs" / f"{kind}_{material}.yaml")
    config.data.splits = {
        "train": list(config.data.splits.train)[:1],
        "val": paths,
        "test": [],
    }
    config.snapshot_cache_dir = None
    config.symmetrize_hamiltonian_targets = False
    config.symmetrize_validation_output = False
    torch.manual_seed(42)
    resolved, layout, model, datasets, _ = build_components(config, resume=True)
    dataset = datasets["val"]
    snapshots = [dataset._load_snapshot(*p) for p in dataset.snapshot_paths]
    expected = [set(s.hamiltonian.lookup) for s in snapshots]
    if layout is None:
        for s in snapshots:
            s.hamiltonian = s.hamiltonian * 0.0
            s.overlap = s.overlap * 0.0
            s.density = s.density * 0.0
    mapper = model.mapper
    codec = None
    if native:
        del model
        gc.collect()
        model, codec = B.upstream_model(root / "upstream", kind, resolved, layout)
        layout.to("cuda")
        mapper.to("cuda")
    model.to("cuda").eval()

    def prep(i):
        s = snapshots[i]
        if layout is not None:
            meta = dict(
                atoms=s.hamiltonian.atoms,
                positions=s.positions.float(),
                box=s.box.float(),
                spinful=False,
                species_shells=layout.species_shells,
            )
            x = build_operator_sample(
                meta,
                {},
                expected[i],
                layout,
                species=list(layout.species_shells),
                target_frame="bond" if kind == "deeph" else "global",
                local_coordinates=kind == "deeph",
                num_l=resolved["parameters"].get("num_l", 5),
                frame_meta=dict(positions=s.positions.double(), box=s.box.double()),
                supports=resolved["data"].get("supports"),
            )["inputs"]
        else:
            matrix, info = dataset.snapshot_paths[i]
            x, _ = dataset._process_snapshot_to_sample(
                s, matrix_path=matrix, info_path=info
            )
        x = move_tree(x, torch.device("cuda"), torch.float32)
        return x, B.native_input(x, kind) if native else x

    def forward(case):
        _, inp = case
        if native:
            return model(*inp) if kind == "deeph" else model(inp)
        return model(inp)

    def decode(case, value):
        x, _ = case
        if native:
            packed = value if kind == "deeph" else codec.get_H(value[1])
            matrix = IrrepsBlockData.from_packed(
                packed,
                x,
                mapper,
                layout,
                frame="bond" if kind == "deeph" else "global",
                operator="hamiltonian",
            ).to_blocks(mapper)
        else:
            matrix = model.predicted_irreps_to_block_matrices(value, x)["hamiltonian"]
        if matrix.basis == "e3nn":
            matrix = OpenMXE3NNConverter(matrix.orbital_cfg).matrix_to_openmx(matrix)
        if matrix.basis != "openmx":
            raise ValueError("Output is outside common OpenMX basis")
        return matrix

    def end_to_end(i):
        case = prep(i)
        return decode(case, forward(case)).to("cpu")

    return dict(
        kind=kind,
        native=native,
        model=model,
        codec=codec,
        layout=layout,
        mapper=mapper,
        expected=expected,
        prep=prep,
        forward=forward,
        decode=decode,
        end_to_end=end_to_end,
    )


# --------------------------------------------------------------------------
# Synchronised inclusive timers, keyed by (phase, label).
# --------------------------------------------------------------------------
class Timers:
    def __init__(self):
        self.active = False
        self.phase = None
        self.seconds = defaultdict(float)
        self.calls = Counter()

    @contextlib.contextmanager
    def in_phase(self, phase):
        previous, self.phase = self.phase, phase
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds[(phase, "<phase total>")] += time.perf_counter() - start
            self.calls[(phase, "<phase total>")] += 1
            self.phase = previous

    def record(self, label, seconds):
        key = (self.phase or "<none>", label)
        self.seconds[key] += seconds
        self.calls[key] += 1

    def wrap(self, label, fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if not self.active:
                return fn(*args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                self.record(label, time.perf_counter() - start)

        return inner

    def hook_module(self, label, module):
        starts = []

        def pre(_module, _inputs):
            if self.active:
                torch.cuda.synchronize()
                starts.append(time.perf_counter())

        def post(_module, _inputs, _output):
            if self.active and starts:
                torch.cuda.synchronize()
                self.record(label, time.perf_counter() - starts.pop())

        return [
            module.register_forward_pre_hook(pre),
            module.register_forward_hook(post),
        ]

    def summary(self, structures):
        table = {}
        for (phase, label), seconds in sorted(self.seconds.items()):
            table.setdefault(phase, {})[label] = {
                "ms_per_structure": 1000 * seconds / structures,
                "calls_per_structure": self.calls[(phase, label)] / structures,
            }
        return table


def install_timers(timers, variant):
    """Patch module attributes looked up at call time; returns an undo list."""
    undo = []

    def patch(owner, attribute, label, kind="function"):
        original = (
            owner.__dict__[attribute]
            if isinstance(owner, type)
            else getattr(owner, attribute)
        )
        if kind == "classmethod":
            replacement = classmethod(timers.wrap(label, original.__func__))
        elif kind == "staticmethod":
            replacement = staticmethod(timers.wrap(label, original.__func__))
        else:
            replacement = timers.wrap(label, original)
        setattr(owner, attribute, replacement)
        undo.append(lambda: setattr(owner, attribute, original))

    patch(GT, "prepare_graph_topology", "prepare_graph_topology")
    patch(GF, "complete_graph_metadata", "complete_graph_metadata")
    patch(GF, "evaluate_graph_geometry", "evaluate_graph_geometry")
    patch(
        EA,
        "build_trace_alignment_from_pair_edges",
        "build_trace_alignment_from_pair_edges",
    )
    patch(EA, "build_prediction_edge_metadata", "build_prediction_edge_metadata")
    patch(IrrepsBlockData, "from_packed", "IrrepsBlockData.from_packed", "classmethod")
    patch(IrrepsBlockData, "to_blocks", "IrrepsBlockData.to_blocks")
    patch(IrrepsBlockData, "pack_matrix", "IrrepsBlockData.pack_matrix", "staticmethod")
    patch(BM.BlockMatrix, "to", "BlockMatrix.to")
    patch(
        OpenMXE3NNConverter, "matrix_to_openmx", "OpenMXE3NNConverter.matrix_to_openmx"
    )
    model = variant["model"]
    hooks = []
    if variant["native"]:
        patch(model, "forward", "native model.forward")
        if variant["codec"] is not None:
            patch(variant["codec"], "get_H", "native codec.get_H")
        for child_name, child in model.named_children():
            hooks += timers.hook_module(f"native child:{child_name}", child)
    else:
        patch(model, "forward", "E3GNN.forward")
        patch(model, "_populate_edge_features", "E3GNN._populate_edge_features")
        patch(model, "_wrap_head_output", "E3GNN._wrap_head_output")
        patch(
            model,
            "predicted_irreps_to_block_matrices",
            "E3GNN.predicted_irreps_to_block_matrices",
        )
        patch(model.pipeline, "forward", "pipeline.forward")
        for sid, module in model.pipeline.transforms.items():
            hooks += timers.hook_module(f"stage:{sid}", module)
    undo += [handle.remove for handle in hooks]
    return undo


# --------------------------------------------------------------------------
# Vectorised from_packed: one device-to-host copy per call instead of per edge.
# --------------------------------------------------------------------------
def vectorized_from_packed(
    cls, values, x, mapper, layout, *, frame="global", operator=None
):
    species = tuple(layout.species_shells)
    types = x["node_type_idx"]
    atoms = tuple(species[int(i)] for i in types.tolist())
    edges = x.get("edge_keys")
    if edges is None:
        shifts = x.get(
            "edge_shift", x["edge_index"].new_zeros((3, x["edge_index"].shape[1]))
        )
        edges = torch.cat((shifts, x["edge_index"]), dim=0)
    count = len(species)
    types = types.to(edges.device).long()
    codes = (types[edges[3]] * count + types[edges[4]]).tolist()
    edge_rows = edges.T.tolist()
    buckets = defaultdict(list)
    for index, code in enumerate(codes):
        buckets[code].append(index)
    pair_vectors, pair_edges, lookup = {}, {}, {}
    for li, left in enumerate(species):
        for ri, right in enumerate(species):
            indices = buckets.get(li * count + ri)
            if not indices:
                continue
            selected = torch.tensor(indices, device=values.device, dtype=torch.long)
            pair = f"{left}-{right}"
            pair_vectors[pair] = layout.encode(values.index_select(0, selected))
            pair_edges[pair] = edges.index_select(1, selected.to(edges.device))
            for local, global_index in enumerate(indices):
                lookup[tuple(edge_rows[global_index])] = (pair, local)
    frames = x.get("bond_frames") if frame == "bond" else None
    if frame == "bond" and frames is None:
        raise ValueError("Bond-frame output requires explicit frames")
    return cls(
        atoms,
        Counter(atoms),
        pair_vectors,
        pair_edges,
        lookup,
        mapper.orbital_cfg,
        "openmx",
        layout,
        values,
        edges,
        frames,
        operator,
    )


def vectorized_trace_alignment(pair_edges_a, pair_edges_b=None, *, device=None):
    """Same mapping as core.sparse_math.build_trace_alignment_from_pair_edges,
    with one device-to-host copy per edge set instead of one per edge."""
    if pair_edges_b is None:
        pair_edges_b = pair_edges_a
    alignment = {}
    host_b = {}
    for key, edges_a in pair_edges_a.items():
        el_a, el_b = key.split("-")
        rev_key = f"{el_b}-{el_a}"
        if rev_key not in pair_edges_b:
            raise ValueError(
                f"Key {rev_key} missing in second edge set (reverse of {key})."
            )
        if rev_key not in host_b:
            host_b[rev_key] = {
                tuple(edge): idx
                for idx, edge in enumerate(pair_edges_b[rev_key].t().tolist())
            }
        mapping = host_b[rev_key]
        indices_b = []
        for sx, sy, sz, i, j in edges_a.t().tolist():
            rev_edge = (-sx, -sy, -sz, j, i)
            if rev_edge not in mapping:
                raise ValueError(
                    f"Edge {rev_edge} (reverse of {(sx, sy, sz, i, j)}) missing in second edge set."
                )
            indices_b.append(mapping[rev_edge])
        idx_device = device if device is not None else edges_a.device
        alignment[key] = (
            rev_key,
            torch.tensor(indices_b, device=idx_device, dtype=torch.long),
        )
    return alignment


def same_alignment(a, b):
    return a.keys() == b.keys() and all(
        a[k][0] == b[k][0] and torch.equal(a[k][1], b[k][1]) for k in a
    )


def same_container(a, b):
    if a.atoms != b.atoms or list(a.lookup.items()) != list(b.lookup.items()):
        return False
    if a.pair_vectors.keys() != b.pair_vectors.keys():
        return False
    return all(
        torch.equal(a.pair_vectors[k], b.pair_vectors[k])
        and torch.equal(a.pair_edges[k], b.pair_edges[k])
        for k in a.pair_vectors
    )


# --------------------------------------------------------------------------
# Profiler summary.
# --------------------------------------------------------------------------
SYNC_EVENTS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpyAsync",
    "cudaMemcpy",
    "aten::_local_scalar_dense",
    "aten::item",
)
LAUNCH_EVENTS = (
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernelEx",
)


def device_time(event):
    for attribute in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(event, attribute):
            return getattr(event, attribute)
    return 0.0


def profile(fn, label, out_dir):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.profiler.profile(activities=activities) as prof:
        result = fn()
        torch.cuda.synchronize()
    wall_ms = 1000 * (time.perf_counter() - start)
    del result
    averages = prof.key_averages()
    counts = {e.key: e.count for e in averages}
    gpu_us = sum(device_time(e) for e in averages)
    table_cpu = averages.table(sort_by="self_cpu_time_total", row_limit=30)
    sort_device = (
        "self_device_time_total"
        if hasattr(averages[0], "self_device_time_total")
        else "self_cuda_time_total"
    )
    table_gpu = averages.table(sort_by=sort_device, row_limit=30)
    (out_dir / f"{label}_by_cpu.txt").write_text(table_cpu)
    (out_dir / f"{label}_by_gpu.txt").write_text(table_gpu)
    return {
        "profiled_wall_ms": wall_ms,
        "gpu_kernel_ms": gpu_us / 1000,
        "gpu_busy_fraction": (gpu_us / 1000) / wall_ms if wall_ms else None,
        "kernel_launches": sum(counts.get(k, 0) for k in LAUNCH_EVENTS),
        "sync_like_events": {k: counts.get(k, 0) for k in SYNC_EVENTS},
        "aten_ops": sum(c for k, c in counts.items() if k.startswith("aten::")),
    }


def describe_inputs(variant, case):
    x, inp = case
    shapes = {}
    for key in ("edge_index", "message_edge_index", "edge_keys"):
        value = x.get(key) if isinstance(x, dict) else None
        if torch.is_tensor(value):
            shapes[key] = list(value.shape)
    if variant["native"] and variant["kind"] != "deeph":
        shapes["native edge_index"] = list(inp.edge_index.shape)
    return shapes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="benchmark run root (configs/, upstream/)",
    )
    p.add_argument("--material", choices=["graphene", "mos2"], required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--stage-repeats", type=int, default=3)
    p.add_argument("--e3nn-jit-mode", choices=["script", "eager"], default="eager")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Needs a CUDA device")

    e3nn.set_optimization_defaults(jit_mode=args.e3nn_jit_mode)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out_dir = args.output_dir / f"{args.material}_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list(
        OmegaConf.load(
            args.root / "configs" / f"mandala_{args.material}.yaml"
        ).data.splits.val
    )[:3]
    result = dict(
        material=args.material,
        variant=args.variant,
        structures=paths,
        torch=str(torch.__version__),
        e3nn=str(e3nn.__version__),
        e3nn_jit_mode=args.e3nn_jit_mode,
        gpu=torch.cuda.get_device_name(),
        node=os.environ.get("SLURMD_NODENAME"),
        job_id=os.environ.get("SLURM_JOB_ID"),
    )

    def save():
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )

    start = time.perf_counter()
    variant = build_variant(args.root, args.material, args.variant, paths)
    result["build_seconds"] = time.perf_counter() - start
    result["parameters"] = sum(p.numel() for p in variant["model"].parameters())
    print("BUILT", args.variant, f"{result['build_seconds']:.1f}s", flush=True)

    with torch.no_grad():
        cases = [variant["prep"](i) for i in range(len(paths))]
        result["input_shapes"] = describe_inputs(variant, cases[0])
        # Support and finite checks, as in the benchmark; also first-call warmup.
        for i, case in enumerate(cases):
            matrix = variant["decode"](case, variant["forward"](case))
            if set(matrix.lookup) != variant["expected"][i]:
                raise ValueError("Output support mismatch")
        torch.cuda.synchronize()

        # 1. Baseline, uninstrumented, identical timing routine.
        result["baseline"] = {
            "forward": B.timing(variant["forward"], cases, 5, args.repeats),
            "end_to_end": B.timing(
                variant["end_to_end"], list(range(len(paths))), 2, args.repeats
            ),
        }
        for block in result["baseline"].values():
            block.pop("samples_ms", None)
        print(
            "BASELINE",
            {k: round(v["median_ms"], 1) for k, v in result["baseline"].items()},
            flush=True,
        )
        save()

        # 2. Stage decomposition.
        timers = Timers()
        undo = install_timers(timers, variant)
        timers.active = True
        for _ in range(args.stage_repeats):
            for i in range(len(paths)):
                with timers.in_phase("prep"):
                    case = variant["prep"](i)
                with timers.in_phase("forward"):
                    value = variant["forward"](case)
                with timers.in_phase("decode"):
                    variant["decode"](case, value).to("cpu")
                del case, value
        timers.active = False
        for function in reversed(undo):
            function()
        result["stages"] = timers.summary(args.stage_repeats * len(paths))
        print("STAGES done", flush=True)
        save()

        # 3. Profiler traces of one structure.
        case = cases[0]
        value = variant["forward"](case)
        result["profiler"] = {
            "forward": profile(lambda: variant["forward"](case), "forward", out_dir),
            "decode": profile(
                lambda: variant["decode"](case, value), "decode", out_dir
            ),
        }
        del value
        print("PROFILER done", flush=True)
        save()

        # 4. Counterfactuals: vectorised replacements, each first checked to give
        # identical results on the exact arguments captured from real passes
        # (separate forwards are not bitwise reproducible on GPU).
        original_fp = IrrepsBlockData.__dict__["from_packed"]
        original_ta = EA.build_trace_alignment_from_pair_edges
        captured = {"from_packed": [], "trace_alignment": []}

        def capture_fp(cls, *a, **k):
            captured["from_packed"].append((a, k))
            return original_fp.__func__(cls, *a, **k)

        def capture_ta(*a, **k):
            captured["trace_alignment"].append((a, k))
            return original_ta(*a, **k)

        IrrepsBlockData.from_packed = classmethod(capture_fp)
        EA.build_trace_alignment_from_pair_edges = capture_ta
        try:
            for i in range(len(paths)):
                variant["end_to_end"](i)
        finally:
            IrrepsBlockData.from_packed = original_fp
            EA.build_trace_alignment_from_pair_edges = original_ta
        identical = {
            "from_packed": [
                same_container(
                    original_fp.__func__(IrrepsBlockData, *a, **k),
                    vectorized_from_packed(IrrepsBlockData, *a, **k),
                )
                for a, k in captured["from_packed"]
            ],
            "trace_alignment": [
                same_alignment(
                    original_ta(*a, **k), vectorized_trace_alignment(*a, **k)
                )
                for a, k in captured["trace_alignment"]
            ],
        }
        result["counterfactual_identical"] = identical
        result["counterfactual_calls_per_end_to_end"] = {
            k: len(v) / len(paths) for k, v in captured.items()
        }
        captured.clear()
        replacements = {
            "vectorized_from_packed": ("from_packed",),
            "vectorized_trace_alignment": ("trace_alignment",),
            "vectorized_both": ("from_packed", "trace_alignment"),
        }
        for label, parts in replacements.items():
            if not all(all(identical[p]) for p in parts):
                result[label] = {"skipped": "replacement not identical"}
                continue
            if all(not identical[p] for p in parts):
                result[label] = {"skipped": "function not called by this variant"}
                continue
            if "from_packed" in parts:
                IrrepsBlockData.from_packed = classmethod(vectorized_from_packed)
            if "trace_alignment" in parts:
                EA.build_trace_alignment_from_pair_edges = vectorized_trace_alignment
            try:
                block = {
                    "forward": B.timing(variant["forward"], cases, 5, args.repeats),
                    "end_to_end": B.timing(
                        variant["end_to_end"], list(range(len(paths))), 2, args.repeats
                    ),
                }
            finally:
                IrrepsBlockData.from_packed = original_fp
                EA.build_trace_alignment_from_pair_edges = original_ta
            for value in block.values():
                value.pop("samples_ms", None)
            result[label] = block
            print(
                label.upper(),
                {k: round(v["median_ms"], 1) for k, v in block.items()},
                flush=True,
            )
            save()
        save()
    print("DONE", args.material, args.variant, flush=True)


if __name__ == "__main__":
    main()
