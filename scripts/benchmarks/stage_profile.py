"""Measure the time of every pipeline stage and compare it with the cost model.

Each stage is timed with CUDA events (or wall clock on CPU) around its forward
call; the record also carries the stage's parameter count and the
cost-model operation estimate for the measured structure, which
``fit_cost_constants.py`` turns into seconds-per-operation constants.

Example::

    PYTHONPATH=src:. python scripts/benchmarks/stage_profile.py \\
        --config configs/reference/deeph_e3.yaml --shells '{"Si": [0, 0, 1, 1, 2]}' \\
        --atoms 64 --output profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_timing import (
    build_model,
    model_inputs,
    synchronize,
    synthetic_crystal,
)  # noqa: E402


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shells", required=True)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.066)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


class StageTimer:
    """Forward hooks recording the duration of every pipeline stage."""

    def __init__(self, pipeline, device):
        self.cuda = str(device).startswith("cuda")
        self.samples = defaultdict(list)
        self.handles = []
        for stage, module in pipeline.transforms.items():
            self.handles.append(module.register_forward_pre_hook(self._start(stage)))
            self.handles.append(module.register_forward_hook(self._stop(stage)))
        self.open = {}

    def _start(self, stage):
        def hook(module, inputs):
            if self.cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.open[stage] = event
            else:
                self.open[stage] = time.perf_counter()

        return hook

    def _stop(self, stage):
        def hook(module, inputs, output):
            start = self.open.pop(stage)
            if self.cuda:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.samples[stage].append((start, end))
            else:
                self.samples[stage].append(time.perf_counter() - start)

        return hook

    def seconds(self):
        synchronize("cuda" if self.cuda else "cpu")
        result = {}
        for stage, values in self.samples.items():
            if self.cuda:
                values = [start.elapsed_time(end) / 1e3 for start, end in values]
            result[stage] = statistics.median(values)
        return result

    def reset(self):
        self.samples.clear()


def main(argv=None):
    from net.cost_model import SystemSize, operations_per_entity, stage_cost

    args = parse_arguments(argv)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    shells = {key: list(value) for key, value in json.loads(args.shells).items()}
    model, cfg, mapper = build_model(args.config, shells, args.radius, args.dtype)
    model = model.to(args.device, dtype).eval()
    symbols, positions, box = synthetic_crystal(
        shells, args.atoms, args.density, args.seed
    )
    x = model_inputs(model, cfg, mapper, symbols, positions, box, args.device, dtype)
    x.pop("_static_features", None)  # time every stage, including the cached ones
    timer = StageTimer(model.pipeline, args.device)
    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                model(x)
            synchronize(args.device)
            timer.reset()
            for _ in range(args.repeats):
                model(x)
    except torch.OutOfMemoryError:
        # A size that does not fit is recorded and left out of the cost fit.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "config": str(args.config),
                    "atoms": args.atoms,
                    "edges": int(x["edge_index"].shape[1]),
                    "out_of_memory": True,
                    "stages": [],
                },
                indent=2,
            )
        )
        print(f"{args.config} ({args.atoms} atoms): out of CUDA memory")
        return
    measured = timer.seconds()
    edges = int(x["edge_index"].shape[1])
    size = SystemSize(atoms=args.atoms, neighbors=edges / args.atoms)
    specs = model.pipeline.feature_specs
    stages = []
    for record in model.pipeline.records:
        module = model.pipeline.transforms[record.id]
        operations, predicted = stage_cost(
            record.transform,
            module,
            [specs[s] for s in record.inputs],
            [specs[s] for s in record.outputs],
            size,
        )
        stages.append(
            {
                "id": record.id,
                "transform": record.transform,
                "group": record.group,
                "parameters": sum(p.numel() for p in module.parameters()),
                "operations": operations,
                "operations_per_entity": operations_per_entity(module),
                "predicted_s": predicted,
                "measured_s": measured.get(record.id),
            }
        )
    result = {
        "config": str(args.config),
        "shells": shells,
        "atoms": args.atoms,
        "edges": edges,
        "device": str(args.device),
        "device_name": (
            torch.cuda.get_device_name()
            if str(args.device).startswith("cuda")
            else "cpu"
        ),
        "dtype": args.dtype,
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    total = sum(stage["measured_s"] or 0.0 for stage in stages)
    print(
        f"{args.config} ({args.atoms} atoms, {edges} edges): {len(stages)} stages, {total * 1e3:.2f} ms"
    )


if __name__ == "__main__":
    main()
