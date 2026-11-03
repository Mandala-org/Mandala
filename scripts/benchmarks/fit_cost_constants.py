"""Fit seconds-per-operation constants of the cost model from stage profiles.

For every transformation, ``seconds = overhead + rate * operations`` is fitted
by non-negative least squares over all its profiled stages; transformations
with fewer than three samples use the pooled fit. Writes the JSON format read
by ``net.cost_model.load_constants``.

Example::

    python scripts/benchmarks/fit_cost_constants.py profiles/*.json --output cost_constants.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


def fit(samples):
    operations = np.array([s[0] for s in samples], dtype=float)
    seconds = np.array([s[1] for s in samples], dtype=float)
    design = np.stack([np.ones_like(operations), operations], axis=1)
    (overhead, rate), _ = nnls(design, seconds)
    predicted = design @ np.array([overhead, rate])
    relative = np.abs(predicted - seconds) / np.maximum(seconds, 1e-9)
    return {
        "overhead_seconds": float(overhead),
        "seconds_per_operation": float(rate),
        "samples": len(samples),
        "median_relative_error": float(np.median(relative)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    by_transform, pooled, devices = defaultdict(list), [], set()
    for path in args.profiles:
        profile = json.loads(path.read_text())
        if profile.get("out_of_memory"):
            continue
        devices.add(profile["device_name"])
        for stage in profile["stages"]:
            if stage["measured_s"] is None:
                continue
            sample = (stage["operations"], stage["measured_s"])
            by_transform[stage["transform"]].append(sample)
            pooled.append(sample)
    constants = {
        "device": sorted(devices),
        "default": fit(pooled),
        "transforms": {
            name: fit(samples)
            for name, samples in sorted(by_transform.items())
            if len(samples) >= 3
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(constants, indent=2))
    print(
        f"fitted {len(constants['transforms'])} transformations from {len(pooled)} stage samples"
    )


if __name__ == "__main__":
    main()
