#!/usr/bin/env python3
"""Generate a frozen large synthetic/real case corpus for Stage-3 confirmation."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--d1-cache-dir", type=Path, required=True)
    result.add_argument("--d1-registry", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--cutoffs-angstrom", type=float, nargs="+", required=True)
    result.add_argument("--neighbor-counts", type=int, nargs="+", required=True)
    result.add_argument("--synthetic-cases-per-neighbor-count", type=int, required=True)
    result.add_argument("--real-cases-per-cutoff", type=int, required=True)
    result.add_argument("--real-neighbor-count", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--num-workers", type=int, required=True)
    return result


def synthetic_case(payload: tuple[float, int, int]) -> dict[str, object]:
    cutoff, neighbors, seed = payload
    generator = np.random.default_rng(seed)
    for _attempt in range(10000):
        directions = generator.normal(size=(neighbors, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = generator.uniform(0.18 * cutoff, 0.72 * cutoff, size=neighbors)
        coordinates = directions * radii[:, None]
        distances = (
            np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
            + np.eye(neighbors) * cutoff
        )
        if neighbors < 2 or np.min(distances) > 0.08 * cutoff:
            break
    else:
        raise RuntimeError("Failed to construct separated synthetic environment")
    species = np.array([8 if index % 2 == 0 else 14 for index in range(neighbors)])
    generator.shuffle(species)
    return {
        "kind": "synthetic_confirmation",
        "coordinates_angstrom": coordinates.tolist(),
        "species": species.tolist(),
        "neighbor_count": neighbors,
        "source": f"synthetic_seed_{seed}",
    }


def real_cases(args: argparse.Namespace, cutoff: float) -> list[dict[str, object]]:
    rows = [
        row
        for row in csv.DictReader(args.d1_registry.open())
        if row["split"] == "train"
    ]
    output = []
    for row in rows:
        index = int(row["structure_index"])
        path = args.d1_cache_dir / "shards" / f"structure_{index:04d}.h5"
        with h5py.File(path, "r") as handle:
            atomic_numbers = handle["atomic_numbers"][:]
            centers = handle["neighbor_center"][:]
            neighbors = handle["neighbor_atom"][:]
            displacement = handle["neighbor_displacement_angstrom"][:].astype(
                np.float64
            )
        distance = np.linalg.norm(displacement, axis=1)
        for center in range(len(atomic_numbers)):
            keep = np.nonzero((centers == center) & (distance < 0.72 * cutoff))[0]
            if len(keep) < args.real_neighbor_count:
                continue
            order = keep[np.argsort(distance[keep])[: args.real_neighbor_count]]
            output.append(
                {
                    "kind": "real_confirmation_subset",
                    "coordinates_angstrom": displacement[order].tolist(),
                    "species": atomic_numbers[neighbors[order]].astype(int).tolist(),
                    "neighbor_count": args.real_neighbor_count,
                    "source": f"structure_{index:04d}_center_{center}",
                }
            )
            if len(output) == args.real_cases_per_cutoff:
                return output
    raise RuntimeError("Insufficient real confirmation cases")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parser().parse_args()
    if (
        min(
            args.synthetic_cases_per_neighbor_count,
            args.real_cases_per_cutoff,
            args.real_neighbor_count,
            args.num_workers,
        )
        <= 0
    ):
        raise ValueError("Case and worker counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for cutoff_index, cutoff in enumerate(sorted(set(args.cutoffs_angstrom))):
        for neighbors in args.neighbor_counts:
            for repeat in range(args.synthetic_cases_per_neighbor_count):
                seed = (
                    args.seed + cutoff_index * 1_000_000 + neighbors * 10_000 + repeat
                )
                payloads.append((cutoff, neighbors, seed))
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        synthetic = list(pool.map(synthetic_case, payloads))
    cases = {}
    for cutoff in sorted(set(args.cutoffs_angstrom)):
        selected = [
            case
            for case, payload in zip(synthetic, payloads, strict=True)
            if payload[0] == cutoff
        ]
        selected.extend(real_cases(args, cutoff))
        cases[str(float(cutoff))] = selected
    encoded = json.dumps(cases, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(encoded.encode()).hexdigest()
    atomic_json(args.output_dir / "cases.json", cases)
    summary = {
        "passed": True,
        "content_hash": content_hash,
        "cutoff_count": len(cases),
        "synthetic_case_count_per_cutoff": len(args.neighbor_counts)
        * args.synthetic_cases_per_neighbor_count,
        "real_case_count_per_cutoff": args.real_cases_per_cutoff,
        "total_case_count": sum(len(values) for values in cases.values()),
        "selection_uses_hamiltonian": False,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
