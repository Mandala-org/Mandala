#!/usr/bin/env python3
"""Fit train-only equivariant RMS scales for all cached SiO2 descriptors."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

from e3nn.o3 import Irreps
import h5py
import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--family",
        action="append",
        nargs=4,
        metavar=("NAME", "CACHE_DIR", "ARTIFACT_DIR", "SCHEMA_JSON"),
        required=True,
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--num-workers", type=int, required=True)
    result.add_argument("--scale-floor", type=float, required=True)
    return result


def _sum_chunk(payload: tuple) -> tuple[str, int, dict[str, np.ndarray]]:
    name, cache_dir, rows, layouts = payload
    sums = {
        key: np.zeros(int(dimension), dtype=np.float64)
        for key, dimension in layouts.items()
    }
    atom_count = 0
    for row in rows:
        path = (
            Path(cache_dir)
            / "shards"
            / f"structure_{int(row['structure_index']):04d}.h5"
        )
        with h5py.File(path, "r") as handle:
            group = handle["descriptors"]
            first = group[next(iter(layouts))]
            atom_count += int(first.shape[0])
            for key in layouts:
                values = group[key][:].astype(np.float64)
                sums[key] += np.square(values).sum(axis=0)
    return name, atom_count, sums


def main() -> None:
    args = parser().parse_args()
    if args.num_workers <= 0 or args.scale_floor <= 0:
        raise ValueError("Invalid worker count or scale floor")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = []
    tasks = []
    for name, cache_dir_text, artifact_dir_text, schema_text in args.family:
        cache_dir, artifact_dir, schema_path = map(
            Path, (cache_dir_text, artifact_dir_text, schema_text)
        )
        summary = json.loads((artifact_dir / "summary.json").read_text())
        if not summary["passed"] or not summary["full_run"]:
            raise ValueError(f"Incomplete source family: {name}")
        schemas = json.loads(schema_path.read_text())
        layouts = {
            schema["key"]: Irreps(schema["irreps_out"]).dim for schema in schemas
        }
        rows = [
            row
            for row in csv.DictReader((artifact_dir / "shards.csv").open())
            if row["split"] == "train"
        ]
        chunks = [
            rows[index :: min(args.num_workers, len(rows))]
            for index in range(min(args.num_workers, len(rows)))
        ]
        tasks.extend(
            (name, str(cache_dir), chunk, layouts) for chunk in chunks if chunk
        )
        families.append((name, schemas, summary, len(rows)))
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        partials = list(pool.map(_sum_chunk, tasks))
    output = {
        "convention": "mandala-irrep-channel-train-rms-v1",
        "centering": "none",
        "scale_floor": args.scale_floor,
        "families": {},
    }
    for name, schemas, summary, train_structures in families:
        selected = [item for item in partials if item[0] == name]
        atom_count = sum(item[1] for item in selected)
        records = []
        for schema in schemas:
            key = schema["key"]
            component_sums = sum(
                (item[2][key] for item in selected),
                np.zeros(Irreps(schema["irreps_out"]).dim),
            )
            channels = []
            offset = 0
            for multiplicity, irrep in Irreps(schema["irreps_out"]):
                for _copy in range(multiplicity):
                    stop = offset + irrep.dim
                    mean_square = float(component_sums[offset:stop].sum()) / (
                        atom_count * irrep.dim
                    )
                    channels.append(
                        {
                            "start": offset,
                            "stop": stop,
                            "l": irrep.l,
                            "parity": irrep.p,
                            "rms": math.sqrt(max(mean_square, args.scale_floor**2)),
                        }
                    )
                    offset = stop
            records.append({"key": key, "dimension": offset, "channels": channels})
        output["families"][name] = {
            "dataset_hash": summary["dataset_hash"],
            "split_hash": summary["split_hash"],
            "train_structure_count": train_structures,
            "train_atom_count": atom_count,
            "descriptors": records,
        }
    hashes = {value["dataset_hash"] for value in output["families"].values()}
    splits = {value["split_hash"] for value in output["families"].values()}
    passed = len(hashes) == len(splits) == 1 and all(
        descriptor["dimension"] > 0
        and all(
            math.isfinite(channel["rms"]) and channel["rms"] >= args.scale_floor
            for channel in descriptor["channels"]
        )
        for family in output["families"].values()
        for descriptor in family["descriptors"]
    )
    encoded = json.dumps(output, sort_keys=True, separators=(",", ":"))
    output["content_hash"] = hashlib.sha256(encoded.encode()).hexdigest()
    temporary = args.output_dir / "normalization.json.tmp"
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_dir / "normalization.json")
    summary = {
        "passed": passed,
        "family_count": len(families),
        "descriptor_configuration_count": sum(
            len(value["descriptors"]) for value in output["families"].values()
        ),
        "dataset_hash": next(iter(hashes)),
        "split_hash": next(iter(splits)),
        "train_only": True,
        "normalization_content_hash": output["content_hash"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
