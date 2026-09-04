#!/usr/bin/env python3
"""Precompute one frozen D2 or D3 SiO2 descriptor grid from the D1 oracle."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pair_descriptors import FourierBesselDescriptor, IrreducibleMomentDescriptor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--family", choices=("d2", "d3"), required=True)
    result.add_argument("--input-oracle-dir", type=Path, required=True)
    result.add_argument("--oracle-summary", type=Path, required=True)
    result.add_argument("--oracle-registry", type=Path, required=True)
    result.add_argument("--output-cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--cutoffs-angstrom", type=float, nargs="+", required=True)
    result.add_argument(
        "--resolution",
        action="append",
        nargs=3,
        metavar=("NAME", "ORDER", "LMAX"),
        required=True,
        help="D2: ORDER=max degree and LMAX must match; D3: ORDER=frequency count.",
    )
    result.add_argument("--num-workers", type=int, required=True)
    result.add_argument("--max-structures", type=int, default=0)
    result.add_argument("--resume", action="store_true")
    return result


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout
    return {"commit": commit, "diff_sha256": hashlib.sha256(diff).hexdigest()}


def make_descriptor(family: str, config: dict[str, object]):
    common = {"species": (8, 14), "cutoff": float(config["cutoff_angstrom"])}
    if family == "d2":
        return IrreducibleMomentDescriptor(**common, max_degree=int(config["order"]))
    return FourierBesselDescriptor(
        **common,
        frequency_count=int(config["order"]),
        l_max=int(config["l_max"]),
    )


def descriptor_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    result = []
    for cutoff in sorted(set(args.cutoffs_angstrom)):
        for name, order, l_max in args.resolution:
            order_value, l_value = int(order), int(l_max)
            if args.family == "d2" and order_value != l_value:
                raise ValueError("D2 ORDER and LMAX must match the total-degree cap")
            seed = {
                "family": args.family,
                "name": name,
                "order": order_value,
                "l_max": l_value,
                "cutoff_angstrom": cutoff,
            }
            descriptor = make_descriptor(args.family, seed)
            key = f"r{cutoff:g}_{args.family}_{name}_o{order}_l{l_max}".replace(
                ".", "p"
            )
            result.append({"key": key, **seed, **descriptor.metadata})
    return result


def _initialize_worker() -> None:
    torch.set_num_threads(1)


def _write_one(payload: tuple) -> dict[str, object]:
    family, index, source, destination, split, configs, manifest_hash = payload
    destination = Path(destination)
    temporary = destination.with_suffix(".h5.partial")
    with h5py.File(source, "r") as handle:
        z = handle["atomic_numbers"][:]
        center = handle["neighbor_center"][:].astype(np.int64)
        neighbor = handle["neighbor_atom"][:].astype(np.int64)
        displacement = handle["neighbor_displacement_angstrom"][:].astype(np.float64)
        source_key = int(handle.attrs["source_key"])
    distance = np.linalg.norm(displacement, axis=1)
    max_cast_relative = 0.0
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            complete=False,
            structure_index=index,
            source_key=source_key,
            split=split,
            manifest_hash=manifest_hash,
        )
        group = handle.create_group("descriptors")
        for config in configs:
            keep = distance < float(config["cutoff_angstrom"])
            descriptor = make_descriptor(family, config).to(dtype=torch.float64)
            values64 = descriptor(
                torch.from_numpy(displacement[keep]),
                torch.from_numpy(z[neighbor[keep]].astype(np.int64)),
                torch.from_numpy(center[keep]),
                num_centers=len(z),
            )
            if not torch.isfinite(values64).all():
                raise FloatingPointError(
                    f"Nonfinite {family} descriptor in shard {index}"
                )
            values32 = values64.float()
            denominator = max(float(torch.linalg.vector_norm(values64)), 1e-30)
            max_cast_relative = max(
                max_cast_relative,
                float(torch.linalg.vector_norm(values32.double() - values64))
                / denominator,
            )
            group.create_dataset(
                config["key"],
                data=values32.numpy(),
                compression="gzip",
                compression_opts=1,
            )
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(temporary, destination)
    return {
        "structure_index": index,
        "source_key": source_key,
        "split": split,
        "atom_count": len(z),
        "neighbor_count": len(center),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "max_float32_cast_relative_error": max_cast_relative,
    }


def main() -> None:
    args = parser().parse_args()
    if args.num_workers <= 0 or args.max_structures < 0:
        raise ValueError("Invalid worker or structure count")
    configs = descriptor_grid(args)
    manifest = hashlib.sha256(
        json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    oracle = json.loads(args.oracle_summary.read_text())
    if not oracle["passed"] or max(args.cutoffs_angstrom) > 10.5:
        raise ValueError("Oracle invalid or requested cutoff exceeds 10.5 angstrom")
    rows = list(csv.DictReader(args.oracle_registry.open()))
    if args.max_structures:
        rows = rows[: args.max_structures]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("Completed output exists")
    concise = [
        {
            "key": c["key"],
            "name": c["name"],
            "order": c["order"],
            "l_max": c["l_max"],
            "cutoff_angstrom": c["cutoff_angstrom"],
            "dimension": make_descriptor(args.family, c).irreps_out.dim,
            "content_hash": c["content_hash"],
        }
        for c in configs
    ]
    projected = (
        sum(int(row["atom_count"]) for row in rows)
        * sum(int(c["dimension"]) for c in concise)
        * 4
    )
    free = (
        os.statvfs(args.output_cache_dir).f_bavail
        * os.statvfs(args.output_cache_dir).f_frsize
    )
    if projected > free or projected > int(1.8 * 2**40):
        raise OSError("Projection violates free-space or 1.8 TiB gate")
    config = {
        **{
            k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "descriptor_configurations": concise,
        "descriptor_manifest_hash": manifest,
        "projected_descriptor_bytes_uncompressed": projected,
        "dataset_hash": oracle["dataset_hash"],
        "split_hash": oracle["split_hash"],
        "oracle_summary_sha256": sha256(args.oracle_summary),
        "oracle_registry_sha256": sha256(args.oracle_registry),
        "git": git_state(),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(args.output_dir / "descriptor_schemas.json", configs)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    tasks = []
    for row in rows:
        index = int(row["structure_index"])
        source = args.input_oracle_dir / "shards" / f"structure_{index:04d}.h5"
        destination = shard_dir / f"structure_{index:04d}.h5"
        valid = False
        if args.resume and destination.is_file():
            try:
                with h5py.File(destination, "r") as handle:
                    valid = (
                        bool(handle.attrs["complete"])
                        and handle.attrs["manifest_hash"] == manifest
                    )
            except (OSError, KeyError):
                pass
        if not valid:
            tasks.append(
                (
                    args.family,
                    index,
                    source,
                    destination,
                    row["split"],
                    configs,
                    manifest,
                )
            )
    started = time.perf_counter()
    print(
        f"[1/4] Computing {len(configs)} {args.family.upper()} configurations for {len(tasks)} shards",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_initialize_worker
    ) as pool:
        fresh = list(
            tqdm(pool.map(_write_one, tasks), total=len(tasks), unit="structure")
        )
    records = {int(row["structure_index"]): row for row in fresh}
    for row in rows:
        index = int(row["structure_index"])
        if index in records:
            continue
        path = shard_dir / f"structure_{index:04d}.h5"
        with h5py.File(path, "r") as handle:
            records[index] = {
                "structure_index": index,
                "source_key": int(handle.attrs["source_key"]),
                "split": str(handle.attrs["split"]),
                "atom_count": int(row["atom_count"]),
                "neighbor_count": int(row["neighbor_count"]),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "max_float32_cast_relative_error": 0.0,
            }
    completed = [records[int(row["structure_index"])] for row in rows]
    print("[2/4] Writing registry", flush=True)
    with (args.output_dir / "shards.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    elapsed = time.perf_counter() - started
    maximum_cast = max(
        float(row["max_float32_cast_relative_error"]) for row in completed
    )
    summary = {
        "passed": len(completed) == len(rows) and maximum_cast <= 5e-7,
        "full_run": args.max_structures == 0,
        "family": args.family,
        "structure_count": len(completed),
        "atom_count": sum(int(r["atom_count"]) for r in completed),
        "descriptor_configuration_count": len(configs),
        "cache_size_bytes": sum(int(r["size_bytes"]) for r in completed),
        "max_float32_cast_relative_error": maximum_cast,
        "elapsed_seconds": elapsed,
        "dataset_hash": oracle["dataset_hash"],
        "split_hash": oracle["split_hash"],
    }
    print("[3/4] Validating accounting", flush=True)
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(
        f"# SiO2 {args.family.upper()} precomputation\n\nAcceptance: **{'PASS' if summary['passed'] else 'FAIL'}**\n"
    )
    print("[4/4] Complete", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
