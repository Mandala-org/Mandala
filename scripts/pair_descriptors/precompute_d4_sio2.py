#!/usr/bin/env python3
"""Precompute a frozen SiO2 D4 grid by lifting the retained D1 cache."""

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

from pair_descriptors import DeterministicCovariantLiftDescriptor


TARGET_IRREPS = ((0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (4, 1))
_DESCRIPTORS: dict[str, DeterministicCovariantLiftDescriptor] = {}
_CONFIGS: list[dict[str, object]] = []


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-d1-cache-dir", type=Path, required=True)
    result.add_argument("--d1-summary", type=Path, required=True)
    result.add_argument("--d1-registry", type=Path, required=True)
    result.add_argument("--output-cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--cutoffs-angstrom", type=float, nargs="+", required=True)
    result.add_argument("--radial-bases", nargs="+", required=True)
    result.add_argument(
        "--resolution",
        action="append",
        nargs=3,
        metavar=("NAME", "N", "LMAX"),
        required=True,
    )
    result.add_argument("--body-degrees", type=int, nargs="+", required=True)
    result.add_argument("--lift-max-input-l", type=int, required=True)
    result.add_argument("--lift-max-radial-index", type=int, required=True)
    result.add_argument("--lift-radial-degree-budget", type=int, required=True)
    result.add_argument("--lift-max-intermediate-l", type=int, required=True)
    result.add_argument("--max-paths-per-degree-irrep", type=int, required=True)
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
        ("git", "diff", "--binary", "HEAD"), cwd=ROOT, capture_output=True, check=False
    ).stdout
    return {"commit": commit, "diff_sha256": hashlib.sha256(diff).hexdigest()}


def make_descriptor(config: dict[str, object]) -> DeterministicCovariantLiftDescriptor:
    return DeterministicCovariantLiftDescriptor(
        tuple(config["species"]),
        radial_basis=str(config["radial_basis"]),
        n_radial=int(config["n_radial"]),
        l_max=int(config["l_max"]),
        cutoff=float(config["cutoff_angstrom"]),
        body_degree=int(config["body_degree"]),
        lift_max_input_l=int(config["lift_max_input_l"]),
        lift_max_radial_index=int(config["lift_max_radial_index"]),
        lift_radial_degree_budget=int(config["lift_radial_degree_budget"]),
        lift_max_intermediate_l=int(config["lift_max_intermediate_l"]),
        target_irreps=TARGET_IRREPS,
        max_paths_per_degree_irrep=int(config["max_paths_per_degree_irrep"]),
    )


def descriptor_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    result = []
    for cutoff in sorted(set(args.cutoffs_angstrom)):
        for radial_basis in args.radial_bases:
            for name, n_radial, l_max in args.resolution:
                for body_degree in sorted(set(args.body_degrees)):
                    seed = {
                        "species": (8, 14),
                        "radial_basis": radial_basis,
                        "n_radial": int(n_radial),
                        "l_max": int(l_max),
                        "cutoff_angstrom": cutoff,
                        "body_degree": body_degree,
                        "lift_max_input_l": args.lift_max_input_l,
                        "lift_max_radial_index": args.lift_max_radial_index,
                        "lift_radial_degree_budget": args.lift_radial_degree_budget,
                        "lift_max_intermediate_l": args.lift_max_intermediate_l,
                        "max_paths_per_degree_irrep": args.max_paths_per_degree_irrep,
                    }
                    descriptor = make_descriptor(seed)
                    d1_key = f"r{cutoff:g}_{radial_basis}_{name}_n{n_radial}_l{l_max}".replace(
                        ".", "p"
                    )
                    key = f"r{cutoff:g}_d4_{radial_basis}_{name}_n{n_radial}_l{l_max}_b{body_degree}".replace(
                        ".", "p"
                    )
                    result.append(
                        {
                            "key": key,
                            "d1_key": d1_key,
                            "name": name,
                            **seed,
                            **descriptor.metadata,
                        }
                    )
    return result


def _initialize_worker(configs: list[dict[str, object]]) -> None:
    global _CONFIGS
    torch.set_num_threads(1)
    _CONFIGS = configs


def _write_one(payload: tuple) -> dict[str, object]:
    index, source, destination, split, manifest_hash = payload
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(".h5.partial")
    maximum_cast = 0.0
    with (
        h5py.File(source, "r") as input_handle,
        h5py.File(temporary, "w") as output_handle,
    ):
        source_key = int(input_handle.attrs["source_key"])
        output_handle.attrs.update(
            complete=False,
            structure_index=index,
            source_key=source_key,
            split=split,
            manifest_hash=manifest_hash,
        )
        group = output_handle.create_group("descriptors")
        for config in _CONFIGS:
            descriptor = _DESCRIPTORS.get(config["key"])
            if (
                descriptor is None
            ):  # spawn-platform fallback; ROSI/Linux inherits the parent cache.
                descriptor = make_descriptor(config).to(dtype=torch.float64)
                _DESCRIPTORS[config["key"]] = descriptor
            density = torch.from_numpy(
                input_handle["descriptors"][config["d1_key"]][:].astype(np.float64)
            )
            values64 = descriptor.lift(density)
            if not torch.isfinite(values64).all():
                raise FloatingPointError(f"Nonfinite D4 descriptor in shard {index}")
            values32 = values64.float()
            denominator = max(float(torch.linalg.vector_norm(values64)), 1e-30)
            maximum_cast = max(
                maximum_cast,
                float(torch.linalg.vector_norm(values32.double() - values64))
                / denominator,
            )
            group.create_dataset(
                config["key"],
                data=values32.numpy(),
                compression="gzip",
                compression_opts=1,
            )
        output_handle.attrs["complete"] = True
        output_handle.flush()
    os.replace(temporary, destination)
    with h5py.File(source, "r") as handle:
        atom_count = int(handle["atomic_numbers"].shape[0])
        neighbor_count = int(handle["neighbor_center"].shape[0])
    return {
        "structure_index": index,
        "source_key": source_key,
        "split": split,
        "atom_count": atom_count,
        "neighbor_count": neighbor_count,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "max_float32_cast_relative_error": maximum_cast,
    }


def main() -> None:
    args = parser().parse_args()
    if args.num_workers <= 0 or args.max_structures < 0:
        raise ValueError("Invalid worker or structure count")
    if any(value not in (2, 3) for value in args.body_degrees):
        raise ValueError("D4 body degrees must be 2 or 3")
    configs = descriptor_grid(args)
    manifest_hash = hashlib.sha256(
        json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    d1_summary = json.loads(args.d1_summary.read_text())
    if not d1_summary["passed"] or max(args.cutoffs_angstrom) > 10.5:
        raise ValueError("D1 cache invalid or requested cutoff exceeds its oracle")
    rows = list(csv.DictReader(args.d1_registry.open()))
    if args.max_structures:
        rows = rows[: args.max_structures]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("Completed D4 output exists")

    concise = [
        {
            "key": config["key"],
            "d1_key": config["d1_key"],
            "name": config["name"],
            "body_degree": config["body_degree"],
            "cutoff_angstrom": config["cutoff_angstrom"],
            "dimension": make_descriptor(config).irreps_out.dim,
            "raw_dimension": config["raw_dimension"],
            "lift_dimension": config["lift_dimension"],
            "path_count": len(config["paths"]),
            "content_hash": config["content_hash"],
        }
        for config in configs
    ]
    projected = (
        sum(int(row["atom_count"]) for row in rows)
        * sum(int(config["dimension"]) for config in concise)
        * 4
    )
    free = (
        os.statvfs(args.output_cache_dir).f_bavail
        * os.statvfs(args.output_cache_dir).f_frsize
    )
    if projected > free or projected > int(1.8 * 2**40):
        raise OSError("D4 projection violates free-space or 1.8 TiB gate")
    config_record = {
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "descriptor_configurations": concise,
        "descriptor_manifest_hash": manifest_hash,
        "target_irreps": TARGET_IRREPS,
        "selection_policy": "radial/angular/index ordering only; no labels, PCA, or validation metrics",
        "projected_descriptor_bytes_uncompressed": projected,
        "dataset_hash": d1_summary["dataset_hash"],
        "split_hash": d1_summary["split_hash"],
        "d1_summary_sha256": sha256(args.d1_summary),
        "d1_registry_sha256": sha256(args.d1_registry),
        "git": git_state(),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json(args.output_dir / "config.json", config_record)
    atomic_json(args.output_dir / "descriptor_schemas.json", configs)
    print(json.dumps(config_record, indent=2, sort_keys=True), flush=True)

    # Construct once before fork so all 64 workers inherit fixed CG buffers.
    _DESCRIPTORS.update(
        {
            config["key"]: make_descriptor(config).to(dtype=torch.float64)
            for config in configs
        }
    )
    tasks = []
    for row in rows:
        index = int(row["structure_index"])
        source = args.input_d1_cache_dir / "shards" / f"structure_{index:04d}.h5"
        destination = shard_dir / f"structure_{index:04d}.h5"
        valid = False
        if args.resume and destination.is_file():
            try:
                with h5py.File(destination, "r") as handle:
                    valid = (
                        bool(handle.attrs["complete"])
                        and handle.attrs["manifest_hash"] == manifest_hash
                    )
            except (OSError, KeyError):
                pass
        if not valid:
            tasks.append((index, source, destination, row["split"], manifest_hash))
    started = time.perf_counter()
    print(
        f"[1/4] Computing {len(configs)} D4 configurations for {len(tasks)} shards",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_initialize_worker,
        initargs=(configs,),
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
        "family": "d4",
        "structure_count": len(completed),
        "atom_count": sum(int(row["atom_count"]) for row in completed),
        "descriptor_configuration_count": len(configs),
        "cache_size_bytes": sum(int(row["size_bytes"]) for row in completed),
        "max_float32_cast_relative_error": maximum_cast,
        "elapsed_seconds": elapsed,
        "dataset_hash": d1_summary["dataset_hash"],
        "split_hash": d1_summary["split_hash"],
    }
    print("[3/4] Validating accounting", flush=True)
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(
        f"# SiO2 D4 precomputation\n\nAcceptance: **{'PASS' if summary['passed'] else 'FAIL'}**\n"
    )
    print("[4/4] Complete", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
