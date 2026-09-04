#!/usr/bin/env python3
"""Precompute the frozen SiO2 D1 radius/resolution grid and neighbor oracle."""

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

from pair_descriptors import RawNeighborDensityDescriptor
from pair_hamiltonian.sio2_cache import _geometry_neighbors


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-cache-dir", type=Path, required=True)
    result.add_argument("--cache-summary", type=Path, required=True)
    result.add_argument("--shard-registry", type=Path, required=True)
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
    return {
        "commit": commit,
        "dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
        ),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def configurations(args: argparse.Namespace) -> list[dict[str, object]]:
    result = []
    for cutoff in sorted(set(args.cutoffs_angstrom)):
        for basis in args.radial_bases:
            for name, n_radial, l_max in args.resolution:
                descriptor = RawNeighborDensityDescriptor(
                    (8, 14),
                    radial_basis=basis,
                    n_radial=int(n_radial),
                    l_max=int(l_max),
                    cutoff=cutoff,
                )
                key = f"r{cutoff:g}_{basis}_{name}_n{n_radial}_l{l_max}".replace(
                    ".", "p"
                )
                result.append({"key": key, "name": name, **descriptor.metadata})
    return result


def _write_one(payload: tuple) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    index, source, destination, split, configs, maximum_cutoff, manifest_hash = payload
    destination = Path(destination)
    temporary = destination.with_suffix(".h5.partial")
    with h5py.File(source, "r") as handle:
        z = handle["atomic_numbers"][:]
        positions = handle["positions_angstrom"][:].astype(np.float64)
        cell = handle["cell_angstrom"][:].astype(np.float64)
        source_key = int(handle.attrs["source_key"])
        reference_center = handle["neighbor_center"][:]
        reference_neighbor = handle["neighbor_atom"][:]
        reference_image = handle["neighbor_image"][:]
        reference_displacement = handle["neighbor_displacement_angstrom"][:]
        input_metadata = json.loads(handle.attrs["metadata_json"])
    center, neighbor, image, displacement = _geometry_neighbors(
        z, positions, cell, maximum_cutoff
    )
    reference_cutoff = float(input_metadata["descriptor_cutoff_angstrom"])
    reference_keep = np.linalg.norm(displacement, axis=1) < reference_cutoff
    if not (
        np.array_equal(center[reference_keep], reference_center)
        and np.array_equal(neighbor[reference_keep], reference_neighbor)
        and np.array_equal(image[reference_keep], reference_image)
    ):
        raise ValueError(
            f"Regenerated PBC neighbor identities differ for shard {index}"
        )
    displacement_error = float(
        np.max(
            np.abs(displacement[reference_keep] - reference_displacement), initial=0.0
        )
    )
    if displacement_error > 2.0e-5:
        raise ValueError(f"Regenerated displacement mismatch for shard {index}")
    max_cast_relative = 0.0
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            complete=False,
            structure_index=index,
            source_key=source_key,
            split=split,
            manifest_hash=manifest_hash,
        )
        handle.create_dataset(
            "atomic_numbers", data=z, compression="gzip", compression_opts=1
        )
        for name, value in (
            ("neighbor_center", center.astype(np.int32)),
            ("neighbor_atom", neighbor.astype(np.int32)),
            ("neighbor_image", image.astype(np.int16)),
            ("neighbor_displacement_angstrom", displacement.astype(np.float32)),
        ):
            handle.create_dataset(
                name, data=value, compression="gzip", compression_opts=1
            )
        group = handle.create_group("descriptors")
        for config in configs:
            cutoff = float(config["cutoff_angstrom"])
            keep = np.linalg.norm(displacement, axis=1) < cutoff
            descriptor = RawNeighborDensityDescriptor(
                tuple(config["species"]),
                radial_basis=config["radial_basis"],
                n_radial=int(config["n_radial"]),
                l_max=int(config["l_max"]),
                cutoff=cutoff,
            ).to(dtype=torch.float64)
            values64 = descriptor(
                torch.from_numpy(displacement[keep]),
                torch.from_numpy(z[neighbor[keep]].astype(np.int64)),
                torch.from_numpy(center[keep]),
                num_centers=len(z),
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
        "max_regenerated_displacement_error_angstrom": displacement_error,
    }


def main() -> None:
    args = parser().parse_args()
    if args.num_workers <= 0 or args.max_structures < 0:
        raise ValueError("num-workers must be positive and max-structures non-negative")
    if any(x <= 0 for x in args.cutoffs_angstrom):
        raise ValueError("cutoffs must be positive")
    if set(args.radial_bases) - {"spherical_bessel", "zernike"}:
        raise ValueError("Unsupported radial basis")
    configs = configurations(args)
    if len({c["key"] for c in configs}) != len(configs):
        raise ValueError("Descriptor configuration keys collide")
    manifest_payload = json.dumps(configs, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_payload.encode()).hexdigest()
    cache_summary = json.loads(args.cache_summary.read_text())
    if not cache_summary["passed"]:
        raise ValueError("Input cache has not passed validation")
    rows = list(csv.DictReader(args.shard_registry.open()))
    if args.max_structures:
        rows = rows[: args.max_structures]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("Completed output exists")
    concise_configs = [
        {
            "key": c["key"],
            "name": c["name"],
            "content_hash": c["content_hash"],
            "dimension": sum(
                channel["stop"] - channel["start"] for channel in c["channels"]
            ),
            "radial_basis": c["radial_basis"],
            "n_radial": c["n_radial"],
            "l_max": c["l_max"],
            "cutoff_angstrom": c["cutoff_angstrom"],
        }
        for c in configs
    ]
    component_sum = sum(c["dimension"] for c in concise_configs)
    atoms = sum(int(row["atom_count"]) for row in rows)
    projected_descriptor_bytes = atoms * component_sum * 4
    free = (
        os.statvfs(args.output_cache_dir).f_bavail
        * os.statvfs(args.output_cache_dir).f_frsize
    )
    if projected_descriptor_bytes > free or projected_descriptor_bytes > int(
        1.8 * 2**40
    ):
        raise OSError("D1 projection violates free-space or 1.8 TiB gate")
    config = {
        **{
            k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "descriptor_configurations": concise_configs,
        "descriptor_manifest_hash": manifest_hash,
        "projected_descriptor_bytes_uncompressed": projected_descriptor_bytes,
        "input_dataset_hash": cache_summary["cache_metadata"]["dataset_sha256"],
        "input_split_hash": cache_summary["cache_metadata"]["split_hash"],
        "input_summary_sha256": sha256(args.cache_summary),
        "input_registry_sha256": sha256(args.shard_registry),
        "git": git_state(),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "storage": "one gzip-1 HDF5 shard per structure; float32 descriptors",
    }
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(args.output_dir / "descriptor_schemas.json", configs)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    tasks = []
    for row in rows:
        index = int(row["structure_index"])
        source = args.input_cache_dir / "shards" / f"structure_{index:04d}.h5"
        destination = shard_dir / f"structure_{index:04d}.h5"
        if args.resume and destination.is_file():
            try:
                with h5py.File(destination, "r") as handle:
                    valid = (
                        bool(handle.attrs["complete"])
                        and handle.attrs["manifest_hash"] == manifest_hash
                    )
            except (OSError, KeyError):
                valid = False
            if valid:
                continue
        tasks.append(
            (
                index,
                source,
                destination,
                row["split"],
                configs,
                max(args.cutoffs_angstrom),
                manifest_hash,
            )
        )
    started = time.perf_counter()
    print(
        f"[1/4] Computing {len(configs)} D1 configurations for {len(tasks)} shards",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        completed = list(
            tqdm(pool.map(_write_one, tasks), total=len(tasks), unit="structure")
        )
    completed_by_index = {int(row["structure_index"]): row for row in completed}
    if args.resume:
        for row in rows:
            index = int(row["structure_index"])
            if index not in completed_by_index:
                path = shard_dir / f"structure_{index:04d}.h5"
                with h5py.File(path, "r") as handle:
                    completed_by_index[index] = {
                        "structure_index": index,
                        "source_key": int(handle.attrs["source_key"]),
                        "split": str(handle.attrs["split"]),
                        "atom_count": len(handle["atomic_numbers"]),
                        "neighbor_count": len(handle["neighbor_center"]),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "max_float32_cast_relative_error": 0.0,
                        "max_regenerated_displacement_error_angstrom": 0.0,
                    }
    completed = [completed_by_index[int(row["structure_index"])] for row in rows]
    print("[2/4] Writing registry", flush=True)
    fields = list(completed[0]) if completed else []
    with (args.output_dir / "shards.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    elapsed = time.perf_counter() - started
    total_neighbors = sum(int(row["neighbor_count"]) for row in completed)
    summary = {
        "passed": len(completed) == len(rows)
        and all(len(row["sha256"]) == 64 for row in completed)
        and max(
            (float(r["max_float32_cast_relative_error"]) for r in completed),
            default=0.0,
        )
        <= 5e-7,
        "full_run": args.max_structures == 0,
        "structure_count": len(completed),
        "atom_count": sum(int(r["atom_count"]) for r in completed),
        "maximum_cutoff_neighbor_count": total_neighbors,
        "descriptor_configuration_count": len(configs),
        "cache_size_bytes": sum(int(r["size_bytes"]) for r in completed),
        "max_float32_cast_relative_error": max(
            (float(r["max_float32_cast_relative_error"]) for r in completed),
            default=0.0,
        ),
        "max_regenerated_displacement_error_angstrom": max(
            (
                float(r["max_regenerated_displacement_error_angstrom"])
                for r in completed
            ),
            default=0.0,
        ),
        "elapsed_seconds": elapsed,
        "neighbor_contributions_per_second": total_neighbors
        * len(configs)
        / max(elapsed, 1e-12),
        "dataset_hash": cache_summary["cache_metadata"]["dataset_sha256"],
        "split_hash": cache_summary["cache_metadata"]["split_hash"],
    }
    print("[3/4] Validating accounting", flush=True)
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(
        f"# SiO2 D1 descriptor precomputation\n\nAcceptance: **{'PASS' if summary['passed'] else 'FAIL'}**\n\n"
        f"Stored {len(configs)} configurations for {len(completed)} structures in "
        f"{summary['cache_size_bytes']/2**30:.3f} GiB.\n"
    )
    print("[4/4] Complete", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
