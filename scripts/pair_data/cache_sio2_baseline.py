#!/usr/bin/env python3
"""Build directed SiO2 pair/target/neighbor shards and fit the frozen envelope."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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
from tqdm.auto import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pair_hamiltonian.range_factorization import (
    RangeEnvelopeAccumulator,
    envelope_manifest,
    plot_range_envelope_fits,
)
from pair_hamiltonian.sio2_cache import (
    DIRECTED_PAIR_NAMES,
    UNORDERED_PAIR_NAMES,
    build_structure_arrays,
    split_hash,
    validate_structure_shard,
    write_structure_shard,
)


class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-npz", type=Path, required=True)
    result.add_argument("--audit-summary", type=Path, required=True)
    result.add_argument("--split-indices", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--hamiltonian-cutoff-angstrom", type=float, required=True)
    result.add_argument("--descriptor-cutoff-angstrom", type=float, required=True)
    result.add_argument("--density-radial-count", type=int, required=True)
    result.add_argument("--density-l-max", type=int, required=True)
    result.add_argument("--envelope-bin-width-angstrom", type=float, required=True)
    result.add_argument("--plot-max-points-per-pair", type=int, required=True)
    result.add_argument("--nao-max", type=int, required=True)
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
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _worker(payload):
    index, source_key, split_name, graph, settings = payload
    arrays = build_structure_arrays(graph, **settings)
    return index, source_key, split_name, arrays


def _load_splits(
    path: Path, structure_count: int
) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path) as archive:
        split = {
            "train": np.asarray(archive["train_idx"], dtype=np.int64),
            "validation": np.asarray(archive["val_idx"], dtype=np.int64),
            "test": np.asarray(archive["test_idx"], dtype=np.int64),
        }
    names = [""] * structure_count
    for name, indices in split.items():
        for index in indices.tolist():
            if not 0 <= index < structure_count or names[index]:
                raise ValueError("Split indices are invalid or overlap")
            names[index] = name
    if any(not name for name in names):
        raise ValueError("Split indices do not cover every structure")
    return split, names


def main() -> None:
    args = parser().parse_args()
    positive = (
        args.hamiltonian_cutoff_angstrom,
        args.descriptor_cutoff_angstrom,
        args.density_radial_count,
        args.envelope_bin_width_angstrom,
        args.plot_max_points_per_pair,
        args.num_workers,
    )
    if any(value <= 0 for value in positive) or args.density_l_max < 0:
        raise ValueError("Cutoffs, counts, bin width, and workers must be positive")
    if args.nao_max != 14 or args.max_structures < 0:
        raise ValueError("This dataset requires nao-max=14 and max-structures >= 0")
    input_npz = args.input_npz.resolve()
    audit_path = args.audit_summary.resolve()
    split_path = args.split_indices.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_npz.is_file() or not audit_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Input NPZ, audit summary, and split indices must exist"
        )
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = (output_dir / "run.log").open("a" if args.resume else "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    configuration = {
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git": git_state(),
        "storage_format": "one gzip-1 HDF5 shard per structure",
        "target_dtype": "float32",
        "distance_dtype": "float32",
    }
    atomic_json(output_dir / "config.json", configuration)
    print(json.dumps(configuration, indent=2, sort_keys=True), flush=True)
    started = time.perf_counter()

    print("[1/7] Validating frozen audit inputs", flush=True)
    audit = json.loads(audit_path.read_text())
    cutoff = audit["hamiltonian_cutoff"]
    if not audit["validation"]["passed"] or not cutoff["found"]:
        raise ValueError("The source audit did not pass")
    if abs(cutoff["cutoff_angstrom"] - args.hamiltonian_cutoff_angstrom) > 1e-12:
        raise ValueError("Requested RH differs from the training-only audit")
    if input_npz.stat().st_size != audit["source"]["size_bytes"]:
        raise ValueError("Input size differs from the fingerprinted audit source")
    source_hash = sha256(input_npz)
    if source_hash != audit["source"]["sha256"]:
        raise ValueError("Input SHA256 differs from the fingerprinted audit source")
    projected_bytes = int(
        (audit["directed_offsite_block_count"] + audit["atom_count"]["total"])
        * 169
        * 4
        * 1.25
    )
    free_bytes = os.statvfs(cache_dir).f_bavail * os.statvfs(cache_dir).f_frsize
    print(
        f"Projected cache upper estimate: {projected_bytes / 2**30:.3f} GiB; "
        f"free: {free_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    if projected_bytes > free_bytes or projected_bytes > int(1.8 * 2**40):
        raise OSError("Projected cache violates free-space or 1.8 TiB reserve gate")

    print("[2/7] Loading released object graph collection", flush=True)
    with np.load(input_npz, allow_pickle=True) as archive:
        graphs = archive["graph"].item()
    ordered_keys = list(graphs)
    ordered_graphs = list(graphs.values())
    if len(ordered_graphs) != audit["structure_count"]:
        raise ValueError("Structure count differs from frozen audit")
    split, split_names = _load_splits(split_path, len(ordered_graphs))
    frozen_split_hash = split_hash(split)
    limit = (
        len(ordered_graphs)
        if args.max_structures == 0
        else min(args.max_structures, len(ordered_graphs))
    )
    print(f"Structures selected: {limit}/{len(ordered_graphs)}", flush=True)

    settings = {
        "descriptor_cutoff_angstrom": args.descriptor_cutoff_angstrom,
        "density_radial_count": args.density_radial_count,
        "density_l_max": args.density_l_max,
        "hamiltonian_cutoff_angstrom": args.hamiltonian_cutoff_angstrom,
        "nao_max": args.nao_max,
    }
    cache_metadata = {
        "version": "mandala-sio2-directed-baseline-cache-v2",
        "cartesian_convention": "openmx_xyz_to_e3nn_yzx",
        "offsite_supervision": "all directed reverse-pair members",
        "dataset_sha256": audit["source"]["sha256"],
        "split_hash": frozen_split_hash,
        **settings,
        "pair_names": DIRECTED_PAIR_NAMES,
        "target_schema_hashes": {
            name: audit["output_irrep_schemas"][name]["content_hash"]
            for name in DIRECTED_PAIR_NAMES
        },
    }
    atomic_json(output_dir / "cache_metadata.json", cache_metadata)
    print("[3/7] Building/resuming deterministic structure shards", flush=True)
    summaries: dict[int, object] = {}
    pending = []
    for index in range(limit):
        shard = cache_dir / "shards" / f"structure_{index:04d}.h5"
        if args.resume and validate_structure_shard(shard, cache_metadata):
            with h5py.File(shard, "r") as handle:
                summaries[index] = {
                    "structure_index": index,
                    "source_key": int(handle.attrs["source_key"]),
                    "split": str(handle.attrs["split"]),
                    "atom_count": int(handle["atomic_numbers"].shape[0]),
                    "offsite_pair_count": int(handle["offsite_source"].shape[0]),
                    "neighbor_count": int(handle["neighbor_center"].shape[0]),
                    "size_bytes": shard.stat().st_size,
                    "sha256": sha256(shard),
                }
            continue
        pending.append(
            (
                index,
                int(ordered_keys[index]),
                split_names[index],
                ordered_graphs[index],
                settings,
            )
        )
    iterator = map(_worker, pending)
    executor = None
    if args.num_workers > 1 and pending:
        executor = ProcessPoolExecutor(max_workers=args.num_workers)
        iterator = executor.map(_worker, pending, chunksize=1)
    try:
        for index, source_key, split_name, arrays in tqdm(
            iterator, total=len(pending), desc="cache shards", unit="structure"
        ):
            shard = cache_dir / "shards" / f"structure_{index:04d}.h5"
            summary = write_structure_shard(
                shard,
                arrays,
                structure_index=index,
                source_key=source_key,
                split=split_name,
                metadata=cache_metadata,
            )
            summaries[index] = asdict(summary)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    print("[4/7] Fitting train-only frozen range envelopes", flush=True)
    accumulators = {
        pair: RangeEnvelopeAccumulator(
            args.hamiltonian_cutoff_angstrom, args.envelope_bin_width_angstrom
        )
        for pair in UNORDERED_PAIR_NAMES
    }
    sample_distances: dict[str, list[np.ndarray]] = {
        pair: [] for pair in UNORDERED_PAIR_NAMES
    }
    sample_magnitudes: dict[str, list[np.ndarray]] = {
        pair: [] for pair in UNORDERED_PAIR_NAMES
    }
    for index in tqdm(range(limit), desc="envelope", unit="structure"):
        if split_names[index] != "train":
            continue
        shard = cache_dir / "shards" / f"structure_{index:04d}.h5"
        with h5py.File(shard, "r") as handle:
            pair_types = handle["offsite_pair_type"][:]
            distances = np.linalg.norm(
                handle["offsite_displacement_angstrom"][:], axis=1
            )
            targets = handle["offsite_target_irreps_hartree"][:]
        for pair_index, directed_pair in enumerate(DIRECTED_PAIR_NAMES):
            pair = (
                directed_pair
                if directed_pair in accumulators
                else "-".join(reversed(directed_pair.split("-")))
            )
            selected = pair_types == pair_index
            accumulators[pair].update(distances[selected], targets[selected])
            sample_distances[pair].append(distances[selected])
            sample_magnitudes[pair].append(
                np.sqrt(
                    np.mean(np.square(targets[selected].astype(np.float64)), axis=1)
                )
            )
    fits = [accumulators[pair].fit(pair) for pair in UNORDERED_PAIR_NAMES]
    envelope = envelope_manifest(
        fits,
        dataset_fingerprint=audit["source"]["sha256"],
        split_hash=frozen_split_hash,
    )
    atomic_json(output_dir / "range_envelope.json", envelope)
    plot_samples = {
        pair: (
            np.concatenate(sample_distances[pair]),
            np.concatenate(sample_magnitudes[pair]),
        )
        for pair in UNORDERED_PAIR_NAMES
    }
    plot_rows = plot_range_envelope_fits(
        fits,
        plot_samples,
        output_dir / "range_envelope_fit.png",
        max_scatter_points=args.plot_max_points_per_pair,
    )
    with (output_dir / "range_envelope_bins.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)

    print("[5/7] Writing cache registry and storage manifest", flush=True)
    registry_path = output_dir / "shards.csv"
    with registry_path.open("w", newline="") as stream:
        rows = [summaries[index] for index in sorted(summaries)]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    actual_bytes = sum(int(row["size_bytes"]) for row in summaries.values())
    storage = {
        "path": str(cache_dir),
        "description": "Directed full-block targets, D1 descriptors, and exact neighbors",
        "size_bytes": actual_bytes,
        "regenerable": True,
        "creation_script": str(Path(__file__).relative_to(REPOSITORY_ROOT)),
        "dataset_sha256": audit["source"]["sha256"],
    }
    atomic_json(output_dir / "storage.json", storage)

    print("[6/7] Validating completed cache", flush=True)
    total_atoms = sum(int(row["atom_count"]) for row in summaries.values())
    total_pairs = sum(int(row["offsite_pair_count"]) for row in summaries.values())
    total_neighbors = sum(int(row["neighbor_count"]) for row in summaries.values())
    full_run = limit == len(ordered_graphs)
    passed = full_run and len(summaries) == audit["structure_count"]
    summary = {
        "passed": passed,
        "full_run": full_run,
        "structure_count": len(summaries),
        "atom_count": total_atoms,
        "directed_offsite_pair_count": total_pairs,
        "exact_neighbor_count": total_neighbors,
        "cache_size_bytes": actual_bytes,
        "cache_metadata": cache_metadata,
        "range_envelope_hash": envelope["content_hash"],
        "range_envelope_plot": str(output_dir / "range_envelope_fit.png"),
        "range_envelope_fits": {fit.pair: asdict(fit) for fit in fits},
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# SiO2 directed baseline cache\n\n"
        f"Acceptance: **{'PASS' if passed else 'SMOKE-ONLY'}**\n\n"
        f"Structures: {len(summaries)}; atoms: {total_atoms}; directed offsite pairs: {total_pairs}.\n\n"
        f"Cache size: {actual_bytes / 2**30:.3f} GiB. The range envelope was fitted only on training shards.\n"
    )
    print("[7/7] Complete", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
