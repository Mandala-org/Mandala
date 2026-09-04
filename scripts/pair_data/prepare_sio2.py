#!/usr/bin/env python3
"""Prepare compact Stage-1 metadata for the released HamGNN SiO2 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import e3nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pair_hamiltonian.hamgnn_sio2 import inspect_graphs, published_protocol_split


class Tee:
    """Mirror progress and errors to the terminal and the persistent run log."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--input",
        type=Path,
        required=True,
        help="graph_data.npz, its directory, or a ZIP containing it",
    )
    result.add_argument(
        "--extraction-dir",
        type=Path,
        required=True,
        help="Large-data directory for necessary ZIP extraction",
    )
    result.add_argument(
        "--output-dir", type=Path, required=True, help="Small artifact directory"
    )
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--train-ratio", type=float, required=True)
    result.add_argument("--validation-ratio", type=float, required=True)
    result.add_argument("--distance-bin-width-angstrom", type=float, required=True)
    result.add_argument("--benchmark-mae-mev", type=float, required=True)
    result.add_argument("--absolute-tail-mae-limit-mev", type=float, required=True)
    result.add_argument("--retained-squared-fraction", type=float, required=True)
    result.add_argument(
        "--descriptor-cutoffs-angstrom", type=float, nargs="+", required=True
    )
    result.add_argument("--nao-max", type=int, required=True)
    result.add_argument(
        "--torch-threads",
        type=int,
        required=True,
        help="CPU threads; one avoids overhead for small block checks",
    )
    result.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a validated extraction and incomplete artifact directory",
    )
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


def resolve_npz(source: Path, extraction_dir: Path, resume: bool) -> Path:
    if source.is_dir():
        source = source / "graph_data.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".npz":
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError("--input must resolve to an .npz or .zip file")
    extraction_dir.mkdir(parents=True, exist_ok=True)
    destination = extraction_dir / "graph_data.npz"
    with zipfile.ZipFile(source) as archive:
        candidates = [
            item
            for item in archive.infolist()
            if Path(item.filename).name == "graph_data.npz" and not item.is_dir()
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one graph_data.npz in archive, found {len(candidates)}"
            )
        member = candidates[0]
        if destination.exists():
            if not resume or destination.stat().st_size != member.file_size:
                raise FileExistsError(
                    f"Refusing to overwrite {destination}; pass --resume only for a matching extraction"
                )
            return destination
        free = shutil.disk_usage(extraction_dir).free
        if free < member.file_size * 1.1:
            raise OSError(
                f"Insufficient extraction space: need about {member.file_size * 1.1:.0f} bytes, have {free}"
            )
        temporary = destination.with_suffix(".npz.partial")
        try:
            with (
                archive.open(member) as source_stream,
                temporary.open("wb") as output_stream,
            ):
                shutil.copyfileobj(
                    source_stream, output_stream, length=16 * 1024 * 1024
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            arguments, cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def main() -> None:
    args = parser().parse_args()
    if (
        args.distance_bin_width_angstrom <= 0
        or args.nao_max != 14
        or args.torch_threads <= 0
    ):
        raise ValueError(
            "distance bin width must be positive and this release requires --nao-max 14"
        )
    if any(value <= 0 for value in args.descriptor_cutoffs_angstrom):
        raise ValueError("descriptor cutoffs must be positive")
    output_dir = args.output_dir.resolve()
    if (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"Completed output already exists: {output_dir / 'summary.json'}"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Non-empty output directory {output_dir}; use --resume for an interrupted run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_stream = (output_dir / "run.log").open("a" if args.resume else "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_stream)
    sys.stderr = Tee(sys.stderr, log_stream)
    torch.set_num_threads(args.torch_threads)
    resolved = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    resolved["software"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "e3nn": e3nn.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    resolved["git"] = git_state()
    atomic_json(output_dir / "config.json", resolved)
    print(json.dumps(resolved, indent=2, sort_keys=True), flush=True)

    started = time.perf_counter()
    print("[1/6] Resolving and validating source", flush=True)
    npz_path = resolve_npz(
        args.input.resolve(), args.extraction_dir.resolve(), args.resume
    )
    print(f"NPZ: {npz_path} ({npz_path.stat().st_size / 2**30:.3f} GiB)", flush=True)
    print("[2/6] Computing source fingerprint", flush=True)
    source_hash = sha256(npz_path)
    print(f"SHA256: {source_hash}", flush=True)
    print("[3/6] Loading object graph collection (high-memory stage)", flush=True)
    with np.load(npz_path, allow_pickle=True) as archive:
        if archive.files != ["graph"]:
            raise ValueError(f"Expected sole NPZ member 'graph', got {archive.files}")
        graphs = archive["graph"].item()
    if not isinstance(graphs, dict) or not all(
        isinstance(key, (int, np.integer)) for key in graphs
    ):
        raise ValueError("Expected an integer-keyed graph dictionary")
    structure_keys = np.asarray(list(graphs), dtype=np.int64)
    if len(np.unique(structure_keys)) != len(structure_keys):
        raise ValueError("Graph dictionary keys are not unique")
    print(f"Loaded {len(graphs)} structures", flush=True)
    split = published_protocol_split(
        len(graphs),
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    np.savez(
        output_dir / "split_indices.npz",
        train_idx=split.train,
        val_idx=split.validation,
        test_idx=split.test,
        train_source_key=structure_keys[split.train],
        val_source_key=structure_keys[split.validation],
        test_source_key=structure_keys[split.test],
    )
    print(
        "[4/6] Validating blocks, symmetry, basis conversion, and locality", flush=True
    )
    summary, bins = inspect_graphs(
        graphs,
        split,
        nao_max=args.nao_max,
        distance_bin_width_angstrom=args.distance_bin_width_angstrom,
        benchmark_mae_mev=args.benchmark_mae_mev,
        absolute_mae_limit_mev=args.absolute_tail_mae_limit_mev,
        retained_squared_fraction=args.retained_squared_fraction,
        descriptor_cutoffs_angstrom=args.descriptor_cutoffs_angstrom,
        show_progress=True,
    )
    summary["source"] = {
        "npz_path": str(npz_path),
        "size_bytes": npz_path.stat().st_size,
        "sha256": source_hash,
    }
    summary["source"]["graph_key_min"] = int(structure_keys.min())
    summary["source"]["graph_key_max"] = int(structure_keys.max())
    summary["source"]["missing_graph_keys"] = sorted(
        set(range(int(structure_keys.min()), int(structure_keys.max()) + 1))
        - set(structure_keys.tolist())
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    print("[5/6] Writing compact artifacts", flush=True)
    distance_bins_path = output_dir / "distance_bins.csv"
    distance_bins_temporary = distance_bins_path.with_suffix(".csv.tmp")
    with distance_bins_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    os.replace(distance_bins_temporary, distance_bins_path)
    atomic_json(output_dir / "irrep_schema.json", summary["output_irrep_schemas"])
    atomic_json(output_dir / "summary.json", summary)
    cutoff = summary["hamiltonian_cutoff"]
    mismatch = (
        "PASS" if summary["publication_count_matches"] else "PARTIAL/INCONCLUSIVE"
    )
    report = f"""# HamGNN SiO2 Stage-1 data audit

## Objective

Validate the released graph bundle, freeze a protocol-matched split, and establish the observed-support Hamiltonian cutoff before model training.

## Acceptance decision

- Block packing, periodic shifts, inverse edges, Hermiticity, and AO-irrep round trip: **{'PASS' if summary['validation']['passed'] else 'FAIL'}**
- Published structure-count agreement (663 expected): **{mismatch}** ({summary['structure_count']} found)
- Training-only cutoff criteria: **{'PASS' if cutoff['found'] else 'FAIL'}**

## Results

- Structures: {summary['structure_count']}
- Split: {summary['split']['sizes']} (`protocol-matched, split-not-identical`)
- Selected RH: {cutoff['cutoff_angstrom']} Å
- Omitted-tail matrix-element MAE floor: {cutoff['omitted_mae_mev']:.6g} meV
- Retained offsite squared Frobenius mass: {cutoff['retained_squared_fraction']:.9f}
- Source SHA256: `{source_hash}`

## Shortcomings and threats to validity

The cutoff criteria can only assess blocks present in the released sparse graph. The exact published split was not present in the inspected release, and the released structure count differs from the publication specification.

## Decision

Do not start baseline training unless all structural validations pass and the cutoff criteria find a practical radius. Preserve the split and source fingerprint for all subsequent comparisons.
"""
    (output_dir / "report.md").write_text(report)
    print("[6/6] Complete", flush=True)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "summary.json"),
                "report": str(output_dir / "report.md"),
                "hamiltonian_cutoff": cutoff,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
