#!/usr/bin/env python3
"""Fit and evaluate the native full-block ACE baseline on frozen SiO2 shards."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from e3nn import __version__ as e3nn_version
import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.envelope import build_edge_envelope, load_pair_envelope_table
from pair_descriptors import AtomicNeighborDensity
from pair_hamiltonian.metrics import FullBlockMetricAccumulator
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_hamiltonian.sio2_cache import CANONICAL_PAIR_NAMES
from pair_mappers import EquivariantRidgeAccumulator, NativeACEPairMapper

SPECIES = ("O", "Si")
ATOMIC_NUMBER_TO_SPECIES = {8: "O", 14: "Si"}


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
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--cache-summary", type=Path, required=True)
    result.add_argument("--shard-registry", type=Path, required=True)
    result.add_argument("--range-envelope", type=Path, required=True)
    result.add_argument("--core-validation-summary", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--onsite-correlation-order", type=int, required=True)
    result.add_argument("--onsite-max-degree", type=int, required=True)
    result.add_argument("--bond-radial-count", type=int, required=True)
    result.add_argument("--bond-l-max", type=int, required=True)
    result.add_argument("--bond-cutoff-angstrom", type=float, required=True)
    result.add_argument("--offsite-max-degree", type=int, required=True)
    result.add_argument("--ridge", type=float, required=True)
    result.add_argument("--batch-size", type=int, required=True)
    result.add_argument("--envelope-floor-hartree", type=float, required=True)
    result.add_argument("--distance-bin-width-angstrom", type=float, required=True)
    result.add_argument("--max-structures-per-split", type=int, default=0)
    return result


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()

    diff = run("git", "diff", "--binary", "HEAD")
    import hashlib

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def selected_registry(path: Path, maximum: int) -> dict[str, list[dict[str, str]]]:
    selected = {"train": [], "validation": [], "test": []}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in selected:
                raise ValueError(f"Unknown split {split!r} in shard registry")
            if maximum == 0 or len(selected[split]) < maximum:
                selected[split].append(row)
    if any(not rows for rows in selected.values()):
        raise ValueError("Every split must contain at least one selected shard")
    return selected


def build_model(
    cache_metadata: dict[str, object], args: argparse.Namespace, device: torch.device
) -> NativeACEPairMapper:
    orbital_config = OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"})
    transform = FullBlockIrrepTransform(
        orbital_config, dtype=torch.float32, device=device
    )
    density = AtomicNeighborDensity(
        (8, 14),
        n_radial=int(cache_metadata["density_radial_count"]),
        l_max=int(cache_metadata["density_l_max"]),
        cutoff=float(cache_metadata["descriptor_cutoff_angstrom"]),
    )
    return NativeACEPairMapper(
        transform,
        density.layout,
        onsite_correlation_order=args.onsite_correlation_order,
        onsite_max_degree=args.onsite_max_degree,
        bond_n_radial=args.bond_radial_count,
        bond_l_max=args.bond_l_max,
        bond_cutoff=args.bond_cutoff_angstrom,
        offsite_max_degree=args.offsite_max_degree,
        ridge=args.ridge,
        dtype=torch.float32,
    ).to(device=device)


def batches(indices: torch.Tensor, size: int):
    for start in range(0, indices.numel(), size):
        yield indices[start : start + size]


def load_shard(path: Path) -> dict[str, np.ndarray]:
    names = (
        "atomic_numbers",
        "descriptor",
        "onsite_target_irreps_hartree",
        "offsite_source",
        "offsite_target",
        "offsite_displacement_angstrom",
        "offsite_pair_type",
        "offsite_target_irreps_hartree",
    )
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"Incomplete cache shard: {path}")
        return {name: handle[name][:] for name in names}


def envelope_values(
    displacement: torch.Tensor,
    pair_types: torch.Tensor,
    envelope_table,
    floor_hartree: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(displacement, dim=1)
    return build_edge_envelope(distance, pair_types, envelope_table).clamp_min(
        floor_hartree
    )


@torch.no_grad()
def accumulate_training(
    model: NativeACEPairMapper,
    rows: list[dict[str, str]],
    cache_dir: Path,
    envelope_table,
    args: argparse.Namespace,
    device: torch.device,
):
    onsite = {
        species: EquivariantRidgeAccumulator(
            model.onsite_basis.irreps_out,
            model.target_transform.irreps((species, species)),
            dtype=torch.float64,
            device=device,
        )
        for species in SPECIES
    }
    offsite = {
        pair: EquivariantRidgeAccumulator(
            model.offsite_basis.irreps_out,
            model.target_transform.irreps(pair),
            dtype=torch.float64,
            device=device,
        )
        for pair in (("O", "O"), ("O", "Si"), ("Si", "Si"))
    }
    normalized_rms: dict[str, list[np.ndarray]] = {
        pair: [] for pair in CANONICAL_PAIR_NAMES
    }
    pair_count = 0
    for row in tqdm(rows, desc="fit sufficient statistics", unit="structure"):
        shard = load_shard(
            cache_dir / "shards" / f"structure_{int(row['structure_index']):04d}.h5"
        )
        descriptor = torch.from_numpy(shard["descriptor"]).to(device)
        atomic_numbers = torch.from_numpy(shard["atomic_numbers"]).to(device)
        onsite_target = torch.from_numpy(shard["onsite_target_irreps_hartree"]).to(
            device
        )
        onsite_features = model.onsite_features(descriptor)
        for atomic_number, species in ATOMIC_NUMBER_TO_SPECIES.items():
            selected = torch.nonzero(
                atomic_numbers == atomic_number, as_tuple=False
            ).flatten()
            onsite[species].update(
                onsite_features.index_select(0, selected),
                onsite_target.index_select(0, selected),
            )
        source = torch.from_numpy(shard["offsite_source"]).to(
            device=device, dtype=torch.long
        )
        target = torch.from_numpy(shard["offsite_target"]).to(
            device=device, dtype=torch.long
        )
        displacement = torch.from_numpy(shard["offsite_displacement_angstrom"]).to(
            device
        )
        pair_types = torch.from_numpy(shard["offsite_pair_type"]).to(
            device=device, dtype=torch.long
        )
        targets = torch.from_numpy(shard["offsite_target_irreps_hartree"]).to(device)
        for pair_index, pair_name in enumerate(CANONICAL_PAIR_NAMES):
            pair = tuple(pair_name.split("-"))
            selected = torch.nonzero(pair_types == pair_index, as_tuple=False).flatten()
            for batch in batches(selected, args.batch_size):
                values = envelope_values(
                    displacement.index_select(0, batch),
                    pair_types.index_select(0, batch),
                    envelope_table,
                    args.envelope_floor_hartree,
                )
                normalized_target = targets.index_select(0, batch) / values[:, None]
                features = model.offsite_features(
                    descriptor.index_select(0, source.index_select(0, batch)),
                    displacement.index_select(0, batch),
                    descriptor.index_select(0, target.index_select(0, batch)),
                )
                offsite[pair].update(features, normalized_target)
                normalized_rms[pair_name].append(
                    torch.sqrt(torch.mean(normalized_target.square(), dim=1))
                    .cpu()
                    .numpy()
                )
                pair_count += batch.numel()
    diagnostics = {"onsite": {}, "offsite": {}}
    for species, accumulator in onsite.items():
        diagnostics["onsite"][species] = asdict(
            model.fit_onsite_from_accumulator(species, accumulator)
        )
    for pair, accumulator in offsite.items():
        pair_name = "-".join(pair)
        diagnostics["offsite"][pair_name] = asdict(
            model.fit_offsite_from_accumulator(pair, accumulator)
        )
    tails = {}
    for pair_name, chunks in normalized_rms.items():
        values = np.concatenate(chunks)
        tails[pair_name] = {
            "count": int(values.size),
            "q50": float(np.quantile(values, 0.5)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "q999": float(np.quantile(values, 0.999)),
            "maximum": float(values.max()),
        }
    return diagnostics, tails, pair_count


@torch.no_grad()
def evaluate(
    model: NativeACEPairMapper,
    rows: list[dict[str, str]],
    cache_dir: Path,
    envelope_table,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], int, float]:
    metric_transform = FullBlockIrrepTransform(
        model.target_transform.orbital_config, dtype=torch.float32, device="cpu"
    )
    metrics = FullBlockMetricAccumulator(
        metric_transform,
        distance_bin_width_angstrom=args.distance_bin_width_angstrom,
    )
    evaluated_blocks = 0
    started = time.perf_counter()
    for row in tqdm(rows, desc="evaluate", unit="structure"):
        shard = load_shard(
            cache_dir / "shards" / f"structure_{int(row['structure_index']):04d}.h5"
        )
        descriptor = torch.from_numpy(shard["descriptor"]).to(device)
        atomic_numbers = torch.from_numpy(shard["atomic_numbers"]).to(device)
        onsite_target = torch.from_numpy(shard["onsite_target_irreps_hartree"]).to(
            device
        )
        for atomic_number, species in ATOMIC_NUMBER_TO_SPECIES.items():
            selected = torch.nonzero(
                atomic_numbers == atomic_number, as_tuple=False
            ).flatten()
            prediction = model.predict_onsite(
                species, descriptor.index_select(0, selected)
            )
            metrics.update(
                (species, species),
                prediction.cpu(),
                onsite_target.index_select(0, selected).cpu(),
                onsite=True,
            )
            evaluated_blocks += selected.numel()
        source = torch.from_numpy(shard["offsite_source"]).to(
            device=device, dtype=torch.long
        )
        target = torch.from_numpy(shard["offsite_target"]).to(
            device=device, dtype=torch.long
        )
        displacement = torch.from_numpy(shard["offsite_displacement_angstrom"]).to(
            device
        )
        pair_types = torch.from_numpy(shard["offsite_pair_type"]).to(
            device=device, dtype=torch.long
        )
        targets = torch.from_numpy(shard["offsite_target_irreps_hartree"]).to(device)
        for pair_index, pair_name in enumerate(CANONICAL_PAIR_NAMES):
            pair = tuple(pair_name.split("-"))
            selected = torch.nonzero(pair_types == pair_index, as_tuple=False).flatten()
            for batch in batches(selected, args.batch_size):
                batch_displacement = displacement.index_select(0, batch)
                normalized_prediction = model.predict_offsite(
                    pair,
                    descriptor.index_select(0, source.index_select(0, batch)),
                    batch_displacement,
                    descriptor.index_select(0, target.index_select(0, batch)),
                )
                values = envelope_values(
                    batch_displacement,
                    pair_types.index_select(0, batch),
                    envelope_table,
                    args.envelope_floor_hartree,
                )
                prediction = normalized_prediction * values[:, None]
                batch_target = targets.index_select(0, batch)
                distances = torch.linalg.vector_norm(batch_displacement, dim=1)
                prediction_cpu = prediction.cpu()
                target_cpu = batch_target.cpu()
                distances_cpu = distances.cpu()
                metrics.update(
                    pair,
                    prediction_cpu,
                    target_cpu,
                    onsite=False,
                    distances_angstrom=distances_cpu,
                )
                reverse_pair = (pair[1], pair[0])
                metrics.update(
                    reverse_pair,
                    metric_transform.reverse(pair, prediction_cpu),
                    metric_transform.reverse(pair, target_cpu),
                    onsite=False,
                    distances_angstrom=distances_cpu,
                )
                evaluated_blocks += 2 * batch.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return metrics.compute(), evaluated_blocks, time.perf_counter() - started


def flatten_metrics(prefix: str, value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_metrics(f"{prefix}.{key}" if prefix else key, child)
    elif isinstance(value, (int, float)):
        yield {"metric": prefix, "value": value}


def main() -> None:
    args = parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if (
        min(
            args.onsite_correlation_order,
            args.onsite_max_degree,
            args.bond_radial_count,
            args.bond_cutoff_angstrom,
            args.offsite_max_degree,
            args.batch_size,
            args.envelope_floor_hartree,
            args.distance_bin_width_angstrom,
        )
        <= 0
        or args.bond_l_max < 0
        or args.ridge < 0
        or args.max_structures_per_split < 0
    ):
        raise ValueError("Counts, cutoffs, batch size, and floor must be positive")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = (
        args.cache_summary,
        args.shard_registry,
        args.range_envelope,
        args.core_validation_summary,
    )
    if not cache_dir.is_dir() or any(not path.resolve().is_file() for path in required):
        raise FileNotFoundError("Cache and all frozen metadata inputs must exist")
    existing = (
        []
        if not output_dir.exists()
        else [path for path in output_dir.iterdir() if path.name != "launcher.log"]
    )
    if existing:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = (output_dir / "run.log").open("w", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cache_summary = json.loads(args.cache_summary.resolve().read_text())
    core_summary = json.loads(args.core_validation_summary.resolve().read_text())
    envelope_payload = json.loads(args.range_envelope.resolve().read_text())
    if not cache_summary["passed"] or not core_summary["passed"]:
        raise ValueError("Cache and native core validation must both pass")
    if envelope_payload["content_hash"] != cache_summary["range_envelope_hash"]:
        raise ValueError("Envelope hash does not match the frozen cache summary")
    cache_metadata = cache_summary["cache_metadata"]
    envelope_table = load_pair_envelope_table(
        args.range_envelope.resolve(),
        pair_order=CANONICAL_PAIR_NAMES,
        dtype=torch.float32,
        device=device,
    )
    registry = selected_registry(
        args.shard_registry.resolve(), args.max_structures_per_split
    )
    configuration = {
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git": git_state(),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "e3nn": e3nn_version,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": str(device),
            "name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "frozen_hashes": {
            "dataset": cache_metadata["dataset_sha256"],
            "split": cache_metadata["split_hash"],
            "envelope": envelope_payload["content_hash"],
            "target_schemas": cache_metadata["target_schema_hashes"],
            "onsite_features": core_summary["onsite_feature_layout_hash"],
            "offsite_features": core_summary["offsite_feature_layout_hash"],
        },
        "selected_structure_counts": {
            split: len(rows) for split, rows in registry.items()
        },
    }
    atomic_json(output_dir / "config.json", configuration)
    print(json.dumps(configuration, indent=2, sort_keys=True), flush=True)
    print("[1/6] Constructing native ACE model and streaming accumulators", flush=True)
    model = build_model(cache_metadata, args, device)
    if (
        model.onsite_basis.layout.content_hash
        != core_summary["onsite_feature_layout_hash"]
    ):
        raise ValueError("Onsite ACE layout differs from validated core")
    if (
        model.offsite_basis.layout.content_hash
        != core_summary["offsite_feature_layout_hash"]
    ):
        raise ValueError("Offsite ACE layout differs from validated core")
    started = time.perf_counter()
    diagnostics, normalized_tails, fitted_pairs = accumulate_training(
        model, registry["train"], cache_dir, envelope_table, args, device
    )
    fit_seconds = time.perf_counter() - started
    print("[2/6] Saving fitted closed-form model", flush=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "configuration": configuration,
        "fit_diagnostics": diagnostics,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    print("[3/6] Evaluating train/validation/test physical metrics", flush=True)
    evaluations = {}
    evaluated_counts = {}
    evaluation_seconds = {}
    for split in ("train", "validation", "test"):
        values, count, elapsed = evaluate(
            model, registry[split], cache_dir, envelope_table, args, device
        )
        evaluations[split] = values
        evaluated_counts[split] = count
        evaluation_seconds[split] = elapsed
    print("[4/6] Checking acceptance and computational profile", flush=True)
    all_finite = all(
        np.isfinite(row["value"])
        for split in evaluations.values()
        for row in flatten_metrics("", split)
    )
    full_run = args.max_structures_per_split == 0
    passed = all_finite and full_run
    fitted_coefficients = sum(
        buffer.numel() for name, buffer in model.named_buffers() if "weight_" in name
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    summary = {
        "passed": passed,
        "full_run": full_run,
        "headline_test_matrix_mae_mev": evaluations["test"]["matrix_elements"]["mae"],
        "evaluations": evaluations,
        "fit_diagnostics": diagnostics,
        "normalized_target_rms": normalized_tails,
        "fitted_canonical_offsite_pairs": fitted_pairs,
        "evaluated_directed_block_counts": evaluated_counts,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_blocks_per_second": {
            split: evaluated_counts[split] / evaluation_seconds[split]
            for split in evaluations
        },
        "fitted_coefficient_count": fitted_coefficients,
        "peak_cuda_memory_bytes": peak_memory,
        "frozen_hashes": configuration["frozen_hashes"],
    }
    print("[5/6] Writing metrics and report", flush=True)
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "metric", "value"))
        writer.writeheader()
        for split, values in evaluations.items():
            for row in flatten_metrics("", values):
                writer.writerow({"split": split, **row})
    (output_dir / "report.md").write_text(
        "# Native ACE SiO2 baseline\n\n"
        f"Acceptance: **{'PASS' if passed else 'SMOKE-ONLY/FAIL'}**\n\n"
        f"Headline test matrix-element MAE: {summary['headline_test_matrix_mae_mev']:.6g} meV.\n\n"
        f"Fitted coefficients: {fitted_coefficients}; peak CUDA memory: {peak_memory / 2**30:.3f} GiB.\n\n"
        "The model is pair-local, predicts one complete block-irrep vector, uses exact canonical reversal, and applies the frozen training-only range envelope.\n"
    )
    print("[6/6] Complete", flush=True)
    print(
        json.dumps(
            {
                "passed": passed,
                "full_run": full_run,
                "headline_test_matrix_mae_mev": summary["headline_test_matrix_mae_mev"],
                "fit_seconds": fit_seconds,
                "evaluation_blocks_per_second": summary["evaluation_blocks_per_second"],
                "fitted_coefficient_count": fitted_coefficients,
                "peak_cuda_memory_bytes": peak_memory,
                "summary": str(output_dir / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not all_finite:
        raise RuntimeError("Native ACE baseline produced non-finite metrics")


if __name__ == "__main__":
    main()
