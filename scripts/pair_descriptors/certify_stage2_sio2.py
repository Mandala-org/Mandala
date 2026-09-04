#!/usr/bin/env python3
"""Certify D1--D4 symmetry, cache integrity, puncturing, and storage gates."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from e3nn import o3
import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pair_descriptors import (
    DeterministicCovariantLiftDescriptor,
    FourierBesselDescriptor,
    IrreducibleMomentDescriptor,
    RawNeighborDensityDescriptor,
)

TARGET_IRREPS = ((0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (4, 1))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--family",
        action="append",
        nargs=3,
        metavar=("NAME", "CACHE_DIR", "ARTIFACT_DIR"),
        required=True,
    )
    result.add_argument("--normalization-summary", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--num-workers", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--symmetry-tolerance", type=float, required=True)
    result.add_argument("--permutation-tolerance", type=float, required=True)
    result.add_argument("--puncture-tolerance", type=float, required=True)
    result.add_argument("--float32-tolerance", type=float, required=True)
    result.add_argument(
        "--pbc-displacement-tolerance-angstrom", type=float, required=True
    )
    result.add_argument("--storage-budget-bytes", type=int, required=True)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_shard(payload: tuple) -> dict[str, object]:
    path_text, expected_hash, expected_index, expected_key, descriptor_keys = payload
    path = Path(path_text)
    actual_hash = sha256(path)
    with h5py.File(path, "r") as handle:
        complete = bool(handle.attrs["complete"])
        structure_index = int(handle.attrs["structure_index"])
        source_key = int(handle.attrs["source_key"])
        keys = set(handle["descriptors"].keys())
    return {
        "valid": complete
        and actual_hash == expected_hash
        and structure_index == expected_index
        and source_key == expected_key
        and keys == set(descriptor_keys),
        "size_bytes": path.stat().st_size,
    }


def make_descriptor(family: str, config: dict[str, object]):
    if family == "d1":
        return RawNeighborDensityDescriptor(
            tuple(config["species"]),
            radial_basis=config["radial_basis"],
            n_radial=int(config["n_radial"]),
            l_max=int(config["l_max"]),
            cutoff=float(config["cutoff_angstrom"]),
        )
    if family == "d2":
        return IrreducibleMomentDescriptor(
            tuple(config["species"]),
            max_degree=int(config["max_degree"]),
            cutoff=float(config["cutoff_angstrom"]),
        )
    if family == "d3":
        return FourierBesselDescriptor(
            tuple(config["species"]),
            frequency_count=int(config["frequency_count"]),
            l_max=int(config["l_max"]),
            cutoff=float(config["cutoff_angstrom"]),
        )
    if family == "d4":
        return DeterministicCovariantLiftDescriptor(
            tuple(config["species"]),
            radial_basis=config["radial_basis"],
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
    raise ValueError(f"Unknown family: {family}")


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(actual - expected)) / max(
        float(torch.linalg.vector_norm(expected)), 1e-30
    )


def transform(values: torch.Tensor, irreps, matrix: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(values)
    offset = 0
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actions = {str(irrep): irrep.D_from_matrix(matrix) for _, irrep in irreps}
    finally:
        torch.set_default_dtype(previous)
    for multiplicity, irrep in irreps:
        for _copy in range(multiplicity):
            stop = offset + irrep.dim
            result[..., offset:stop] = values[..., offset:stop] @ actions[str(irrep)].T
            offset = stop
    return result


def certify_descriptor(
    family: str, config: dict[str, object], seed: int
) -> dict[str, object]:
    torch.manual_seed(seed)
    descriptor = make_descriptor(family, config).to(dtype=torch.float64)
    directions = torch.randn(6, 3, dtype=torch.float64)
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    radii = torch.linspace(0.18, 0.62, 6, dtype=torch.float64) * descriptor.cutoff
    vectors = directions * radii[:, None]
    species = torch.tensor([8, 14, 8, 14, 8, 14])
    reference = descriptor(vectors, species)
    repeated = descriptor(vectors, species)
    proper = o3.rand_matrix(dtype=torch.float64)
    improper = -o3.rand_matrix(dtype=torch.float64)
    proper_error = relative_error(
        descriptor(vectors @ proper.T, species),
        transform(reference, descriptor.irreps_out, proper),
    )
    improper_error = relative_error(
        descriptor(vectors @ improper.T, species),
        transform(reference, descriptor.irreps_out, improper),
    )
    permutation = torch.tensor([4, 2, 0, 5, 3, 1])
    permutation_error = relative_error(
        descriptor(vectors[permutation], species[permutation]), reference
    )
    punctured = descriptor.puncture(reference, species[:1], vectors[:1])
    puncture_error = relative_error(punctured, descriptor(vectors[1:], species[1:]))
    return {
        "family": family,
        "key": config["key"],
        "dimension": descriptor.irreps_out.dim,
        "proper_o3_relative_error": proper_error,
        "improper_o3_relative_error": improper_error,
        "permutation_relative_error": permutation_error,
        "puncture_relative_error": puncture_error,
        "deterministic_recompute_bitwise": torch.equal(reference, repeated),
        "finite": bool(torch.isfinite(reference).all()),
        "metadata_hash_matches": descriptor.metadata["content_hash"]
        == config["content_hash"],
        "d4_retains_d1": family != "d4"
        or torch.equal(
            reference[..., : descriptor.base.irreps_out.dim],
            descriptor.base(vectors, species),
        ),
    }


def main() -> None:
    args = parser().parse_args()
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        key: (
            [list(item) for item in value]
            if key == "family"
            else str(value.resolve()) if isinstance(value, Path) else value
        )
        for key, value in vars(args).items()
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n"
    )
    families = []
    hash_tasks = []
    dataset_hashes, split_hashes = set(), set()
    persistent_bytes = 0
    for name, cache_text, artifact_text in args.family:
        cache_dir, artifact_dir = Path(cache_text), Path(artifact_text)
        summary = json.loads((artifact_dir / "summary.json").read_text())
        config = json.loads((artifact_dir / "config.json").read_text())
        schemas = json.loads((artifact_dir / "descriptor_schemas.json").read_text())
        rows = list(csv.DictReader((artifact_dir / "shards.csv").open()))
        keys = [schema["key"] for schema in schemas]
        if len(rows) != 630 or len({row["structure_index"] for row in rows}) != 630:
            raise ValueError(f"Invalid registry accounting for {name}")
        for row in rows:
            index = int(row["structure_index"])
            hash_tasks.append(
                (
                    str(cache_dir / "shards" / f"structure_{index:04d}.h5"),
                    row["sha256"],
                    index,
                    int(row["source_key"]),
                    keys,
                )
            )
        persistent_bytes += int(summary["cache_size_bytes"])
        dataset_hashes.add(summary["dataset_hash"])
        split_hashes.add(summary["split_hash"])
        families.append((name, summary, config, schemas))
    normalization = json.loads(args.normalization_summary.read_text())
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        shard_results = list(pool.map(_check_shard, hash_tasks))
    metrics = []
    for family_index, (name, _summary, _config, schemas) in enumerate(families):
        for config_index, schema in enumerate(schemas):
            metrics.append(
                certify_descriptor(
                    name, schema, args.seed + family_index * 1000 + config_index
                )
            )
    with (args.output_dir / "descriptor_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    max_symmetry = max(
        max(row["proper_o3_relative_error"], row["improper_o3_relative_error"])
        for row in metrics
    )
    max_permutation = max(row["permutation_relative_error"] for row in metrics)
    max_puncture = max(row["puncture_relative_error"] for row in metrics)
    max_float32 = max(
        float(summary["max_float32_cast_relative_error"])
        for _, summary, _, _ in families
    )
    d1_summary = next(summary for name, summary, _, _ in families if name == "d1")
    acceptance = {
        "source_summaries_pass": all(
            summary["passed"] and summary["full_run"] for _, summary, _, _ in families
        ),
        "dataset_and_split_hashes_agree": len(dataset_hashes) == len(split_hashes) == 1,
        "all_registered_shards_valid": all(row["valid"] for row in shard_results),
        "all_metadata_hashes_match": all(
            row["metadata_hash_matches"] for row in metrics
        ),
        "all_values_finite": all(row["finite"] for row in metrics),
        "deterministic_recompute": all(
            row["deterministic_recompute_bitwise"] for row in metrics
        ),
        "d4_retains_d1": all(row["d4_retains_d1"] for row in metrics),
        "o3_symmetry": max_symmetry <= args.symmetry_tolerance,
        "permutation": max_permutation <= args.permutation_tolerance,
        "puncturing": max_puncture <= args.puncture_tolerance,
        "float32_storage": max_float32 <= args.float32_tolerance,
        "pbc_remapping": float(
            d1_summary["max_regenerated_displacement_error_angstrom"]
        )
        <= args.pbc_displacement_tolerance_angstrom,
        "train_only_normalization": bool(
            normalization["passed"] and normalization["train_only"]
        ),
        "storage_budget": persistent_bytes < args.storage_budget_bytes,
    }
    family_resources = {
        name: {
            "configuration_count": int(summary["descriptor_configuration_count"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "cache_size_bytes": int(summary["cache_size_bytes"]),
            "bytes_per_atom": float(summary["cache_size_bytes"])
            / float(summary["atom_count"]),
            "descriptor_atom_configurations_per_second": float(summary["atom_count"])
            * float(summary["descriptor_configuration_count"])
            / max(float(summary["elapsed_seconds"]), 1e-30),
        }
        for name, summary, _, _ in families
    }
    summary = {
        "passed": all(acceptance.values()),
        "acceptance": acceptance,
        "descriptor_configuration_count": len(metrics),
        "family_count": len(families),
        "structure_count_per_family": 630,
        "atom_count_per_family": 21561,
        "max_o3_relative_error": max_symmetry,
        "max_permutation_relative_error": max_permutation,
        "max_puncture_relative_error": max_puncture,
        "max_float32_cast_relative_error": max_float32,
        "max_pbc_displacement_error_angstrom": d1_summary[
            "max_regenerated_displacement_error_angstrom"
        ],
        "persistent_cache_bytes": persistent_bytes,
        "storage_budget_bytes": args.storage_budget_bytes,
        "family_resources": family_resources,
        "dataset_hash": next(iter(dataset_hashes)),
        "split_hash": next(iter(split_hashes)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Stage 2 SiO2 descriptor certification",
        "",
        f"Overall acceptance: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "| Gate | Result |",
        "|---|---:|",
        *[
            f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
            for name, passed in acceptance.items()
        ],
        "",
        f"Configurations: {len(metrics)}; persistent cache: {persistent_bytes / 2**30:.3f} GiB.",
        f"Worst O(3) relative error: {max_symmetry:.3e}; puncture: {max_puncture:.3e}.",
        "",
        "Per-family bytes/atom and measured precompute throughput are recorded in `summary.json`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
