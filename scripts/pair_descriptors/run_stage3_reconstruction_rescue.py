#!/usr/bin/env python3
"""Re-run Stage-3 inversion with constructive l=0/l=1 initialization."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pair_descriptors import (
    FourierBesselDescriptor,
    IrreducibleMomentDescriptor,
    RawNeighborDensityDescriptor,
)
from pair_descriptors.information_certification import (
    constructive_l0_l1_initializer,
    expanded_irrep_scales,
    pair_distance_rmsd,
    reconstruct_multistart,
    species_assigned_rmsd,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--family",
        action="append",
        nargs=2,
        metavar=("NAME", "SCHEMA_JSON"),
        required=True,
    )
    result.add_argument("--normalization-json", type=Path, required=True)
    result.add_argument("--cases-json", type=Path, required=True)
    result.add_argument("--config-key", action="append", default=[])
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--num-workers", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--radial-starts", type=int, required=True)
    result.add_argument("--inverse-starts", type=int, required=True)
    result.add_argument("--inverse-steps", type=int, required=True)
    result.add_argument("--inverse-polish-steps", type=int, required=True)
    result.add_argument("--inverse-polish-candidates", type=int, required=True)
    result.add_argument("--inverse-learning-rate", type=float, required=True)
    result.add_argument("--descriptor-match-tolerance", type=float, required=True)
    result.add_argument(
        "--reconstruction-rmsd-tolerance-angstrom", type=float, required=True
    )
    result.add_argument("--resume", action="store_true")
    return result


def make_descriptor(family: str, config: dict[str, object]):
    species = tuple(config["species"])
    cutoff = float(config["cutoff_angstrom"])
    if family == "d1":
        return RawNeighborDensityDescriptor(
            species,
            radial_basis=config["radial_basis"],
            n_radial=int(config["n_radial"]),
            l_max=int(config["l_max"]),
            cutoff=cutoff,
        )
    if family == "d2":
        return IrreducibleMomentDescriptor(
            species, max_degree=int(config["max_degree"]), cutoff=cutoff
        )
    if family == "d3":
        return FourierBesselDescriptor(
            species,
            frequency_count=int(config["frequency_count"]),
            l_max=int(config["l_max"]),
            cutoff=cutoff,
        )
    raise ValueError(f"Unsupported additive family: {family}")


def _initialize_worker() -> None:
    torch.set_num_threads(1)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run(payload: dict[str, object]) -> dict[str, object]:
    descriptor = make_descriptor(payload["family"], payload["config"]).double()
    coordinates = torch.tensor(
        payload["case"]["coordinates_angstrom"], dtype=torch.float64
    )
    species = torch.tensor(payload["case"]["species"], dtype=torch.long)
    scales = expanded_irrep_scales(descriptor.irreps_out, payload["channel_rms"])
    target = descriptor(coordinates, species)[0].detach()
    initialization = None
    initialization_error = None
    try:
        initialization, radial_residual = constructive_l0_l1_initializer(
            descriptor,
            target,
            species,
            scales,
            payload["config"]["channels"],
            starts=payload["radial_starts"],
            seed=payload["task_seed"],
        )
        initial_descriptor_rms = float(
            torch.sqrt(
                torch.mean(
                    (
                        (descriptor(initialization, species)[0] - target) / scales
                    ).square()
                )
            )
        )
        initial_cartesian_rmsd = species_assigned_rmsd(
            initialization, coordinates, species
        )
        initial_pair_rmsd = pair_distance_rmsd(initialization, coordinates)
    except (ValueError, RuntimeError, torch.linalg.LinAlgError) as error:
        radial_residual = float("nan")
        initial_descriptor_rms = float("nan")
        initial_cartesian_rmsd = float("nan")
        initial_pair_rmsd = float("nan")
        initialization_error = str(error)
    initialization_passed = (
        initialization is not None
        and initial_descriptor_rms <= payload["descriptor_match_tolerance"]
        and initial_cartesian_rmsd <= payload["reconstruction_rmsd_tolerance_angstrom"]
    )
    if initialization_passed:
        inverse_descriptor_rms = initial_descriptor_rms
        inverse_cartesian_rmsd = initial_cartesian_rmsd
        inverse_pair_rmsd = initial_pair_rmsd
    else:
        component_l = torch.empty(descriptor.irreps_out.dim, dtype=torch.long)
        for channel in payload["config"]["channels"]:
            component_l[int(channel["start"]) : int(channel["stop"])] = int(
                channel["l"]
            )
        stages = tuple(
            component_l <= l_value for l_value in range(int(component_l.max()) + 1)
        )
        inverse = reconstruct_multistart(
            descriptor,
            coordinates,
            species,
            scales,
            starts=payload["inverse_starts"],
            steps=payload["inverse_steps"],
            polish_steps=payload["inverse_polish_steps"],
            polish_candidates=payload["inverse_polish_candidates"],
            learning_rate=payload["inverse_learning_rate"],
            seed=payload["task_seed"] + 10_000_000,
            stage_masks=stages,
            initial_coordinates=initialization,
        )
        inverse_descriptor_rms = inverse.normalized_descriptor_rms
        inverse_cartesian_rmsd = inverse.assigned_cartesian_rmsd_angstrom
        inverse_pair_rmsd = inverse.pair_distance_rmsd_angstrom
    result = {
        "manifest_hash": payload["manifest_hash"],
        "family": payload["family"],
        "key": payload["config"]["key"],
        "case_id": payload["case_id"],
        "case_kind": payload["case"]["kind"],
        "neighbor_count": len(coordinates),
        "dimension": descriptor.irreps_out.dim,
        "constructive_available": initialization is not None,
        "constructive_error": initialization_error or "",
        "constructive_radial_descriptor_rms": radial_residual,
        "constructive_full_descriptor_rms": initial_descriptor_rms,
        "constructive_assigned_cartesian_rmsd_angstrom": initial_cartesian_rmsd,
        "constructive_pair_distance_rmsd_angstrom": initial_pair_rmsd,
        "constructive_passed": initialization_passed,
        "optimization_run": not initialization_passed,
        "inverse_normalized_descriptor_rms": inverse_descriptor_rms,
        "inverse_assigned_cartesian_rmsd_angstrom": inverse_cartesian_rmsd,
        "inverse_pair_distance_rmsd_angstrom": inverse_pair_rmsd,
    }
    _atomic_json(Path(payload["destination"]), result)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.num_workers, args.radial_starts, args.inverse_starts) <= 0:
        raise ValueError("Worker/start counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = args.output_dir / "tasks"
    task_dir.mkdir(exist_ok=True)
    schemas = {name: json.loads(Path(path).read_text()) for name, path in args.family}
    if set(schemas) != {"d1", "d2", "d3"}:
        raise ValueError("Rescue suite requires exactly D1, D2, and D3")
    if args.config_key:
        requested = set(args.config_key)
        available = {
            config["key"]
            for configurations in schemas.values()
            for config in configurations
        }
        missing = requested - available
        if missing:
            raise ValueError(f"Unknown requested configuration keys: {sorted(missing)}")
        schemas = {
            family: [config for config in configurations if config["key"] in requested]
            for family, configurations in schemas.items()
        }
        if any(not configurations for configurations in schemas.values()):
            raise ValueError("Filtered confirmation grid must retain every family")
    normalization = json.loads(args.normalization_json.read_text())
    channel_rms = {
        (family, descriptor["key"]): [
            channel["rms"] for channel in descriptor["channels"]
        ]
        for family, family_value in normalization["families"].items()
        for descriptor in family_value["descriptors"]
        if family in schemas
    }
    cases_by_cutoff = json.loads(args.cases_json.read_text())
    manifest_seed = {
        "algorithm": "constructive-l0-l1-plus-angular-homotopy-v1",
        "descriptor_hashes": {
            family: [config["content_hash"] for config in values]
            for family, values in schemas.items()
        },
        "normalization_hash": normalization["content_hash"],
        "cases_sha256": hashlib.sha256(args.cases_json.read_bytes()).hexdigest(),
        "settings": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output_dir", "resume"}
        },
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_seed, sort_keys=True).encode()
    ).hexdigest()
    _atomic_json(
        args.output_dir / "config.json",
        {**manifest_seed, "manifest_hash": manifest_hash},
    )
    common = {
        "manifest_hash": manifest_hash,
        "radial_starts": args.radial_starts,
        "inverse_starts": args.inverse_starts,
        "inverse_steps": args.inverse_steps,
        "inverse_polish_steps": args.inverse_polish_steps,
        "inverse_polish_candidates": args.inverse_polish_candidates,
        "inverse_learning_rate": args.inverse_learning_rate,
        "descriptor_match_tolerance": args.descriptor_match_tolerance,
        "reconstruction_rmsd_tolerance_angstrom": args.reconstruction_rmsd_tolerance_angstrom,
    }
    tasks, completed = [], []
    task_index = 0
    for family, configurations in schemas.items():
        for config in configurations:
            cases = cases_by_cutoff[str(float(config["cutoff_angstrom"]))]
            for case_index, case in enumerate(cases):
                destination = task_dir / f"task_{task_index:05d}.json"
                payload = {
                    **common,
                    "family": family,
                    "config": config,
                    "channel_rms": channel_rms[(family, config["key"])],
                    "case": case,
                    "case_id": f"{case['kind']}_{case_index}_{case['source']}",
                    "task_seed": args.seed + task_index * 19,
                    "destination": str(destination),
                }
                task_index += 1
                if args.resume and destination.is_file():
                    previous = json.loads(destination.read_text())
                    if previous.get("manifest_hash") == manifest_hash:
                        completed.append(previous)
                        continue
                tasks.append(payload)
    print(
        f"Running {len(tasks)} reconstruction-rescue tasks; {len(completed)} resumed",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_initialize_worker
    ) as pool:
        completed.extend(tqdm(pool.map(_run, tasks), total=len(tasks), unit="task"))
    completed.sort(key=lambda row: (row["family"], row["key"], row["case_id"]))
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in completed:
        groups.setdefault((row["family"], row["key"]), []).append(row)
    metrics = []
    for (family, key), rows in sorted(groups.items()):

        def passed(row):
            return (
                row["inverse_normalized_descriptor_rms"]
                <= args.descriptor_match_tolerance
                and row["inverse_assigned_cartesian_rmsd_angstrom"]
                <= args.reconstruction_rmsd_tolerance_angstrom
            )

        successes = [row for row in rows if passed(row)]
        synthetic_rows = [
            row for row in rows if row["case_kind"].startswith("synthetic")
        ]
        real_rows = [row for row in rows if row["case_kind"].startswith("real")]
        metrics.append(
            {
                "family": family,
                "key": key,
                "dimension": rows[0]["dimension"],
                "case_count": len(rows),
                "constructive_available_fraction": sum(
                    row["constructive_available"] for row in rows
                )
                / len(rows),
                "reconstruction_success_fraction": len(successes) / len(rows),
                "synthetic_case_count": len(synthetic_rows),
                "synthetic_reconstruction_success_fraction": sum(
                    passed(row) for row in synthetic_rows
                )
                / max(len(synthetic_rows), 1),
                "real_case_count": len(real_rows),
                "real_reconstruction_success_fraction": sum(
                    passed(row) for row in real_rows
                )
                / max(len(real_rows), 1),
                "median_descriptor_rms": float(
                    np.median(
                        [row["inverse_normalized_descriptor_rms"] for row in rows]
                    )
                ),
                "median_cartesian_rmsd_angstrom": float(
                    np.median(
                        [
                            row["inverse_assigned_cartesian_rmsd_angstrom"]
                            for row in rows
                        ]
                    )
                ),
            }
        )
    with (args.output_dir / "configuration_metrics.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    success_count = sum(
        row["inverse_normalized_descriptor_rms"] <= args.descriptor_match_tolerance
        and row["inverse_assigned_cartesian_rmsd_angstrom"]
        <= args.reconstruction_rmsd_tolerance_angstrom
        for row in completed
    )
    synthetic_rows = [
        row for row in completed if row["case_kind"].startswith("synthetic")
    ]
    synthetic_success_count = sum(
        row["inverse_normalized_descriptor_rms"] <= args.descriptor_match_tolerance
        and row["inverse_assigned_cartesian_rmsd_angstrom"]
        <= args.reconstruction_rmsd_tolerance_angstrom
        for row in synthetic_rows
    )
    summary = {
        "completed": True,
        "manifest_hash": manifest_hash,
        "task_count": len(completed),
        "reconstruction_success_count": success_count,
        "reconstruction_success_fraction": success_count / len(completed),
        "synthetic_case_count": len(synthetic_rows),
        "synthetic_reconstruction_success_count": synthetic_success_count,
        "synthetic_reconstruction_success_fraction": synthetic_success_count
        / len(synthetic_rows),
        "constructive_available_count": sum(
            row["constructive_available"] for row in completed
        ),
        "selection_uses_hamiltonian": False,
        "descriptor_match_tolerance": args.descriptor_match_tolerance,
        "reconstruction_rmsd_tolerance_angstrom": args.reconstruction_rmsd_tolerance_angstrom,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(
        "# Stage 3 constructive reconstruction rescue\n\n"
        f"Completed {len(completed)} geometry-only tasks; strict successes: "
        f"{success_count}/{len(completed)}.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
