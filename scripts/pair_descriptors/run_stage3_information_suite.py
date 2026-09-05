#!/usr/bin/env python3
"""Run geometry-only reconstruction, rank, continuation, and collision studies."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np
import torch
from tqdm.auto import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pair_descriptors import (
    DeterministicCovariantLiftDescriptor,
    FourierBesselDescriptor,
    IrreducibleMomentDescriptor,
    RawNeighborDensityDescriptor,
)
from pair_descriptors.information_certification import (
    adversarial_collision_search,
    descriptor_jacobian,
    expanded_irrep_scales,
    null_direction_continuation,
    reconstruct_multistart,
)


TARGET_IRREPS = ((0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (4, 1))


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
    result.add_argument("--d1-cache-dir", type=Path, required=True)
    result.add_argument("--d1-registry", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--num-workers", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--neighbor-counts", type=int, nargs="+", required=True)
    result.add_argument("--synthetic-repeats", type=int, required=True)
    result.add_argument("--real-case-count", type=int, required=True)
    result.add_argument("--real-neighbor-count", type=int, required=True)
    result.add_argument("--inverse-starts", type=int, required=True)
    result.add_argument("--inverse-steps", type=int, required=True)
    result.add_argument("--inverse-polish-steps", type=int, required=True)
    result.add_argument("--inverse-learning-rate", type=float, required=True)
    result.add_argument("--collision-starts", type=int, required=True)
    result.add_argument("--collision-steps", type=int, required=True)
    result.add_argument("--collision-learning-rate", type=float, required=True)
    result.add_argument(
        "--collision-minimum-geometry-rms-angstrom", type=float, required=True
    )
    result.add_argument("--continuation-steps", type=int, required=True)
    result.add_argument("--continuation-learning-rate", type=float, required=True)
    result.add_argument("--descriptor-match-tolerance", type=float, required=True)
    result.add_argument(
        "--reconstruction-rmsd-tolerance-angstrom", type=float, required=True
    )
    result.add_argument("--rank-relative-tolerance", type=float, required=True)
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
    if family == "d4":
        return DeterministicCovariantLiftDescriptor(
            species,
            radial_basis=config["radial_basis"],
            n_radial=int(config["n_radial"]),
            l_max=int(config["l_max"]),
            cutoff=cutoff,
            body_degree=int(config["body_degree"]),
            lift_max_input_l=int(config["lift_max_input_l"]),
            lift_max_radial_index=int(config["lift_max_radial_index"]),
            lift_radial_degree_budget=int(config["lift_radial_degree_budget"]),
            lift_max_intermediate_l=int(config["lift_max_intermediate_l"]),
            target_irreps=TARGET_IRREPS,
            max_paths_per_degree_irrep=int(config["max_paths_per_degree_irrep"]),
        )
    raise ValueError(f"Unknown descriptor family: {family}")


def synthetic_case(cutoff: float, neighbors: int, seed: int) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    for _attempt in range(10000):
        directions = generator.normal(size=(neighbors, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = generator.uniform(0.18 * cutoff, 0.72 * cutoff, size=neighbors)
        coordinates = directions * radii[:, None]
        if (
            neighbors < 2
            or np.min(
                np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
                + np.eye(neighbors) * cutoff
            )
            > 0.08 * cutoff
        ):
            break
    else:
        raise RuntimeError("Failed to construct separated synthetic environment")
    species = np.array([8 if index % 2 == 0 else 14 for index in range(neighbors)])
    generator.shuffle(species)
    return {
        "kind": "synthetic",
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
        for center in range(len(atomic_numbers)):
            keep = np.nonzero(
                (centers == center)
                & (np.linalg.norm(displacement, axis=1) < 0.72 * cutoff)
            )[0]
            if len(keep) < args.real_neighbor_count:
                continue
            order = keep[
                np.argsort(np.linalg.norm(displacement[keep], axis=1))[
                    : args.real_neighbor_count
                ]
            ]
            output.append(
                {
                    "kind": "real_subset",
                    "coordinates_angstrom": displacement[order].tolist(),
                    "species": atomic_numbers[neighbors[order]].astype(int).tolist(),
                    "neighbor_count": args.real_neighbor_count,
                    "source": f"structure_{index:04d}_center_{center}",
                }
            )
            if len(output) == args.real_case_count:
                return output
    raise RuntimeError(
        f"Could not extract {args.real_case_count} real cases at cutoff {cutoff}"
    )


def _initialize_worker() -> None:
    torch.set_num_threads(1)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run_task(payload: dict[str, object]) -> dict[str, object]:
    destination = Path(payload["destination"])
    descriptor = make_descriptor(payload["family"], payload["config"]).to(
        dtype=torch.float64
    )
    coordinates = torch.tensor(
        payload["case"]["coordinates_angstrom"], dtype=torch.float64
    )
    species = torch.tensor(payload["case"]["species"], dtype=torch.long)
    channel_rms = payload["channel_rms"]
    scales = expanded_irrep_scales(descriptor.irreps_out, channel_rms)
    jacobian = descriptor_jacobian(descriptor, coordinates, species, scales)
    singular = torch.linalg.svdvals(jacobian)
    rank_threshold = max(float(singular[0]) * payload["rank_relative_tolerance"], 1e-12)
    rank = int(torch.count_nonzero(singular > rank_threshold))
    result = {
        "manifest_hash": payload["manifest_hash"],
        "family": payload["family"],
        "key": payload["config"]["key"],
        "case_id": payload["case_id"],
        "case_kind": payload["case"]["kind"],
        "neighbor_count": len(coordinates),
        "dimension": descriptor.irreps_out.dim,
        "coordinate_dimension": coordinates.numel(),
        "jacobian_rank": rank,
        "jacobian_rank_deficiency": coordinates.numel() - rank,
        "jacobian_largest_singular_value": float(singular[0]),
        "jacobian_smallest_singular_value": float(singular[-1]),
        "jacobian_condition_number": float(
            singular[0] / singular[-1].clamp_min(1e-300)
        ),
    }
    continuation = null_direction_continuation(
        descriptor,
        coordinates,
        species,
        scales,
        jacobian,
        steps=payload["continuation_steps"],
        learning_rate=payload["continuation_learning_rate"],
    )
    result.update({f"continuation_{key}": value for key, value in continuation.items()})
    if payload["run_inverse"]:
        inverse = reconstruct_multistart(
            descriptor,
            coordinates,
            species,
            scales,
            starts=payload["inverse_starts"],
            steps=payload["inverse_steps"],
            polish_steps=payload["inverse_polish_steps"],
            learning_rate=payload["inverse_learning_rate"],
            seed=payload["task_seed"],
        )
        result.update(
            {
                "inverse_normalized_descriptor_rms": inverse.normalized_descriptor_rms,
                "inverse_assigned_cartesian_rmsd_angstrom": inverse.assigned_cartesian_rmsd_angstrom,
                "inverse_pair_distance_rmsd_angstrom": inverse.pair_distance_rmsd_angstrom,
            }
        )
        collision = adversarial_collision_search(
            descriptor,
            coordinates,
            species,
            scales,
            starts=payload["collision_starts"],
            steps=payload["collision_steps"],
            learning_rate=payload["collision_learning_rate"],
            minimum_geometry_rms=payload["collision_minimum_geometry_rms_angstrom"],
            seed=payload["task_seed"] + 1_000_000,
        )
        result.update({f"collision_{key}": value for key, value in collision.items()})
    _atomic_json(destination, result)
    return result


def negative_control(config: dict[str, object]) -> dict[str, float | bool]:
    descriptor = make_descriptor("d1", config).to(dtype=torch.float64)
    radii = torch.tensor([1.0, 1.4, 1.8, 2.2], dtype=torch.float64)
    first = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 1.8], [-2.2, 0.0, 0.0]],
        dtype=torch.float64,
    )
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.4, 0.916515, 0.0],
            [-0.2, 0.3, 0.932738],
            [0.1, -0.7, 0.707107],
        ],
        dtype=torch.float64,
    )
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    second = directions * radii[:, None]
    species = torch.tensor([8, 14, 8, 14])
    radial_signature_equal = bool(
        torch.max(torch.abs(radii - torch.linalg.vector_norm(second, dim=-1))) < 1e-12
    )
    separation = float(
        torch.linalg.vector_norm(
            descriptor(first, species) - descriptor(second, species)
        )
    )
    pair_rms = float(
        torch.sqrt(
            torch.mean(
                (
                    torch.sort(torch.pdist(first))[0]
                    - torch.sort(torch.pdist(second))[0]
                ).square()
            )
        )
    )
    return {
        "radial_only_invariant_collision": radial_signature_equal,
        "covariant_descriptor_separation": separation,
        "pair_distance_rmsd_angstrom": pair_rms,
        "negative_control_passed": radial_signature_equal
        and separation > 1e-6
        and pair_rms > 1e-3,
    }


def main() -> None:
    args = parser().parse_args()
    if args.num_workers <= 0 or args.synthetic_repeats <= 0:
        raise ValueError("Worker and repeat counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = args.output_dir / "tasks"
    task_dir.mkdir(exist_ok=True)
    normalization = json.loads(args.normalization_json.read_text())
    schemas_by_family = {
        name: json.loads(Path(path).read_text()) for name, path in args.family
    }
    normalizations = {
        (family, descriptor["key"]): [
            channel["rms"] for channel in descriptor["channels"]
        ]
        for family, family_value in normalization["families"].items()
        for descriptor in family_value["descriptors"]
    }
    cutoffs = sorted(
        {
            float(config["cutoff_angstrom"])
            for schemas in schemas_by_family.values()
            for config in schemas
        }
    )
    cases_by_cutoff = {}
    for cutoff_index, cutoff in enumerate(cutoffs):
        cases = []
        for neighbors in args.neighbor_counts:
            for repeat in range(args.synthetic_repeats):
                cases.append(
                    synthetic_case(
                        cutoff,
                        neighbors,
                        args.seed + cutoff_index * 1000 + neighbors * 10 + repeat,
                    )
                )
        cases.extend(real_cases(args, cutoff))
        cases_by_cutoff[cutoff] = cases
    concise_cases = {
        str(cutoff): [
            dict(case, coordinates_angstrom=case["coordinates_angstrom"])
            for case in cases
        ]
        for cutoff, cases in cases_by_cutoff.items()
    }
    _atomic_json(args.output_dir / "cases.json", concise_cases)
    settings = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"output_dir", "resume"}
    }
    manifest_seed = {
        "families": {
            name: [config["content_hash"] for config in schemas]
            for name, schemas in schemas_by_family.items()
        },
        "normalization_hash": normalization["content_hash"],
        "cases": concise_cases,
        "settings": settings,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_seed, sort_keys=True, default=str).encode()
    ).hexdigest()
    _atomic_json(
        args.output_dir / "config.json",
        {**manifest_seed, "manifest_hash": manifest_hash},
    )
    common = {
        "manifest_hash": manifest_hash,
        "rank_relative_tolerance": args.rank_relative_tolerance,
        "inverse_starts": args.inverse_starts,
        "inverse_steps": args.inverse_steps,
        "inverse_polish_steps": args.inverse_polish_steps,
        "inverse_learning_rate": args.inverse_learning_rate,
        "collision_starts": args.collision_starts,
        "collision_steps": args.collision_steps,
        "collision_learning_rate": args.collision_learning_rate,
        "collision_minimum_geometry_rms_angstrom": args.collision_minimum_geometry_rms_angstrom,
        "continuation_steps": args.continuation_steps,
        "continuation_learning_rate": args.continuation_learning_rate,
    }
    tasks, completed = [], []
    task_index = 0
    for family, schemas in schemas_by_family.items():
        for config in schemas:
            cases = cases_by_cutoff[float(config["cutoff_angstrom"])]
            selected_cases = (
                cases if family != "d4" else [cases[len(cases) // 2], cases[-1]]
            )
            for case_index, case in enumerate(selected_cases):
                case_id = f"{case['kind']}_{case_index}_{case['source']}"
                destination = task_dir / f"task_{task_index:05d}.json"
                payload = {
                    **common,
                    "family": family,
                    "config": config,
                    "channel_rms": normalizations[(family, config["key"])],
                    "case": case,
                    "case_id": case_id,
                    "task_seed": args.seed + task_index * 17,
                    "run_inverse": family != "d4",
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
        f"Running {len(tasks)} new Stage-3 tasks; {len(completed)} resumed", flush=True
    )
    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_initialize_worker
    ) as pool:
        completed.extend(
            tqdm(pool.map(_run_task, tasks), total=len(tasks), unit="task")
        )
    completed.sort(key=lambda row: (row["family"], row["key"], row["case_id"]))
    scalar_keys = sorted(
        {
            key
            for row in completed
            for key, value in row.items()
            if isinstance(value, (str, int, float, bool))
        }
    )
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in completed:
        groups.setdefault((row["family"], row["key"]), []).append(row)
    configuration_metrics = []
    for (family, key), rows in sorted(groups.items()):
        inverse_rows = [
            row for row in rows if "inverse_normalized_descriptor_rms" in row
        ]
        successes = [
            row
            for row in inverse_rows
            if row["inverse_normalized_descriptor_rms"]
            <= args.descriptor_match_tolerance
            and row["inverse_assigned_cartesian_rmsd_angstrom"]
            <= args.reconstruction_rmsd_tolerance_angstrom
        ]
        candidate_collisions = [
            row
            for row in inverse_rows
            if row["collision_normalized_descriptor_rms"]
            <= args.descriptor_match_tolerance
            and row["collision_geometry_signature_rms_angstrom"]
            >= args.collision_minimum_geometry_rms_angstrom
        ]
        configuration_metrics.append(
            {
                "family": family,
                "key": key,
                "dimension": rows[0]["dimension"],
                "case_count": len(rows),
                "reconstruction_success_fraction": (
                    len(successes) / len(inverse_rows) if inverse_rows else ""
                ),
                "median_inverse_descriptor_rms": (
                    float(
                        np.median(
                            [
                                row["inverse_normalized_descriptor_rms"]
                                for row in inverse_rows
                            ]
                        )
                    )
                    if inverse_rows
                    else ""
                ),
                "median_inverse_cartesian_rmsd_angstrom": (
                    float(
                        np.median(
                            [
                                row["inverse_assigned_cartesian_rmsd_angstrom"]
                                for row in inverse_rows
                            ]
                        )
                    )
                    if inverse_rows
                    else ""
                ),
                "rank_deficiency_fraction": sum(
                    row["jacobian_rank_deficiency"] > 0 for row in rows
                )
                / len(rows),
                "median_smallest_singular_value": float(
                    np.median([row["jacobian_smallest_singular_value"] for row in rows])
                ),
                "median_condition_number": float(
                    np.median([row["jacobian_condition_number"] for row in rows])
                ),
                "candidate_collision_count": len(candidate_collisions),
            }
        )
    with (args.output_dir / "configuration_metrics.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(configuration_metrics[0]))
        writer.writeheader()
        writer.writerows(configuration_metrics)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family in ("d1", "d2", "d3"):
        selected = [row for row in configuration_metrics if row["family"] == family]
        axes[0].scatter(
            [row["dimension"] for row in selected],
            [row["reconstruction_success_fraction"] for row in selected],
            label=family.upper(),
        )
        axes[1].scatter(
            [row["dimension"] for row in selected],
            [row["median_smallest_singular_value"] for row in selected],
            label=family.upper(),
        )
    axes[0].set(
        xlabel="descriptor dimension",
        ylabel="reconstruction success fraction",
        ylim=(-0.03, 1.03),
    )
    axes[1].set(
        xlabel="descriptor dimension",
        ylabel="median smallest normalized-Jacobian singular value",
        yscale="log",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "basis_convergence.png", dpi=180)
    plt.close(figure)
    additive = [row for row in completed if row["family"] != "d4"]
    reconstructed = [
        row
        for row in additive
        if row["inverse_normalized_descriptor_rms"] <= args.descriptor_match_tolerance
        and row["inverse_assigned_cartesian_rmsd_angstrom"]
        <= args.reconstruction_rmsd_tolerance_angstrom
    ]
    collisions = [
        row
        for row in additive
        if row["collision_normalized_descriptor_rms"] <= args.descriptor_match_tolerance
        and row["collision_geometry_signature_rms_angstrom"]
        >= args.collision_minimum_geometry_rms_angstrom
    ]
    deficient = [row for row in completed if row["jacobian_rank_deficiency"] > 0]
    first_d1 = schemas_by_family["d1"][0]
    control = negative_control(first_d1)
    summary = {
        "completed": True,
        "manifest_hash": manifest_hash,
        "task_count": len(completed),
        "additive_inverse_task_count": len(additive),
        "reconstruction_success_count": len(reconstructed),
        "reconstruction_success_fraction": len(reconstructed) / max(len(additive), 1),
        "unexpected_rank_deficiency_count": len(deficient),
        "unexpected_rank_deficiency_fraction": len(deficient) / max(len(completed), 1),
        "candidate_collision_count": len(collisions),
        "negative_control": control,
        "selection_uses_hamiltonian": False,
        "descriptor_match_tolerance": args.descriptor_match_tolerance,
        "reconstruction_rmsd_tolerance_angstrom": args.reconstruction_rmsd_tolerance_angstrom,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(
        "# Stage 3 descriptor-information suite\n\n"
        "This batch is geometry-only and does not read Hamiltonian targets.\n\n"
        f"Tasks completed: {len(completed)}. Reconstruction success: {len(reconstructed)}/{len(additive)}. "
        f"Rank-deficient cases: {len(deficient)}. Candidate collisions: {len(collisions)}.\n\n"
        "These are gate inputs, not an automatic completeness claim or promotion decision.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
