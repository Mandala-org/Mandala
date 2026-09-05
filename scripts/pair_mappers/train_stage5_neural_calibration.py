#!/usr/bin/env python3
"""Train one validation-only M3/M5 calibration model on frozen SiO2 caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import torch
from e3nn import o3

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.envelope import build_edge_envelope, load_pair_envelope_table
from pair_hamiltonian.metrics import FullBlockMetricAccumulator
from pair_hamiltonian.hermiticity import (
    project_directed_irreps,
    project_onsite_irreps,
)
from pair_hamiltonian.output_schema import (
    FullBlockIrrepTransform,
    o3_representation_matrix,
)
from pair_hamiltonian.range_objective import range_factored_mse
from pair_hamiltonian.sio2_cache import DIRECTED_PAIR_NAMES
from pair_mappers.neural import FullBlockNeuralPairMapper

SPECIES = ("O", "Si")
SPECIES_Z = {"O": 8, "Si": 14}
PAIRS = tuple(tuple(name.split("-")) for name in DIRECTED_PAIR_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("d1", "d2", "d3", "d4"), required=True)
    parser.add_argument("--descriptor-key", required=True)
    parser.add_argument("--descriptor-cache-dir", type=Path, required=True)
    parser.add_argument("--descriptor-summary", type=Path, required=True)
    parser.add_argument("--descriptor-schemas", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--normalization-summary", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--baseline-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--shard-registry", type=Path, required=True)
    parser.add_argument("--range-envelope", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("m3", "m5"), required=True)
    parser.add_argument(
        "--resource-band", choices=("small", "medium", "large"), required=True
    )
    parser.add_argument("--descriptor-multiplicity-cap", type=int, required=True)
    parser.add_argument("--generator-multiplicity", type=int, required=True)
    parser.add_argument("--hidden-multiplicity", type=int, required=True)
    parser.add_argument("--hidden-l-max", type=int, required=True)
    parser.add_argument("--invariant-hidden", type=int, required=True)
    parser.add_argument("--factorization-rank", type=int, required=True)
    parser.add_argument("--bond-radial-count", type=int, required=True)
    parser.add_argument("--bond-l-max", type=int, required=True)
    parser.add_argument("--bond-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument(
        "--range-loss-mode",
        choices=("physical_mse", "weighted_normalized_mse"),
        required=True,
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, required=True)
    parser.add_argument("--early-stopping-evaluations", type=int, required=True)
    parser.add_argument("--onsite-batch-size", type=int, required=True)
    parser.add_argument("--offsite-batch-size", type=int, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, required=True)
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--envelope-floor-hartree", type=float, required=True)
    parser.add_argument("--distance-bin-width-angstrom", type=float, required=True)
    parser.add_argument("--float32-symmetry-tolerance", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rows(
    path: Path, train_fraction: float, seed: int
) -> dict[str, list[dict[str, str]]]:
    selected = {"train": [], "validation": []}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in selected:
                selected[row["split"]].append(row)
    if any(not values for values in selected.values()):
        raise ValueError("train and validation partitions must be nonempty")
    rng = np.random.default_rng(seed)
    count = max(1, math.ceil(train_fraction * len(selected["train"])))
    indices = np.sort(rng.choice(len(selected["train"]), count, replace=False))
    selected["train"] = [selected["train"][int(index)] for index in indices]
    return selected


def _normalization_scale(path: Path, family: str, key: str) -> torch.Tensor:
    payload = json.loads(path.read_text())
    records = payload["families"][family]["descriptors"]
    record = next((item for item in records if item["key"] == key), None)
    if record is None:
        raise KeyError(f"normalization missing {family}/{key}")
    scale = torch.empty(int(record["dimension"]), dtype=torch.float32)
    covered = torch.zeros_like(scale, dtype=torch.bool)
    for channel in record["channels"]:
        start, stop = int(channel["start"]), int(channel["stop"])
        scale[start:stop] = float(channel["rms"])
        covered[start:stop] = True
    if (
        not torch.all(covered)
        or not torch.all(torch.isfinite(scale))
        or torch.any(scale <= 0)
    ):
        raise ValueError("invalid descriptor normalization channels")
    return scale


def _load_shard(
    baseline_cache: Path,
    descriptor_cache: Path,
    index: int,
    descriptor_key: str,
    scale: torch.Tensor,
):
    baseline_path = baseline_cache / "shards" / f"structure_{index:04d}.h5"
    descriptor_path = descriptor_cache / "shards" / f"structure_{index:04d}.h5"
    names = (
        "atomic_numbers",
        "onsite_target_irreps_hartree",
        "offsite_source",
        "offsite_target",
        "offsite_inverse",
        "offsite_displacement_angstrom",
        "offsite_pair_type",
        "offsite_target_irreps_hartree",
    )
    with h5py.File(baseline_path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete baseline shard {baseline_path}")
        result = {name: torch.from_numpy(handle[name][:]) for name in names}
    with h5py.File(descriptor_path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete descriptor shard {descriptor_path}")
        descriptor = torch.from_numpy(handle["descriptors"][descriptor_key][:])
    result["descriptor"] = descriptor / scale
    return result


def _concatenate_partition(
    rows: list[dict[str, str]],
    baseline_cache: Path,
    descriptor_cache: Path,
    descriptor_key: str,
    scale: torch.Tensor,
    envelope_table,
    envelope_floor: float,
    device: torch.device,
) -> dict[str, object]:
    descriptors = []
    onsite = {species: {"descriptor": [], "target": []} for species in SPECIES}
    offsite = {
        pair: {
            "source": [],
            "target_index": [],
            "displacement": [],
            "target": [],
            "envelope": [],
        }
        for pair in PAIRS
    }
    offsite_flat = {
        key: []
        for key in (
            "source",
            "target_index",
            "displacement",
            "target",
            "envelope",
            "pair_type",
            "inverse",
        )
    }
    atom_offset = 0
    edge_offset = 0
    for row in rows:
        shard = _load_shard(
            baseline_cache,
            descriptor_cache,
            int(row["structure_index"]),
            descriptor_key,
            scale,
        )
        descriptor = shard["descriptor"]
        descriptors.append(descriptor)
        atomic_numbers = shard["atomic_numbers"]
        for species in SPECIES:
            mask = atomic_numbers == SPECIES_Z[species]
            onsite[species]["descriptor"].append(descriptor[mask])
            onsite[species]["target"].append(
                shard["onsite_target_irreps_hartree"][mask]
            )
        pair_types = shard["offsite_pair_type"].to(torch.long)
        displacement = shard["offsite_displacement_angstrom"]
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        all_envelope = build_edge_envelope(
            distance, pair_types, envelope_table
        ).clamp_min(envelope_floor)
        offsite_flat["source"].append(
            shard["offsite_source"].to(torch.long) + atom_offset
        )
        offsite_flat["target_index"].append(
            shard["offsite_target"].to(torch.long) + atom_offset
        )
        offsite_flat["displacement"].append(displacement)
        offsite_flat["target"].append(shard["offsite_target_irreps_hartree"])
        offsite_flat["envelope"].append(all_envelope)
        offsite_flat["pair_type"].append(pair_types)
        offsite_flat["inverse"].append(
            shard["offsite_inverse"].to(torch.long) + edge_offset
        )
        for pair_index, pair in enumerate(PAIRS):
            mask = pair_types == pair_index
            pair_displacement = displacement[mask]
            envelope = all_envelope[mask]
            offsite[pair]["source"].append(
                shard["offsite_source"][mask].to(torch.long) + atom_offset
            )
            offsite[pair]["target_index"].append(
                shard["offsite_target"][mask].to(torch.long) + atom_offset
            )
            offsite[pair]["displacement"].append(pair_displacement)
            offsite[pair]["target"].append(shard["offsite_target_irreps_hartree"][mask])
            offsite[pair]["envelope"].append(envelope)
        atom_offset += descriptor.shape[0]
        edge_offset += pair_types.shape[0]
    packed: dict[str, object] = {
        "descriptor": torch.cat(descriptors).to(device),
        "onsite": {},
        "offsite": {},
        "offsite_flat": {
            key: torch.cat(chunks).to(device) for key, chunks in offsite_flat.items()
        },
    }
    for species, values in onsite.items():
        packed["onsite"][species] = {
            key: torch.cat(chunks).to(device) for key, chunks in values.items()
        }
    for pair, values in offsite.items():
        packed["offsite"][pair] = {
            key: torch.cat(chunks).to(device) for key, chunks in values.items()
        }
    return packed


def _sample(
    count: int, batch_size: int, generator: torch.Generator, device: torch.device
):
    return torch.randint(
        count, (min(count, batch_size),), generator=generator, device=device
    )


def _training_loss(
    model: FullBlockNeuralPairMapper,
    data: dict[str, object],
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    group_counts = [
        data["onsite"][species]["target"].shape[0] for species in SPECIES
    ] + [data["offsite"][pair]["target"].shape[0] for pair in PAIRS]
    total_count = sum(group_counts)
    weighted_loss = torch.zeros((), dtype=torch.float32, device=device)
    sampled = 0
    for species, group_count in zip(SPECIES, group_counts[: len(SPECIES)]):
        values = data["onsite"][species]
        indices = _sample(
            values["target"].shape[0], args.onsite_batch_size, generator, device
        )
        prediction = model.predict_onsite(
            species, values["descriptor"].index_select(0, indices)
        )
        target = values["target"].index_select(0, indices)
        weighted_loss = weighted_loss + (group_count / total_count) * torch.mean(
            (prediction - target).square()
        )
        sampled += indices.numel()
    descriptors = data["descriptor"]
    for pair, group_count in zip(PAIRS, group_counts[len(SPECIES) :]):
        values = data["offsite"][pair]
        indices = _sample(
            values["target"].shape[0], args.offsite_batch_size, generator, device
        )
        source = values["source"].index_select(0, indices)
        target_index = values["target_index"].index_select(0, indices)
        prediction = model.predict_offsite(
            pair,
            descriptors.index_select(0, source),
            values["displacement"].index_select(0, indices),
            descriptors.index_select(0, target_index),
        )
        target = values["target"].index_select(0, indices)
        envelope = values["envelope"].index_select(0, indices)
        weighted_loss = weighted_loss + (
            group_count / total_count
        ) * range_factored_mse(
            prediction,
            target,
            envelope,
            mode=args.range_loss_mode,
        )
        sampled += indices.numel()
    return weighted_loss, sampled


@torch.no_grad()
def _symmetry_errors(
    model: FullBlockNeuralPairMapper, device: torch.device, seed: int
) -> dict[str, float]:
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    descriptor_i = torch.randn(
        3,
        model.input_descriptor_irreps.dim,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    descriptor_j = torch.randn(
        descriptor_i.shape, generator=generator, dtype=torch.float32, device=device
    )
    displacement = torch.randn(
        3, 3, generator=generator, dtype=torch.float32, device=device
    )

    def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
        return float((torch.linalg.vector_norm(actual - expected) / denominator).item())

    reference = model.predict_offsite(
        ("O", "Si"), descriptor_i, displacement, descriptor_j
    )
    errors = {}
    for label, determinant in (("proper_o3", 1), ("improper_o3", -1)):
        rotation = o3.rand_matrix(dtype=torch.float32).to(device)
        if determinant == -1:
            rotation = -rotation
        descriptor_action = o3_representation_matrix(
            model.input_descriptor_irreps, rotation
        )
        target_action = model.target_transform.output_action(("O", "Si"), rotation)
        actual = model.predict_offsite(
            ("O", "Si"),
            descriptor_i @ descriptor_action.T,
            displacement @ rotation.T,
            descriptor_j @ descriptor_action.T,
        )
        errors[label] = relative(actual, reference @ target_action.T)
    same = model.predict_offsite(("O", "O"), descriptor_i, displacement, descriptor_j)
    same_reverse = model.predict_offsite(
        ("O", "O"), descriptor_j, -displacement, descriptor_i
    )
    errors["raw_homogeneous_reversal"] = relative(
        same_reverse, model.target_transform.reverse(("O", "O"), same)
    )
    same_raw = torch.cat((same, same_reverse), dim=0)
    same_inverse = torch.cat(
        (
            torch.arange(3, 6, device=device),
            torch.arange(0, 3, device=device),
        )
    )
    same_projected = project_directed_irreps(
        model.target_transform,
        DIRECTED_PAIR_NAMES,
        same_raw,
        torch.zeros(6, device=device, dtype=torch.long),
        same_inverse,
    )
    errors["projected_homogeneous_reversal"] = relative(
        same_projected[3:],
        model.target_transform.reverse(("O", "O"), same_projected[:3]),
    )
    onsite = model.predict_onsite("O", descriptor_i)
    errors["raw_onsite_hermiticity"] = relative(
        onsite, model.target_transform.reverse(("O", "O"), onsite)
    )
    heterogeneous_reverse = model.predict_offsite(
        ("Si", "O"), descriptor_j, -displacement, descriptor_i
    )
    raw = torch.cat((reference, heterogeneous_reverse), dim=0)
    inverse = torch.cat(
        (
            torch.arange(3, 6, device=device),
            torch.arange(0, 3, device=device),
        )
    )
    pair_types = torch.cat(
        (
            torch.full((3,), 1, device=device, dtype=torch.long),
            torch.full((3,), 2, device=device, dtype=torch.long),
        )
    )
    projected = project_directed_irreps(
        model.target_transform,
        DIRECTED_PAIR_NAMES,
        raw,
        pair_types,
        inverse,
    )
    errors["raw_heterogeneous_reversal"] = relative(
        heterogeneous_reverse, model.target_transform.reverse(("O", "Si"), reference)
    )
    errors["projected_heterogeneous_reversal"] = relative(
        projected[3:],
        model.target_transform.reverse(("O", "Si"), projected[:3]),
    )
    projected_onsite = project_onsite_irreps(model.target_transform, "O", onsite)
    errors["projected_onsite_hermiticity"] = relative(
        projected_onsite,
        model.target_transform.reverse(("O", "O"), projected_onsite),
    )
    return errors


@torch.no_grad()
def _evaluate(
    model: FullBlockNeuralPairMapper,
    data: dict[str, object],
    transform_cpu: FullBlockIrrepTransform,
    batch_size: int,
    distance_bin_width: float,
) -> tuple[dict[str, object], int]:
    model.eval()
    metrics = {
        mode: FullBlockMetricAccumulator(
            transform_cpu, distance_bin_width_angstrom=distance_bin_width
        )
        for mode in ("raw", "projected")
    }
    directed_blocks = 0
    projection_change_square = [0.0, 0.0]
    for species in SPECIES:
        values = data["onsite"][species]
        for start in range(0, values["target"].shape[0], batch_size):
            stop = min(start + batch_size, values["target"].shape[0])
            prediction = model.predict_onsite(species, values["descriptor"][start:stop])
            target = values["target"][start:stop]
            projected = project_onsite_irreps(
                model.target_transform, species, prediction
            )
            metrics["raw"].update(
                (species, species),
                prediction.cpu(),
                target.cpu(),
                onsite=True,
            )
            metrics["projected"].update(
                (species, species),
                projected.cpu(),
                target.cpu(),
                onsite=True,
            )
            projection_change_square[0] += float(
                (prediction - projected).square().sum().item()
            )
            projection_change_square[1] += float(prediction.square().sum().item())
            directed_blocks += stop - start
    descriptors = data["descriptor"]
    flat = data["offsite_flat"]
    raw = torch.empty_like(flat["target"])
    for pair_index, pair in enumerate(PAIRS):
        selected = torch.nonzero(flat["pair_type"] == pair_index).flatten()
        for start in range(0, selected.numel(), batch_size):
            batch = selected[start : start + batch_size]
            source = flat["source"].index_select(0, batch)
            target_index = flat["target_index"].index_select(0, batch)
            prediction = model.predict_offsite(
                pair,
                descriptors.index_select(0, source),
                flat["displacement"].index_select(0, batch),
                descriptors.index_select(0, target_index),
            )
            prediction = prediction * flat["envelope"].index_select(0, batch)[:, None]
            raw.index_copy_(0, batch, prediction)
    projected = project_directed_irreps(
        model.target_transform,
        DIRECTED_PAIR_NAMES,
        raw,
        flat["pair_type"],
        flat["inverse"],
    )
    projection_change_square[0] += float((raw - projected).square().sum().item())
    projection_change_square[1] += float(raw.square().sum().item())
    for pair_index, pair in enumerate(PAIRS):
        selected = torch.nonzero(flat["pair_type"] == pair_index).flatten()
        for start in range(0, selected.numel(), batch_size):
            batch = selected[start : start + batch_size]
            distance = torch.linalg.vector_norm(
                flat["displacement"].index_select(0, batch), dim=-1
            )
            target = flat["target"].index_select(0, batch)
            metrics["raw"].update(
                pair,
                raw.index_select(0, batch).cpu(),
                target.cpu(),
                onsite=False,
                distances_angstrom=distance.cpu(),
            )
            metrics["projected"].update(
                pair,
                projected.index_select(0, batch).cpu(),
                target.cpu(),
                onsite=False,
                distances_angstrom=distance.cpu(),
            )
            directed_blocks += batch.numel()
    model.train()
    result = {mode: accumulator.compute() for mode, accumulator in metrics.items()}
    result["raw_relative_projection_change"] = (
        projection_change_square[0] / max(projection_change_square[1], 1.0e-300)
    ) ** 0.5
    return result, directed_blocks


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0 < args.train_fraction <= 1:
        raise ValueError("train fraction must lie in (0, 1]")
    positive = (
        args.descriptor_multiplicity_cap,
        args.generator_multiplicity,
        args.hidden_multiplicity,
        args.invariant_hidden,
        args.factorization_rank,
        args.bond_radial_count,
        args.bond_cutoff_angstrom,
        args.learning_rate,
        args.steps,
        args.eval_interval,
        args.early_stopping_evaluations,
        args.onsite_batch_size,
        args.offsite_batch_size,
        args.evaluation_batch_size,
        args.envelope_floor_hartree,
        args.distance_bin_width_angstrom,
        args.float32_symmetry_tolerance,
    )
    if (
        min(positive) <= 0
        or args.weight_decay < 0
        or args.hidden_l_max < 0
        or args.bond_l_max < 0
    ):
        raise ValueError("invalid calibration settings")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    descriptor_summary = json.loads(args.descriptor_summary.read_text())
    baseline_summary = json.loads(args.baseline_summary.read_text())
    normalization_summary = json.loads(args.normalization_summary.read_text())
    promotions = json.loads(args.promotions.read_text())
    calibration = json.loads(args.calibration_manifest.read_text())
    schemas = {
        item["key"]: item for item in json.loads(args.descriptor_schemas.read_text())
    }
    if not all(
        value["passed"]
        for value in (descriptor_summary, baseline_summary, normalization_summary)
    ):
        raise ValueError("all source artifacts must pass")
    promoted = next(
        (
            item
            for item in promotions["promotions"]
            if item["key"] == args.descriptor_key
        ),
        None,
    )
    if promoted is None or promoted["family"] != args.family:
        raise ValueError("descriptor is absent from the frozen promotion manifest")
    if float(promoted["cutoff_angstrom"]) != 6.5:
        raise ValueError("neural calibration is frozen at R_D = R_H = 6.5 angstrom")
    task = next(
        (item for item in calibration["tasks"] if item["task_id"] == args.task_id),
        None,
    )
    if task is None:
        raise ValueError(f"unknown frozen calibration task {args.task_id!r}")
    frozen_values = {
        "architecture": args.architecture,
        "resource_band": args.resource_band,
        "learning_rate": args.learning_rate,
        "descriptor_multiplicity_cap": args.descriptor_multiplicity_cap,
        "generator_multiplicity": args.generator_multiplicity,
        "hidden_multiplicity": args.hidden_multiplicity,
        "invariant_hidden": args.invariant_hidden,
        "factorization_rank": args.factorization_rank,
        "range_loss_mode": args.range_loss_mode,
    }
    if any(task[key] != value for key, value in frozen_values.items()):
        raise ValueError("CLI settings differ from the frozen calibration task")
    frozen_run = {
        "descriptor_key": args.descriptor_key,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "early_stopping_evaluations": args.early_stopping_evaluations,
    }
    if any(calibration[key] != value for key, value in frozen_run.items()):
        raise ValueError("CLI run settings differ from the frozen calibration manifest")
    fixed_run = {
        "hidden_l_max": args.hidden_l_max,
        "bond_radial_count": args.bond_radial_count,
        "bond_l_max": args.bond_l_max,
        "bond_cutoff_angstrom": args.bond_cutoff_angstrom,
        "weight_decay": args.weight_decay,
        "onsite_batch_size": args.onsite_batch_size,
        "offsite_batch_size": args.offsite_batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "envelope_floor_hartree": args.envelope_floor_hartree,
        "distance_bin_width_angstrom": args.distance_bin_width_angstrom,
        "float32_symmetry_tolerance": args.float32_symmetry_tolerance,
        "device": args.device,
    }
    if fixed_run != calibration["fixed_settings"]:
        raise ValueError("CLI fixed settings differ from the calibration manifest")
    schema = schemas[args.descriptor_key]
    if schema["content_hash"] != promoted["content_hash"]:
        raise ValueError("descriptor schema hash mismatch")
    if (
        normalization_summary["normalization_content_hash"]
        != json.loads(args.normalization.read_text())["content_hash"]
    ):
        raise ValueError("normalization content hash mismatch")
    envelope_payload = json.loads(args.range_envelope.read_text())
    if envelope_payload["content_hash"] != baseline_summary["range_envelope_hash"]:
        raise ValueError("range envelope hash mismatch")

    rows = _rows(args.shard_registry, args.train_fraction, args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "resume"
        },
        "convention": "mandala-stage5-directed-neural-calibration-v2",
        "descriptor_content_hash": schema["content_hash"],
        "promotion_manifest_hash": promotions["manifest_hash"],
        "calibration_manifest_hash": calibration["manifest_hash"],
        "normalization_hash": normalization_summary["normalization_content_hash"],
        "range_envelope_hash": envelope_payload["content_hash"],
        "selected_structure_indices": {
            split: [int(row["structure_index"]) for row in values]
            for split, values in rows.items()
        },
        "selection_uses_test_hamiltonian": False,
        "test_shards_read": False,
        "training_time_hermiticity_enforcement": "none",
        "directed_training": "both reverse-pair records, one raw call each",
        "evaluation_hermiticity": "global reverse-pair projection",
        "gradient_clipping": False,
        "mixed_precision": False,
        "loss": {
            "space": "complete full-block irreps",
            "reduction": "block-count-weighted mean squared error",
            "offsite_objective": args.range_loss_mode,
            "physical_prediction": "model output multiplied by frozen range envelope",
            "normalized_target_weight": "G(r)^2 when normalized-target form is selected",
            "onsite_target": "unscaled target",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "source_hashes": {
            "descriptor_summary": _hash_file(args.descriptor_summary),
            "descriptor_schemas": _hash_file(args.descriptor_schemas),
            "baseline_summary": _hash_file(args.baseline_summary),
            "registry": _hash_file(args.shard_registry),
            "normalization": _hash_file(args.normalization),
            "promotions": _hash_file(args.promotions),
            "calibration_manifest": _hash_file(args.calibration_manifest),
            "range_envelope": _hash_file(args.range_envelope),
        },
    }
    config["manifest_hash"] = _hash(config)
    config_path = output / "config.json"
    existing = [path for path in output.iterdir() if path.name != "launcher.log"]
    if existing and not args.resume:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    if config_path.exists():
        if (
            json.loads(config_path.read_text())["manifest_hash"]
            != config["manifest_hash"]
        ):
            raise ValueError("resume configuration differs from the saved manifest")
    else:
        _atomic_json(config_path, config)

    scale = _normalization_scale(args.normalization, args.family, args.descriptor_key)
    envelope_table = load_pair_envelope_table(
        args.range_envelope,
        pair_order=DIRECTED_PAIR_NAMES,
        dtype=torch.float32,
        device="cpu",
    )
    train_data = _concatenate_partition(
        rows["train"],
        args.baseline_cache_dir,
        args.descriptor_cache_dir,
        args.descriptor_key,
        scale,
        envelope_table,
        args.envelope_floor_hartree,
        device,
    )
    validation_data = _concatenate_partition(
        rows["validation"],
        args.baseline_cache_dir,
        args.descriptor_cache_dir,
        args.descriptor_key,
        scale,
        envelope_table,
        args.envelope_floor_hartree,
        device,
    )
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"}),
        dtype=torch.float32,
        device=device,
    )
    transform_cpu = FullBlockIrrepTransform(
        transform.orbital_config, dtype=torch.float32, device="cpu"
    )
    model = FullBlockNeuralPairMapper(
        args.architecture,
        transform,
        schema["irreps_out"],
        bond_n_radial=args.bond_radial_count,
        bond_l_max=args.bond_l_max,
        bond_cutoff=args.bond_cutoff_angstrom,
        hidden_multiplicity=args.hidden_multiplicity,
        hidden_l_max=args.hidden_l_max,
        invariant_hidden=args.invariant_hidden,
        factorization_rank=args.factorization_rank,
        descriptor_multiplicity_cap=args.descriptor_multiplicity_cap,
        generator_multiplicity=args.generator_multiplicity,
        dtype=torch.float32,
    ).to(device)
    initial_symmetry = _symmetry_errors(model, device, args.seed + 2)
    initial_gate = {
        key: value
        for key, value in initial_symmetry.items()
        if key.startswith(("proper_", "improper_", "projected_"))
    }
    if max(initial_gate.values()) > args.float32_symmetry_tolerance:
        raise ValueError(f"initial float32 symmetry gate failed: {initial_symmetry}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    checkpoint_path = output / "last.pt"
    history = []
    start_step = 0
    best_mae = float("inf")
    best_step = 0
    stale = 0
    sampled_blocks = 0
    elapsed_before = 0.0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator_state"])
        start_step = int(checkpoint["step"])
        best_mae = float(checkpoint["best_mae"])
        best_step = int(checkpoint["best_step"])
        stale = int(checkpoint["stale"])
        history = checkpoint["history"]
        sampled_blocks = int(checkpoint["sampled_blocks"])
        elapsed_before = float(checkpoint["elapsed_seconds"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()
    stopped_early = False
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, sampled = _training_loss(model, train_data, args, generator, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        sampled_blocks += sampled
        if step % args.eval_interval != 0 and step != args.steps:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics, validation_blocks = _evaluate(
            model,
            validation_data,
            transform_cpu,
            args.evaluation_batch_size,
            args.distance_bin_width_angstrom,
        )
        mae = float(metrics["projected"]["matrix_elements"]["mae"])
        history.append(
            {
                "step": step,
                "training_loss": float(loss.item()),
                "validation_matrix_mae_mev": mae,
            }
        )
        improved = mae < best_mae
        if improved:
            best_mae, best_step, stale = mae, step, 0
            torch.save(model.state_dict(), output / "best_model.pt")
            _atomic_json(output / "best_validation_metrics.json", metrics)
        else:
            stale += 1
        elapsed = elapsed_before + time.perf_counter() - started
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "step": step,
                "best_mae": best_mae,
                "best_step": best_step,
                "stale": stale,
                "history": history,
                "sampled_blocks": sampled_blocks,
                "elapsed_seconds": elapsed,
            },
            checkpoint_path,
        )
        _atomic_json(output / "history.json", history)
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if stale >= args.early_stopping_evaluations:
            stopped_early = True
            break

    elapsed = elapsed_before + time.perf_counter() - started
    best_state = torch.load(
        output / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_state)
    final_metrics, validation_blocks = _evaluate(
        model,
        validation_data,
        transform_cpu,
        args.evaluation_batch_size,
        args.distance_bin_width_angstrom,
    )
    final_symmetry = _symmetry_errors(model, device, args.seed + 3)
    _atomic_json(output / "validation_metrics.json", final_metrics)
    summary = {
        "completed": True,
        "passed": math.isfinite(
            float(final_metrics["projected"]["matrix_elements"]["mae"])
        )
        and max(
            value
            for key, value in final_symmetry.items()
            if key.startswith(("proper_", "improper_", "projected_"))
        )
        <= args.float32_symmetry_tolerance,
        "manifest_hash": config["manifest_hash"],
        "architecture": args.architecture,
        "resource_band": args.resource_band,
        "descriptor_key": args.descriptor_key,
        "parameter_count": parameter_count,
        "best_step": best_step,
        "range_loss_mode": args.range_loss_mode,
        "best_validation_matrix_mae_mev": final_metrics["projected"]["matrix_elements"][
            "mae"
        ],
        "best_validation_matrix_rmse_mev": final_metrics["projected"][
            "matrix_elements"
        ]["rmse"],
        "best_validation_raw_matrix_mae_mev": final_metrics["raw"]["matrix_elements"][
            "mae"
        ],
        "best_validation_raw_matrix_rmse_mev": final_metrics["raw"]["matrix_elements"][
            "rmse"
        ],
        "raw_relative_projection_change": final_metrics[
            "raw_relative_projection_change"
        ],
        "stopped_early": stopped_early,
        "training_seconds": elapsed,
        "sampled_training_blocks": sampled_blocks,
        "sampled_training_blocks_per_second": sampled_blocks / elapsed,
        "validation_directed_block_count": validation_blocks,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "test_shards_read": False,
        "initial_float32_symmetry_errors": initial_symmetry,
        "final_float32_symmetry_errors": final_symmetry,
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
