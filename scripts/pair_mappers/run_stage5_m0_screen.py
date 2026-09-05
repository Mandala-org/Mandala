#!/usr/bin/env python3
"""Fit and validate closed-form M0 for one promoted SiO2 descriptor family."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from tqdm.auto import tqdm

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.envelope import build_edge_envelope, load_pair_envelope_table
from pair_hamiltonian.metrics import FullBlockMetricAccumulator
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_hamiltonian.hermiticity import (
    project_directed_irreps,
    project_onsite_irreps,
)
from pair_hamiltonian.range_objective import closed_form_range_batch
from pair_hamiltonian.sio2_cache import DIRECTED_PAIR_NAMES
from pair_mappers import ClosedFormM0PairMapper

SPECIES = ("O", "Si")
ATOMIC_NUMBER_TO_SPECIES = {8: "O", 14: "Si"}
DIRECTED_PAIRS = tuple(tuple(value.split("-")) for value in DIRECTED_PAIR_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("d1", "d2", "d3", "d4"), required=True)
    parser.add_argument("--descriptor-cache-dir", type=Path, required=True)
    parser.add_argument("--descriptor-summary", type=Path, required=True)
    parser.add_argument("--descriptor-schemas", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--baseline-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--shard-registry", type=Path, required=True)
    parser.add_argument("--range-envelope", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bond-radial-count", type=int, required=True)
    parser.add_argument("--bond-l-max", type=int, required=True)
    parser.add_argument("--bond-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument(
        "--range-fit-mode",
        choices=("physical_design", "weighted_normalized_target"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--envelope-floor-hartree", type=float, required=True)
    parser.add_argument("--distance-bin-width-angstrom", type=float, required=True)
    parser.add_argument("--max-structures-per-split", type=int, default=0)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry(path: Path, maximum: int) -> dict[str, list[dict[str, str]]]:
    selected = {"train": [], "validation": []}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in selected:
                continue
            if maximum == 0 or len(selected[split]) < maximum:
                selected[split].append(row)
    if any(not rows for rows in selected.values()):
        raise ValueError("train and validation registries must both be nonempty")
    return selected


def _baseline_shard(cache_dir: Path, index: int) -> dict[str, np.ndarray]:
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
    path = cache_dir / "shards" / f"structure_{index:04d}.h5"
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete baseline shard {path}")
        return {name: handle[name][:] for name in names}


def _descriptor_shard(
    cache_dir: Path, index: int, keys: list[str]
) -> dict[str, np.ndarray]:
    path = cache_dir / "shards" / f"structure_{index:04d}.h5"
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete descriptor shard {path}")
        return {key: handle["descriptors"][key][:] for key in keys}


def _batches(indices: torch.Tensor, size: int):
    for start in range(0, indices.numel(), size):
        yield indices[start : start + size]


def _envelope(
    displacement: torch.Tensor,
    pair_types: torch.Tensor,
    table,
    floor: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    return build_edge_envelope(distance, pair_types, table).clamp_min(floor)


def _build_model(
    schema: dict[str, object],
    transform: FullBlockIrrepTransform,
    args: argparse.Namespace,
    device: torch.device,
) -> ClosedFormM0PairMapper:
    return ClosedFormM0PairMapper(
        transform,
        str(schema["irreps_out"]),
        bond_n_radial=args.bond_radial_count,
        bond_l_max=args.bond_l_max,
        bond_cutoff=args.bond_cutoff_angstrom,
        ridge=args.ridge,
        dtype=torch.float64,
    ).to(device)


def _learned_coefficient_count(model: ClosedFormM0PairMapper) -> int:
    return sum(
        value.numel() for name, value in model.named_buffers() if ".weight_" in name
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if (
        min(
            args.bond_radial_count,
            args.bond_cutoff_angstrom,
            args.batch_size,
            args.envelope_floor_hartree,
            args.distance_bin_width_angstrom,
        )
        <= 0
        or args.bond_l_max < 0
        or args.ridge < 0
    ):
        raise ValueError("invalid positive-valued configuration")
    output = args.output_dir.resolve()
    existing = (
        []
        if not output.exists()
        else [p for p in output.iterdir() if p.name != "launcher.log"]
    )
    if existing:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    baseline_summary = json.loads(args.baseline_summary.read_text())
    descriptor_summary = json.loads(args.descriptor_summary.read_text())
    promotion_manifest = json.loads(args.promotions.read_text())
    schemas = {
        item["key"]: item for item in json.loads(args.descriptor_schemas.read_text())
    }
    if not baseline_summary["passed"] or not descriptor_summary["passed"]:
        raise ValueError("baseline and descriptor caches must pass")
    if promotion_manifest["selection_uses_hamiltonian"]:
        raise ValueError("promotion manifest must be geometry-only")
    promotions = [
        item
        for item in promotion_manifest["promotions"]
        if item["family"] == args.family
    ]
    if len(promotions) != 8:
        raise ValueError(f"expected eight promoted {args.family} configurations")
    for item in promotions:
        schema = schemas[item["key"]]
        if schema["content_hash"] != item["content_hash"]:
            raise ValueError(f"schema hash mismatch for {item['key']}")

    envelope_payload = json.loads(args.range_envelope.read_text())
    if envelope_payload["content_hash"] != baseline_summary["range_envelope_hash"]:
        raise ValueError("range envelope differs from frozen baseline")
    envelope_table = load_pair_envelope_table(
        args.range_envelope,
        pair_order=DIRECTED_PAIR_NAMES,
        dtype=torch.float32,
        device=device,
    )
    registry = _registry(args.shard_registry, args.max_structures_per_split)
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"}),
        dtype=torch.float64,
        device=device,
    )
    models = {
        item["key"]: _build_model(schemas[item["key"]], transform, args, device)
        for item in promotions
    }
    accumulators = {
        key: {
            "onsite": {
                species: model.onsite_accumulator(species, device=device)
                for species in SPECIES
            },
            "offsite": {
                pair: model.offsite_accumulator(pair, device=device)
                for pair in DIRECTED_PAIRS
            },
        }
        for key, model in models.items()
    }
    config = {
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "convention": "mandala-stage5-directed-m0-validation-screen-v2",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "training_time_hermiticity_enforcement": "none",
        "directed_training": "both reverse-pair records, one raw call each",
        "evaluation_hermiticity": "global reverse-pair projection",
        "promotion_manifest_hash": promotion_manifest["manifest_hash"],
        "source_hashes": {
            "baseline_summary": _sha256(args.baseline_summary),
            "descriptor_summary": _sha256(args.descriptor_summary),
            "descriptor_schemas": _sha256(args.descriptor_schemas),
            "promotions": _sha256(args.promotions),
            "range_envelope": _sha256(args.range_envelope),
            "registry": _sha256(args.shard_registry),
        },
        "selected_structure_counts": {
            key: len(value) for key, value in registry.items()
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    config["manifest_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "config.json", config)

    keys = [item["key"] for item in promotions]
    started = time.perf_counter()
    for row in tqdm(registry["train"], desc=f"fit {args.family}", unit="structure"):
        index = int(row["structure_index"])
        base = _baseline_shard(args.baseline_cache_dir, index)
        descriptors = _descriptor_shard(args.descriptor_cache_dir, index, keys)
        atomic_numbers = torch.from_numpy(base["atomic_numbers"]).to(device)
        onsite_target = torch.from_numpy(base["onsite_target_irreps_hartree"]).to(
            device
        )
        source = torch.from_numpy(base["offsite_source"]).to(
            device=device, dtype=torch.long
        )
        target = torch.from_numpy(base["offsite_target"]).to(
            device=device, dtype=torch.long
        )
        displacement = torch.from_numpy(base["offsite_displacement_angstrom"]).to(
            device
        )
        pair_types = torch.from_numpy(base["offsite_pair_type"]).to(
            device=device, dtype=torch.long
        )
        offsite_target = torch.from_numpy(base["offsite_target_irreps_hartree"]).to(
            device
        )
        for key, model in models.items():
            descriptor = torch.from_numpy(descriptors[key]).to(device)
            for atomic_number, species in ATOMIC_NUMBER_TO_SPECIES.items():
                selected = torch.nonzero(atomic_numbers == atomic_number).flatten()
                selected_descriptor = descriptor.index_select(0, selected)
                selected_target = onsite_target.index_select(0, selected)
                onsite_accumulator = accumulators[key]["onsite"][species]
                onsite_accumulator.update(selected_descriptor, selected_target)
            for pair_index, pair in enumerate(DIRECTED_PAIRS):
                selected = torch.nonzero(pair_types == pair_index).flatten()
                for batch in _batches(selected, args.batch_size):
                    batch_displacement = displacement.index_select(0, batch)
                    values = _envelope(
                        batch_displacement,
                        pair_types.index_select(0, batch),
                        envelope_table,
                        args.envelope_floor_hartree,
                    )
                    features = model.offsite_features(
                        descriptor.index_select(0, source.index_select(0, batch)),
                        batch_displacement,
                        descriptor.index_select(0, target.index_select(0, batch)),
                    )
                    fit_features, fit_target, sample_weight = closed_form_range_batch(
                        features,
                        offsite_target.index_select(0, batch),
                        values,
                        mode=args.range_fit_mode,
                    )
                    accumulators[key]["offsite"][pair].update(
                        fit_features, fit_target, sample_weight=sample_weight
                    )
    fit_stream_seconds = time.perf_counter() - started

    diagnostics: dict[str, object] = {}
    for key, model in models.items():
        diagnostics[key] = {"onsite": {}, "offsite": {}}
        for species in SPECIES:
            diagnostics[key]["onsite"][species] = asdict(
                model.fit_onsite_from_accumulator(
                    species, accumulators[key]["onsite"][species]
                )
            )
        for pair in DIRECTED_PAIRS:
            diagnostics[key]["offsite"]["-".join(pair)] = asdict(
                model.fit_offsite_from_accumulator(
                    pair, accumulators[key]["offsite"][pair]
                )
            )
        torch.save(model.state_dict(), model_dir / f"{key}.pt")
    _atomic_json(output / "fit_diagnostics.json", diagnostics)

    metric_transform = FullBlockIrrepTransform(
        transform.orbital_config, dtype=torch.float32, device="cpu"
    )
    metrics = {
        key: {
            mode: FullBlockMetricAccumulator(
                metric_transform,
                distance_bin_width_angstrom=args.distance_bin_width_angstrom,
            )
            for mode in ("raw", "projected")
        }
        for key in keys
    }
    block_counts = {key: 0 for key in keys}
    projection_change_square = {key: [0.0, 0.0] for key in keys}
    started = time.perf_counter()
    for row in tqdm(
        registry["validation"], desc=f"validate {args.family}", unit="structure"
    ):
        index = int(row["structure_index"])
        base = _baseline_shard(args.baseline_cache_dir, index)
        descriptors = _descriptor_shard(args.descriptor_cache_dir, index, keys)
        atomic_numbers = torch.from_numpy(base["atomic_numbers"]).to(device)
        onsite_target = torch.from_numpy(base["onsite_target_irreps_hartree"]).to(
            device
        )
        source = torch.from_numpy(base["offsite_source"]).to(
            device=device, dtype=torch.long
        )
        target = torch.from_numpy(base["offsite_target"]).to(
            device=device, dtype=torch.long
        )
        displacement = torch.from_numpy(base["offsite_displacement_angstrom"]).to(
            device
        )
        pair_types = torch.from_numpy(base["offsite_pair_type"]).to(
            device=device, dtype=torch.long
        )
        inverse = torch.from_numpy(base["offsite_inverse"]).to(
            device=device, dtype=torch.long
        )
        offsite_target = torch.from_numpy(base["offsite_target_irreps_hartree"]).to(
            device
        )
        for key, model in models.items():
            descriptor = torch.from_numpy(descriptors[key]).to(device)
            for atomic_number, species in ATOMIC_NUMBER_TO_SPECIES.items():
                selected = torch.nonzero(atomic_numbers == atomic_number).flatten()
                prediction = model.predict_onsite(
                    species, descriptor.index_select(0, selected)
                )
                target_values = onsite_target.index_select(0, selected)
                projected_prediction = project_onsite_irreps(
                    transform, species, prediction
                )
                metrics[key]["raw"].update(
                    (species, species),
                    prediction.cpu(),
                    target_values.cpu(),
                    onsite=True,
                )
                metrics[key]["projected"].update(
                    (species, species),
                    projected_prediction.cpu(),
                    target_values.cpu(),
                    onsite=True,
                )
                projection_change_square[key][0] += float(
                    (prediction - projected_prediction).square().sum().item()
                )
                projection_change_square[key][1] += float(
                    prediction.square().sum().item()
                )
                block_counts[key] += selected.numel()
            raw_offsite = torch.empty_like(offsite_target)
            for pair_index, pair in enumerate(DIRECTED_PAIRS):
                selected = torch.nonzero(pair_types == pair_index).flatten()
                for batch in _batches(selected, args.batch_size):
                    batch_displacement = displacement.index_select(0, batch)
                    prediction = model.predict_offsite(
                        pair,
                        descriptor.index_select(0, source.index_select(0, batch)),
                        batch_displacement,
                        descriptor.index_select(0, target.index_select(0, batch)),
                    )
                    values = _envelope(
                        batch_displacement,
                        pair_types.index_select(0, batch),
                        envelope_table,
                        args.envelope_floor_hartree,
                    )
                    prediction = prediction * values[:, None]
                    raw_offsite.index_copy_(0, batch, prediction)
            projected_offsite = project_directed_irreps(
                transform,
                DIRECTED_PAIR_NAMES,
                raw_offsite,
                pair_types,
                inverse,
            )
            projection_change_square[key][0] += float(
                (raw_offsite - projected_offsite).square().sum().item()
            )
            projection_change_square[key][1] += float(raw_offsite.square().sum().item())
            for pair_index, pair in enumerate(DIRECTED_PAIRS):
                selected = torch.nonzero(pair_types == pair_index).flatten()
                for batch in _batches(selected, args.batch_size):
                    distance = torch.linalg.vector_norm(
                        displacement.index_select(0, batch), dim=-1
                    )
                    batch_target = offsite_target.index_select(0, batch)
                    metrics[key]["raw"].update(
                        pair,
                        raw_offsite.index_select(0, batch).cpu(),
                        batch_target.cpu(),
                        onsite=False,
                        distances_angstrom=distance.cpu(),
                    )
                    metrics[key]["projected"].update(
                        pair,
                        projected_offsite.index_select(0, batch).cpu(),
                        batch_target.cpu(),
                        onsite=False,
                        distances_angstrom=distance.cpu(),
                    )
                    block_counts[key] += batch.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    validation_seconds = time.perf_counter() - started
    detailed = {
        key: {mode: accumulator.compute() for mode, accumulator in value.items()}
        for key, value in metrics.items()
    }
    _atomic_json(output / "validation_metrics.json", detailed)

    rows = []
    promotion_by_key = {item["key"]: item for item in promotions}
    for key in keys:
        raw_headline = detailed[key]["raw"]["matrix_elements"]
        headline = detailed[key]["projected"]["matrix_elements"]
        item = promotion_by_key[key]
        missing = sorted(
            {
                irrep
                for site in diagnostics[key].values()
                for pair in site.values()
                for irrep, condition in pair["condition_numbers"].items()
                if math.isinf(condition)
            }
        )
        rows.append(
            {
                "manifest_hash": config["manifest_hash"],
                "family": args.family,
                "key": key,
                "resolution": item["resolution"],
                "cutoff_angstrom": item["cutoff_angstrom"],
                "descriptor_dimension": item["dimension"],
                "uncompressed_bytes_per_atom": 4 * int(item["dimension"]),
                "learned_coefficient_count": _learned_coefficient_count(models[key]),
                "descriptor_cache_size_bytes": descriptor_summary["cache_size_bytes"],
                "descriptor_precompute_seconds": descriptor_summary["elapsed_seconds"],
                "descriptor_neighbor_contributions_per_second": descriptor_summary.get(
                    "neighbor_contributions_per_second"
                ),
                "validation_matrix_mae_mev": headline["mae"],
                "validation_matrix_rmse_mev": headline["rmse"],
                "validation_raw_matrix_mae_mev": raw_headline["mae"],
                "validation_raw_matrix_rmse_mev": raw_headline["rmse"],
                "raw_relative_projection_change": math.sqrt(
                    projection_change_square[key][0]
                    / max(projection_change_square[key][1], 1.0e-300)
                ),
                "validation_matrix_element_count": headline["scalar_count"],
                "validation_block_count": block_counts[key],
                "missing_target_irreps": ";".join(missing),
                "finite": bool(
                    math.isfinite(headline["mae"]) and math.isfinite(headline["rmse"])
                ),
            }
        )
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "completed": True,
        "passed": all(row["finite"] for row in rows),
        "manifest_hash": config["manifest_hash"],
        "family": args.family,
        "configuration_count": len(rows),
        "finite_count": sum(row["finite"] for row in rows),
        "test_shards_read": False,
        "fit_stream_seconds": fit_stream_seconds,
        "fit_structures_per_second": (len(registry["train"]) / fit_stream_seconds),
        "validation_seconds": validation_seconds,
        "validation_directed_blocks_per_second_across_eight_models": (
            sum(block_counts.values()) / validation_seconds
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    _atomic_json(output / "summary.json", summary)
    lines = [
        f"# Stage 5 M0 validation screen: {args.family.upper()}",
        "",
        "Test shards were not read. All offsite targets use the frozen train-only range envelope.",
        "",
        "| Resolution | $R_D$ (Å) | Dimension | Validation MAE (meV) | RMSE (meV) | Missing types |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {float(row['cutoff_angstrom']):.1f} | "
            f"{row['descriptor_dimension']} | {float(row['validation_matrix_mae_mev']):.6f} | "
            f"{float(row['validation_matrix_rmse_mev']):.6f} | {row['missing_target_irreps'] or 'none'} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
