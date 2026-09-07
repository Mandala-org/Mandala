#!/usr/bin/env python3
"""Fit validation-only onsite baselines and equivariant ridge models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.hermiticity import project_onsite_irreps
from pair_hamiltonian.metrics import FullBlockMetricAccumulator
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_mappers.m0 import ClosedFormM0PairMapper

SPECIES = ("O", "Si")
SPECIES_Z = {"O": 8, "Si": 14}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("d1", "d2", "d3", "d4"), required=True)
    parser.add_argument("--descriptor-cache-dir", type=Path, required=True)
    parser.add_argument("--descriptor-summary", type=Path, required=True)
    parser.add_argument("--descriptor-schemas", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--normalization-summary", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--baseline-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--shard-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--ridge-values", nargs="+", type=float, required=True)
    parser.add_argument("--include-mean-baseline", action="store_true")
    parser.add_argument("--distance-bin-width-angstrom", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _non_log_output_entries(path: Path) -> list[Path]:
    """Return material outputs, ignoring the log pre-created by ``tee``."""
    if not path.exists():
        return []
    return [entry for entry in path.iterdir() if entry.name != "launcher.log"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry(path: Path) -> dict[str, list[int]]:
    result = {"train": [], "validation": []}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in result:
                result[row["split"]].append(int(row["structure_index"]))
    if any(not values for values in result.values()):
        raise ValueError("train and validation partitions must be nonempty")
    return result


def _normalization_scales(path: Path, family: str) -> dict[str, torch.Tensor]:
    records = json.loads(path.read_text())["families"][family]["descriptors"]
    result = {}
    for record in records:
        scale = torch.empty(int(record["dimension"]), dtype=torch.float32)
        covered = torch.zeros_like(scale, dtype=torch.bool)
        for channel in record["channels"]:
            start, stop = int(channel["start"]), int(channel["stop"])
            scale[start:stop] = float(channel["rms"])
            covered[start:stop] = True
        if (
            not torch.all(covered)
            or torch.any(scale <= 0)
            or not torch.all(torch.isfinite(scale))
        ):
            raise ValueError(f"invalid normalization for {record['key']}")
        result[record["key"]] = scale
    return result


def _load_onsite(
    baseline_dir: Path, descriptor_dir: Path, index: int, keys: list[str]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    baseline_path = baseline_dir / "shards" / f"structure_{index:04d}.h5"
    descriptor_path = descriptor_dir / "shards" / f"structure_{index:04d}.h5"
    with h5py.File(baseline_path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete baseline shard {baseline_path}")
        atomic_numbers = torch.from_numpy(handle["atomic_numbers"][:])
        targets = torch.from_numpy(handle["onsite_target_irreps_hartree"][:])
    with h5py.File(descriptor_path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete descriptor shard {descriptor_path}")
        descriptors = {
            key: torch.from_numpy(handle["descriptors"][key][:]) for key in keys
        }
    return atomic_numbers, targets, descriptors


def _invariant_mean(
    mean: torch.Tensor, transform: FullBlockIrrepTransform, species: str
) -> torch.Tensor:
    result = torch.zeros_like(mean)
    offset = 0
    for multiplicity, irrep in transform.irreps((species, species)):
        width = multiplicity * irrep.dim
        if irrep.l == 0 and irrep.p == 1:
            result[offset : offset + width] = mean[offset : offset + width]
        offset += width
    projected = project_onsite_irreps(transform, species, result)
    cleaned = torch.zeros_like(projected)
    offset = 0
    for multiplicity, irrep in transform.irreps((species, species)):
        width = multiplicity * irrep.dim
        if irrep.l == 0 and irrep.p == 1:
            cleaned[offset : offset + width] = projected[offset : offset + width]
        offset += width
    return cleaned


def _ridge_label(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if any(value < 0 for value in args.ridge_values):
        raise ValueError("ridge values must be non-negative")
    if args.distance_bin_width_angstrom <= 0:
        raise ValueError("distance bin width must be positive")
    output = args.output_dir.resolve()
    existing = _non_log_output_entries(output)
    if existing:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    baseline_summary = json.loads(args.baseline_summary.read_text())
    descriptor_summary = json.loads(args.descriptor_summary.read_text())
    normalization_summary = json.loads(args.normalization_summary.read_text())
    promotions_payload = json.loads(args.promotions.read_text())
    if not all(
        payload["passed"]
        for payload in (baseline_summary, descriptor_summary, normalization_summary)
    ):
        raise ValueError("source caches and normalization must pass")
    if promotions_payload["selection_uses_hamiltonian"]:
        raise ValueError("descriptor promotion must remain geometry-only")
    promotions = [
        item
        for item in promotions_payload["promotions"]
        if item["family"] == args.family
    ]
    if len(promotions) != 8:
        raise ValueError(f"expected eight promoted {args.family} descriptors")
    schemas = {
        item["key"]: item for item in json.loads(args.descriptor_schemas.read_text())
    }
    for item in promotions:
        if schemas[item["key"]]["content_hash"] != item["content_hash"]:
            raise ValueError(f"descriptor schema mismatch for {item['key']}")
    keys = [item["key"] for item in promotions]
    scales = _normalization_scales(args.normalization, args.family)
    if any(key not in scales for key in keys):
        raise ValueError("normalization is missing a promoted descriptor")
    registry = _registry(args.shard_registry)
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"}),
        dtype=torch.float32,
        device=device,
    )

    models: dict[str, ClosedFormM0PairMapper] = {}
    model_metadata: dict[str, dict[str, object]] = {}
    for key in keys:
        for affine in (False, True):
            for ridge in args.ridge_values:
                method = "affine_ridge" if affine else "linear_ridge"
                model_id = f"{key}__{method}_{_ridge_label(ridge)}"
                models[model_id] = ClosedFormM0PairMapper(
                    transform,
                    schemas[key]["irreps_out"],
                    bond_n_radial=1,
                    bond_l_max=0,
                    bond_cutoff=6.5,
                    ridge=ridge,
                    onsite_affine=affine,
                    enabled_scope="onsite",
                    dtype=torch.float64,
                ).to(device)
                model_metadata[model_id] = {
                    "descriptor_key": key,
                    "method": method,
                    "ridge": ridge,
                }
    accumulators = {
        model_id: {
            species: model.onsite_accumulator(species, device=device)
            for species in SPECIES
        }
        for model_id, model in models.items()
    }
    target_sum: dict[str, torch.Tensor | None] = {species: None for species in SPECIES}
    target_count = {species: 0 for species in SPECIES}
    started = time.perf_counter()
    for index in tqdm(
        registry["train"], desc=f"fit onsite {args.family}", unit="structure"
    ):
        atomic_numbers, targets, descriptors = _load_onsite(
            args.baseline_cache_dir, args.descriptor_cache_dir, index, keys
        )
        atomic_numbers, targets = atomic_numbers.to(device), targets.to(device)
        normalized = {
            key: descriptors[key].to(device) / scales[key].to(device) for key in keys
        }
        for species in SPECIES:
            selected = torch.nonzero(atomic_numbers == SPECIES_Z[species]).flatten()
            selected_target = targets.index_select(0, selected)
            summed = selected_target.sum(dim=0)
            target_sum[species] = (
                summed if target_sum[species] is None else target_sum[species] + summed
            )
            target_count[species] += selected.numel()
            for model_id, model in models.items():
                key = str(model_metadata[model_id]["descriptor_key"])
                features = model.onsite_features(
                    normalized[key].index_select(0, selected)
                )
                accumulators[model_id][species].update(features, selected_target)
    fit_stream_seconds = time.perf_counter() - started

    diagnostics = {}
    for model_id, model in models.items():
        diagnostics[model_id] = {}
        for species in SPECIES:
            diagnostics[model_id][species] = asdict(
                model.fit_onsite_from_accumulator(
                    species, accumulators[model_id][species]
                )
            )
        torch.save(model.state_dict(), output / "models" / f"{model_id}.pt")
    _atomic_json(output / "fit_diagnostics.json", diagnostics)
    means = {
        species: _invariant_mean(
            target_sum[species] / target_count[species], transform, species
        )
        for species in SPECIES
    }
    if args.include_mean_baseline:
        torch.save(
            {key: value.cpu() for key, value in means.items()},
            output / "invariant_mean.pt",
        )

    metric_transform = FullBlockIrrepTransform(
        transform.orbital_config, dtype=torch.float32, device="cpu"
    )
    method_ids = list(models) + (
        ["invariant_mean"] if args.include_mean_baseline else []
    )
    metrics = {
        model_id: {
            mode: FullBlockMetricAccumulator(
                metric_transform,
                distance_bin_width_angstrom=args.distance_bin_width_angstrom,
            )
            for mode in ("raw", "projected")
        }
        for model_id in method_ids
    }
    projection_squares = {model_id: [0.0, 0.0] for model_id in method_ids}
    validation_blocks = 0
    started = time.perf_counter()
    for index in tqdm(
        registry["validation"], desc=f"validate onsite {args.family}", unit="structure"
    ):
        atomic_numbers, targets, descriptors = _load_onsite(
            args.baseline_cache_dir, args.descriptor_cache_dir, index, keys
        )
        atomic_numbers, targets = atomic_numbers.to(device), targets.to(device)
        normalized = {
            key: descriptors[key].to(device) / scales[key].to(device) for key in keys
        }
        for species in SPECIES:
            selected = torch.nonzero(atomic_numbers == SPECIES_Z[species]).flatten()
            selected_target = targets.index_select(0, selected)
            validation_blocks += selected.numel()
            for model_id, model in models.items():
                key = str(model_metadata[model_id]["descriptor_key"])
                raw = model.predict_onsite(
                    species, normalized[key].index_select(0, selected)
                )
                projected = project_onsite_irreps(transform, species, raw)
                for mode, prediction in (("raw", raw), ("projected", projected)):
                    metrics[model_id][mode].update(
                        (species, species),
                        prediction.cpu(),
                        selected_target.cpu(),
                        onsite=True,
                    )
                projection_squares[model_id][0] += float(
                    (raw - projected).square().sum()
                )
                projection_squares[model_id][1] += float(raw.square().sum())
            if args.include_mean_baseline:
                raw = means[species].expand(selected.numel(), -1)
                projected = project_onsite_irreps(transform, species, raw)
                for mode, prediction in (("raw", raw), ("projected", projected)):
                    metrics["invariant_mean"][mode].update(
                        (species, species),
                        prediction.cpu(),
                        selected_target.cpu(),
                        onsite=True,
                    )
    validation_seconds = time.perf_counter() - started
    detailed = {
        model_id: {mode: accumulator.compute() for mode, accumulator in modes.items()}
        for model_id, modes in metrics.items()
    }
    _atomic_json(output / "validation_metrics.json", detailed)
    rows = []
    for model_id in method_ids:
        metadata = model_metadata.get(
            model_id,
            {"descriptor_key": None, "method": "invariant_mean", "ridge": None},
        )
        projected = detailed[model_id]["projected"]["matrix_elements"]
        raw = detailed[model_id]["raw"]["matrix_elements"]
        numerator, denominator = projection_squares[model_id]
        rows.append(
            {
                "model_id": model_id,
                "family": args.family,
                **metadata,
                "validation_matrix_mae_mev": projected["mae"],
                "validation_matrix_rmse_mev": projected["rmse"],
                "validation_raw_matrix_mae_mev": raw["mae"],
                "raw_relative_projection_change": math.sqrt(
                    numerator / max(denominator, 1e-300)
                ),
                "validation_matrix_element_count": projected["scalar_count"],
                "finite": math.isfinite(float(projected["mae"])),
            }
        )
    rows.sort(key=lambda row: float(row["validation_matrix_mae_mev"]))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "convention": "mandala-stage6-independent-onsite-ridge-v1",
        "optimization_scope": "onsite_only",
        "training_time_hermiticity_enforcement": "none",
        "evaluation_hermiticity": "onsite transpose projection",
        "selection_partition": "validation",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "promotion_manifest_hash": promotions_payload["manifest_hash"],
        "source_hashes": {
            "baseline_summary": _sha256(args.baseline_summary),
            "descriptor_summary": _sha256(args.descriptor_summary),
            "descriptor_schemas": _sha256(args.descriptor_schemas),
            "normalization": _sha256(args.normalization),
            "normalization_summary": _sha256(args.normalization_summary),
            "promotions": _sha256(args.promotions),
            "registry": _sha256(args.shard_registry),
        },
    }
    config["manifest_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "config.json", config)
    summary = {
        "completed": True,
        "passed": all(row["finite"] for row in rows),
        "manifest_hash": config["manifest_hash"],
        "family": args.family,
        "configuration_count": len(rows),
        "best_model_id": rows[0]["model_id"],
        "best_validation_matrix_mae_mev": rows[0]["validation_matrix_mae_mev"],
        "fit_stream_seconds": fit_stream_seconds,
        "validation_seconds": validation_seconds,
        "validation_onsite_block_count": validation_blocks,
        "test_shards_read": False,
    }
    _atomic_json(output / "summary.json", summary)
    lines = [
        f"# Independent onsite screen: {args.family.upper()}",
        "",
        "Only onsite targets were read. Models are ranked on validation after onsite Hermitian projection.",
        "",
        "| Rank | Method | Descriptor | Ridge | Validation MAE (meV) | RMSE (meV) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {row['method']} | {row['descriptor_key'] or 'none'} | "
            f"{row['ridge'] if row['ridge'] is not None else 'n/a'} | "
            f"{float(row['validation_matrix_mae_mev']):.6f} | "
            f"{float(row['validation_matrix_rmse_mev']):.6f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
