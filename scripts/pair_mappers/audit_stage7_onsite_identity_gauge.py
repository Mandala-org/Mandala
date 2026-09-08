#!/usr/bin/env python3
"""Diagnose structure-level identity-gauge error of a frozen onsite model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.hermiticity import project_onsite_irreps
from pair_hamiltonian.metrics import FullBlockMetricAccumulator
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_mappers.m0 import ClosedFormM0PairMapper
from scripts.pair_mappers.run_stage6_onsite_ridge_screen import (
    SPECIES,
    SPECIES_Z,
    _load_onsite,
    _normalization_scales,
    _registry,
)

HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-cache-dir", type=Path, required=True)
    parser.add_argument("--descriptor-schemas", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--baseline-cache-dir", type=Path, required=True)
    parser.add_argument("--shard-registry", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--descriptor-key", required=True)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--cpu-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _optimal_identity_shifts(
    residual: torch.Tensor, identities: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return least-squares common-structure and independent-atom shifts."""
    if residual.shape != identities.shape or residual.ndim != 2:
        raise ValueError(
            "residual and identity arrays must have equal (atom, irrep) shape"
        )
    norms = identities.square().sum(dim=1)
    if torch.any(norms <= 0):
        raise ValueError("identity directions must have positive norm")
    atom_shift = (residual * identities).sum(dim=1) / norms
    structure_shift = (residual * identities).sum() / norms.sum()
    return structure_shift, atom_shift


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and [
        path for path in output.iterdir() if path.name != "launcher.log"
    ]:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.cpu_manifest.read_text())
    if (
        manifest["test_shards_read"]
        or manifest["ridge_family"] != "d4"
        or not str(manifest["gauge_model_id"]).endswith("affine_ridge_1em06")
        or args.ridge != 1.0e-6
    ):
        raise ValueError("CLI gauge model differs from the frozen CPU manifest")

    schemas = {
        item["key"]: item for item in json.loads(args.descriptor_schemas.read_text())
    }
    schema = schemas[args.descriptor_key]
    scales = _normalization_scales(args.normalization, "d4")
    registry = _registry(args.shard_registry)
    device = torch.device("cpu")
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"}),
        dtype=torch.float64,
        device=device,
    )
    model = ClosedFormM0PairMapper(
        transform,
        schema["irreps_out"],
        bond_n_radial=1,
        bond_l_max=0,
        bond_cutoff=6.5,
        ridge=args.ridge,
        onsite_affine=True,
        enabled_scope="onsite",
        dtype=torch.float64,
    )
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    identity = {
        species: transform.blocks_to_irreps(
            (species, species), torch.eye(13, dtype=torch.float64)
        )
        for species in SPECIES
    }
    identity_norm = {
        species: float(value.square().sum()) for species, value in identity.items()
    }
    if any(abs(value - 13.0) > 1.0e-8 for value in identity_norm.values()):
        raise ValueError(f"identity transform is not orthonormal: {identity_norm}")

    all_metrics: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    cached_fields: list[str] | None = None
    with h5py.File(
        args.baseline_cache_dir / "shards" / "structure_0000.h5", "r"
    ) as handle:
        cached_fields = sorted(handle.keys())
    metadata_fields = [
        name
        for name in cached_fields
        if any(
            token in name.lower() for token in ("fermi", "energy", "chemical_potential")
        )
    ]

    with torch.no_grad():
        for split in ("train", "validation"):
            metrics = {
                mode: FullBlockMetricAccumulator(
                    transform, distance_bin_width_angstrom=0.5
                )
                for mode in (
                    "uncorrected",
                    "structure_identity_oracle",
                    "atom_identity_oracle",
                )
            }
            for index in registry[split]:
                atomic_numbers, target, descriptors = _load_onsite(
                    args.baseline_cache_dir,
                    args.descriptor_cache_dir,
                    index,
                    [args.descriptor_key],
                )
                target = target.to(torch.float64)
                descriptor = (
                    descriptors[args.descriptor_key] / scales[args.descriptor_key]
                ).to(torch.float64)
                prediction = torch.empty_like(target)
                identities = torch.empty_like(target)
                for species in SPECIES:
                    selected = torch.nonzero(
                        atomic_numbers == SPECIES_Z[species]
                    ).flatten()
                    raw = model.predict_onsite(
                        species, descriptor.index_select(0, selected)
                    )
                    projected = project_onsite_irreps(transform, species, raw)
                    prediction.index_copy_(0, selected, projected)
                    identities.index_copy_(
                        0, selected, identity[species].expand(selected.numel(), -1)
                    )
                residual = target - prediction
                structure_shift_tensor, atom_shift = _optimal_identity_shifts(
                    residual, identities
                )
                structure_shift = float(structure_shift_tensor)
                structure_prediction = prediction + structure_shift * identities
                atom_prediction = prediction + atom_shift[:, None] * identities
                target_trace_mean = float(
                    (target * identities).sum() / identities.square().sum()
                )
                rows.append(
                    {
                        "split": split,
                        "structure_index": index,
                        "atom_count": int(atomic_numbers.numel()),
                        "oxygen_fraction": float((atomic_numbers == 8).float().mean()),
                        "oracle_identity_shift_hartree": structure_shift,
                        "oracle_identity_shift_ev": structure_shift * HARTREE_TO_EV,
                        "target_mean_identity_hartree": target_trace_mean,
                        "residual_mse_before_hartree2": float(residual.square().mean()),
                        "residual_mse_after_structure_shift_hartree2": float(
                            (target - structure_prediction).square().mean()
                        ),
                    }
                )
                for species in SPECIES:
                    selected = torch.nonzero(
                        atomic_numbers == SPECIES_Z[species]
                    ).flatten()
                    pair = (species, species)
                    chosen_target = target.index_select(0, selected)
                    for mode, values in (
                        ("uncorrected", prediction),
                        ("structure_identity_oracle", structure_prediction),
                        ("atom_identity_oracle", atom_prediction),
                    ):
                        metrics[mode].update(
                            pair,
                            values.index_select(0, selected),
                            chosen_target,
                            onsite=True,
                        )
            all_metrics[split] = {
                mode: accumulator.compute() for mode, accumulator in metrics.items()
            }

    with (output / "structure_shifts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write(output / "metrics.json", all_metrics)
    validation_rows = [row for row in rows if row["split"] == "validation"]
    shifts_ev = [float(row["oracle_identity_shift_ev"]) for row in validation_rows]
    atom_counts = [float(row["atom_count"]) for row in validation_rows]
    oxygen = [float(row["oxygen_fraction"]) for row in validation_rows]
    target_trace = [
        float(row["target_mean_identity_hartree"]) for row in validation_rows
    ]
    base = all_metrics["validation"]["uncorrected"]["matrix_elements"]
    structure = all_metrics["validation"]["structure_identity_oracle"][
        "matrix_elements"
    ]
    atom = all_metrics["validation"]["atom_identity_oracle"]["matrix_elements"]
    summary = {
        "completed": True,
        "passed": all(math.isfinite(float(value)) for value in shifts_ev),
        "diagnostic_only": True,
        "deployable_predictor": False,
        "validation_structure_count": len(validation_rows),
        "uncorrected_validation_mae_mev": base["mae"],
        "structure_identity_oracle_validation_mae_mev": structure["mae"],
        "atom_identity_oracle_validation_mae_mev": atom["mae"],
        "structure_identity_oracle_mae_reduction_fraction": 1.0
        - float(structure["mae"]) / float(base["mae"]),
        "validation_shift_ev": {
            "mean": float(np.mean(shifts_ev)),
            "std": float(np.std(shifts_ev)),
            "min": float(np.min(shifts_ev)),
            "max": float(np.max(shifts_ev)),
        },
        "shift_correlations": {
            "atom_count": _correlation(shifts_ev, atom_counts),
            "oxygen_fraction": _correlation(shifts_ev, oxygen),
            "target_mean_identity": _correlation(shifts_ev, target_trace),
        },
        "energy_zero_or_fermi_metadata_available_in_cache": bool(metadata_fields),
        "matching_cached_metadata_fields": metadata_fields,
        "interpretation_caution": (
            "The target-derived oracle shift is an upper-bound diagnostic. It cannot "
            "be used at inference and may absorb physical structure-level variation."
        ),
        "test_shards_read": False,
    }
    config = {
        "convention": "mandala-stage7-onsite-identity-gauge-audit-v1",
        "descriptor_key": args.descriptor_key,
        "ridge": args.ridge,
        "selection_partition": "validation",
        "test_shards_read": False,
        "source_hashes": {
            "descriptor_schemas": _sha256(args.descriptor_schemas),
            "normalization": _sha256(args.normalization),
            "registry": _sha256(args.shard_registry),
            "model_checkpoint": _sha256(args.model_checkpoint),
            "cpu_manifest": _sha256(args.cpu_manifest),
        },
    }
    config["manifest_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary["manifest_hash"] = config["manifest_hash"]
    _write(output / "config.json", config)
    _write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
