#!/usr/bin/env python3
"""Freeze the validation-only M3/M5 calibration grid after the M0 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-aggregate", type=Path)
    parser.add_argument("--m0-results", type=Path)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and list(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    promotions = json.loads(args.promotions.read_text())
    if (args.m0_aggregate is None) != (args.m0_results is None):
        raise ValueError("provide both M0 inputs or neither")
    aggregate = None
    best_cutoffs = None
    if args.m0_aggregate is not None:
        aggregate = json.loads(args.m0_aggregate.read_text())
        if not aggregate["passed"] or aggregate["test_shards_read"]:
            raise ValueError("M0 validation gate must pass without test access")
        with args.m0_results.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 32:
            raise ValueError("M0 gate must contain 32 configurations")
        best_cutoffs = {}
        for family in ("d1", "d2", "d3", "d4"):
            for resolution in ("compact", "high"):
                subset = [
                    row
                    for row in rows
                    if row["family"] == family and row["resolution"] == resolution
                ]
                best = min(
                    subset,
                    key=lambda row: float(row["validation_matrix_mae_mev"]),
                )
                best_cutoffs[f"{family}_{resolution}"] = float(best["cutoff_angstrom"])
    descriptor_key = "r6p5_spherical_bessel_high_n8_l6"
    promoted = next(
        (item for item in promotions["promotions"] if item["key"] == descriptor_key),
        None,
    )
    if promoted is None:
        raise ValueError("D1-high calibration descriptor is not geometry-promoted")

    bands = {
        "small": {
            "descriptor_multiplicity_cap": 2,
            "generator_multiplicity": 2,
            "hidden_multiplicity": 2,
            "invariant_hidden": 64,
            "factorization_rank": 8,
        },
        "medium": {
            "descriptor_multiplicity_cap": 4,
            "generator_multiplicity": 4,
            "hidden_multiplicity": 4,
            "invariant_hidden": 128,
            "factorization_rank": 32,
        },
        "large": {
            "descriptor_multiplicity_cap": 8,
            "generator_multiplicity": 8,
            "hidden_multiplicity": 8,
            "invariant_hidden": 256,
            "factorization_rank": 128,
        },
    }
    tasks = []
    for architecture in ("m3", "m5"):
        for band, settings in bands.items():
            for learning_rate, label in ((3.0e-4, "3em4"), (1.0e-3, "1em3")):
                for range_loss_mode, loss_label in (
                    ("physical_mse", "physical"),
                    ("weighted_normalized_mse", "weighted_h_over_g"),
                ):
                    tasks.append(
                        {
                            "task_id": (
                                f"{architecture}_{band}_lr{label}_{loss_label}"
                            ),
                            "architecture": architecture,
                            "resource_band": band,
                            "learning_rate": learning_rate,
                            "range_loss_mode": range_loss_mode,
                            **settings,
                        }
                    )
    manifest = {
        "convention": "mandala-stage5-directed-neural-calibration-grid-v2",
        "selection_partition": "validation",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "descriptor_family": "d1",
        "descriptor_key": descriptor_key,
        "descriptor_content_hash": promoted["content_hash"],
        "descriptor_cutoff_angstrom": 6.5,
        "train_fraction": 0.25,
        "structure_sampling_before_pair_expansion": True,
        "seed": 20260905,
        "steps": 1500,
        "eval_interval": 250,
        "early_stopping_evaluations": 3,
        "learning_rates": [3.0e-4, 1.0e-3],
        "fixed_settings": {
            "hidden_l_max": 4,
            "bond_radial_count": 2,
            "bond_l_max": 4,
            "bond_cutoff_angstrom": 6.5,
            "weight_decay": 1.0e-6,
            "onsite_batch_size": 256,
            "offsite_batch_size": 512,
            "evaluation_batch_size": 2048,
            "envelope_floor_hartree": 1.0e-8,
            "distance_bin_width_angstrom": 0.5,
            "float32_symmetry_tolerance": 2.0e-5,
            "device": "cuda",
        },
        "bands": bands,
        "range_loss_modes": ["physical_mse", "weighted_normalized_mse"],
        "tasks": tasks,
        "selection_basis": (
            "pre_registered_geometry_promoted_d1_high_6p5"
            if aggregate is None
            else "optional_m0_validation_context"
        ),
        "m0_gate": (
            None
            if aggregate is None
            else {
                "manifest_hash": aggregate["manifest_hash"],
                "best_validation_matrix_mae_mev": aggregate[
                    "best_overall_validation_matrix_mae_mev"
                ],
                "best_cutoffs": best_cutoffs,
                "pareto_keys": aggregate["pareto_keys"],
            }
        ),
        "source_hashes": {
            "promotions": _sha256(args.promotions),
            **(
                {}
                if args.m0_aggregate is None
                else {
                    "m0_aggregate": _sha256(args.m0_aggregate),
                    "m0_results": _sha256(args.m0_results),
                }
            ),
        },
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "calibration_manifest.json", manifest)
    summary = {
        "completed": True,
        "passed": len(tasks) == 24,
        "manifest_hash": manifest["manifest_hash"],
        "task_count": len(tasks),
        "descriptor_key": descriptor_key,
        "test_shards_read": False,
    }
    _atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Stage 5 neural calibration freeze\n\n"
        "D1-high at 6.5 Å is pre-registered from the geometry-only promotion manifest as the "
        "representative raw-density descriptor; this freeze does not depend on corrected M0 outcomes.\n\n"
        "The frozen grid contains M3 and M5 at three resource bands, two learning rates, "
        "and two algebraically equivalent physical-space loss formulations (24 one-seed runs). "
        "It uses 25% of training structures and the full validation split. "
        "No test target is read.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
