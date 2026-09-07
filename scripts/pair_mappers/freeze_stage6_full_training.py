#!/usr/bin/env python3
"""Freeze full-data, onsite-balanced M3/M5 follow-up training tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-aggregate", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and list(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads(args.calibration_aggregate.read_text())
    promotions = json.loads(args.promotions.read_text())
    if not aggregate["passed"] or aggregate["test_shards_read"]:
        raise ValueError("calibration must pass without test access")
    expected = {
        "m3": "m3_large_lr1em3_physical",
        "m5": "m5_large_lr1em3_physical",
    }
    if aggregate["best_by_architecture"] != expected:
        raise ValueError("calibration leaders differ from the evaluated gate")
    descriptor_key = "r6p5_spherical_bessel_high_n8_l6"
    promoted = next(
        (item for item in promotions["promotions"] if item["key"] == descriptor_key),
        None,
    )
    if promoted is None:
        raise ValueError("full-training descriptor is not geometry-promoted")
    large = {
        "resource_band": "large",
        "descriptor_multiplicity_cap": 8,
        "generator_multiplicity": 8,
        "hidden_multiplicity": 8,
        "invariant_hidden": 256,
        "factorization_rank": 128,
    }
    tasks = []
    for architecture in ("m3", "m5"):
        for onsite_loss_weight, label in ((0.05, "0p05"), (0.2, "0p2"), (0.5, "0p5")):
            tasks.append(
                {
                    "task_id": f"{architecture}_large_lr1em3_onsite{label}",
                    "architecture": architecture,
                    "learning_rate": 1.0e-3,
                    "range_loss_mode": "physical_mse",
                    "onsite_loss_weight": onsite_loss_weight,
                    **large,
                }
            )
    manifest = {
        "convention": "mandala-stage6-full-data-onsite-balance-v1",
        "selection_partition": "validation",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "descriptor_family": "d1",
        "descriptor_key": descriptor_key,
        "descriptor_content_hash": promoted["content_hash"],
        "descriptor_cutoff_angstrom": 6.5,
        "train_fraction": 1.0,
        "structure_sampling_before_pair_expansion": True,
        "seed": 20260907,
        "steps": 10000,
        "eval_interval": 500,
        "early_stopping_evaluations": 6,
        "range_loss_modes": ["physical_mse"],
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
        "tasks": tasks,
        "m0_gate": None,
        "calibration_gate": {
            "manifest_hash": aggregate["calibration_manifest_hash"],
            "best_by_architecture": expected,
            "best_validation_matrix_mae_mev": aggregate[
                "best_overall_validation_matrix_mae_mev"
            ],
        },
        "source_hashes": {
            "calibration_aggregate": sha256(args.calibration_aggregate),
            "promotions": sha256(args.promotions),
        },
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output / "calibration_manifest.json", manifest)
    summary = {
        "completed": True,
        "passed": len(tasks) == 6,
        "manifest_hash": manifest["manifest_hash"],
        "task_count": len(tasks),
        "test_shards_read": False,
    }
    atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Stage 6 full-data neural training freeze\n\n"
        "M3-large and M5-large at learning rate 1e-3 were selected on validation. "
        "Each is trained on all training structures for up to 10,000 steps with "
        "onsite objective weights 0.05, 0.2, and 0.5. The physical range loss is "
        "used because Stage 5 demonstrated its weighted H/G form is equivalent. "
        "No test target is read.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
