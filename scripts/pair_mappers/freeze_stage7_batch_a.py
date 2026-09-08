#!/usr/bin/env python3
"""Freeze the Stage-7 gauge, ridge, and offsite continuation batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DESCRIPTORS = {
    "d1": "r6p5_spherical_bessel_high_n8_l6",
    "d2": "r6p5_d2_degree10_o10_l10",
    "d3": "r6p5_d3_high_o8_l6",
    "d4": "r6p5_d4_spherical_bessel_high_n8_l6_b3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage6-root", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    stage6 = args.stage6_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and list(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage6_summary = json.loads((stage6 / "aggregate_v1" / "summary.json").read_text())
    selected = json.loads(
        (stage6 / "aggregate_v1" / "selected_bundle.json").read_text()
    )
    promotions = json.loads(args.promotions.read_text())
    if (
        not stage6_summary["passed"]
        or stage6_summary["test_shards_read"]
        or selected["test_shards_read"]
        or selected["offsite"]["model_id"] != "offsite_m5_large_lr1em3"
        or selected["onsite"]["family"] != "d4"
    ):
        raise ValueError("Stage-6 selection gate is incomplete or differs from Batch A")
    promoted = {(item["family"], item["key"]) for item in promotions["promotions"]}
    if any((family, key) not in promoted for family, key in DESCRIPTORS.items()):
        raise ValueError("a frozen high-resolution descriptor is not promoted")

    task_common = {
        "architecture": "m5",
        "learning_rate": 1.0e-3,
        "range_loss_mode": "physical_mse",
        "target_scope": "offsite",
        "onsite_baseline": "none",
        "onsite_loss_weight": None,
        "separate_descriptor_projections": True,
        "resource_band": "large",
        "descriptor_multiplicity_cap": 8,
        "generator_multiplicity": 8,
        "hidden_multiplicity": 8,
        "invariant_hidden": 256,
        "factorization_rank": 128,
    }
    tasks = []
    for family, descriptor_key in DESCRIPTORS.items():
        tasks.append(
            {
                **task_common,
                "task_id": f"offsite_m5_{family}_high_decay30k",
                "family": family,
                "descriptor_key": descriptor_key,
                "warm_start_task_id": (
                    "offsite_m5_large_lr1em3" if family == "d1" else None
                ),
            }
        )
    manifest = {
        "convention": "mandala-stage7-offsite-descriptor-decay-v1",
        "selection_partition": "validation",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "train_fraction": 1.0,
        "structure_sampling_before_pair_expansion": True,
        "seed": 20260907,
        "range_loss_modes": ["physical_mse"],
        "steps": 30000,
        "eval_interval": 500,
        # There are 60 validation evaluations in a fresh 30k run.  A patience
        # of 61 deliberately disables early stopping so all descriptor families
        # receive the complete preregistered schedule.
        "early_stopping_evaluations": 61,
        "learning_rate_schedule": "constant_then_cosine",
        "minimum_learning_rate": 1.0e-4,
        "decay_start_step": 10000,
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
        "source_hashes": {
            "stage6_summary": _sha256(stage6 / "aggregate_v1" / "summary.json"),
            "stage6_selection": _sha256(
                stage6 / "aggregate_v1" / "selected_bundle.json"
            ),
            "promotions": _sha256(args.promotions),
        },
    }
    manifest["manifest_hash"] = _hash(manifest)
    (output / "offsite").mkdir()
    _write(output / "offsite" / "calibration_manifest.json", manifest)
    cpu_manifest = {
        "convention": "mandala-stage7-onsite-diagnostics-v1",
        "selection_partition": "validation",
        "test_shards_read": False,
        "ridge_family": "d4",
        "ridge_values": [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "gauge_model_id": selected["onsite"]["model_id"],
        "gauge_diagnostic": (
            "oracle per-structure common scalar identity correction of the frozen "
            "Stage-6 D4 ridge residual; diagnostic only, never a deployable predictor"
        ),
        "source_selection_hash": selected["manifest_hash"],
    }
    cpu_manifest["manifest_hash"] = _hash(cpu_manifest)
    _write(output / "cpu_manifest.json", cpu_manifest)
    summary = {
        "completed": True,
        "passed": len(tasks) == 4,
        "gpu_task_count": len(tasks),
        "cpu_task_count": 2,
        "gpu_manifest_hash": manifest["manifest_hash"],
        "cpu_manifest_hash": cpu_manifest["manifest_hash"],
        "test_shards_read": False,
    }
    _write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
