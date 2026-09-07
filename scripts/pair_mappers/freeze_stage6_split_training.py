#!/usr/bin/env python3
"""Freeze independent onsite and offsite Stage-6 validation experiments."""

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


def _finalize(payload: dict[str, object]) -> dict[str, object]:
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and list(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    offsite_dir = output / "offsite"
    onsite_dir = output / "onsite_neural"
    offsite_dir.mkdir()
    onsite_dir.mkdir()
    aggregate = json.loads(args.calibration_aggregate.read_text())
    promotions = json.loads(args.promotions.read_text())
    if not aggregate["passed"] or aggregate["test_shards_read"]:
        raise ValueError("Stage-5 calibration must pass without test access")
    expected = {
        "m3": "m3_large_lr1em3_physical",
        "m5": "m5_large_lr1em3_physical",
    }
    if aggregate["best_by_architecture"] != expected:
        raise ValueError("offsite calibration leaders differ from the evaluated gate")
    promoted = {
        (item["family"], item["key"]): item for item in promotions["promotions"]
    }
    descriptors = {
        "d1": "r6p5_spherical_bessel_high_n8_l6",
        "d2": "r6p5_d2_degree10_o10_l10",
        "d3": "r6p5_d3_high_o8_l6",
        "d4": "r6p5_d4_spherical_bessel_high_n8_l6_b3",
    }
    for family, key in descriptors.items():
        item = promoted.get((family, key))
        if item is None or float(item["cutoff_angstrom"]) != 6.5:
            raise ValueError(
                f"missing frozen high-resolution descriptor {family}/{key}"
            )

    resource_bands = {
        "medium": {
            "descriptor_multiplicity_cap": 4,
            "generator_multiplicity": 4,
            "hidden_multiplicity": 4,
            "invariant_hidden": 128,
            "factorization_rank": 64,
        },
        "large": {
            "descriptor_multiplicity_cap": 8,
            "generator_multiplicity": 8,
            "hidden_multiplicity": 8,
            "invariant_hidden": 256,
            "factorization_rank": 128,
        },
    }
    common = {
        "selection_partition": "validation",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "train_fraction": 1.0,
        "structure_sampling_before_pair_expansion": True,
        "seed": 20260907,
        "range_loss_modes": ["physical_mse"],
        "source_hashes": {
            "calibration_aggregate": _sha256(args.calibration_aggregate),
            "promotions": _sha256(args.promotions),
        },
        "calibration_gate": {
            "manifest_hash": aggregate["calibration_manifest_hash"],
            "best_by_architecture": expected,
        },
    }
    offsite_tasks = []
    for architecture in ("m3", "m5"):
        offsite_tasks.append(
            {
                "task_id": f"offsite_{architecture}_large_lr1em3",
                "family": "d1",
                "descriptor_key": descriptors["d1"],
                "architecture": architecture,
                "learning_rate": 1.0e-3,
                "range_loss_mode": "physical_mse",
                "target_scope": "offsite",
                "onsite_baseline": "none",
                "onsite_loss_weight": None,
                "separate_descriptor_projections": True,
                "resource_band": "large",
                **resource_bands["large"],
            }
        )
    offsite_manifest = _finalize(
        {
            **common,
            "convention": "mandala-stage6-independent-directed-offsite-v1",
            "steps": 10000,
            "eval_interval": 500,
            "early_stopping_evaluations": 8,
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
            "tasks": offsite_tasks,
        }
    )
    onsite_tasks = []
    for family, descriptor_key in descriptors.items():
        for resource_band, baseline in (
            ("medium", "none"),
            ("medium", "invariant_mean"),
            ("large", "invariant_mean"),
        ):
            task_id = f"onsite_{family}_{resource_band}_{baseline}"
            onsite_tasks.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "descriptor_key": descriptor_key,
                    "architecture": "m3",
                    "learning_rate": 1.0e-3,
                    "range_loss_mode": "physical_mse",
                    "target_scope": "onsite",
                    "onsite_baseline": baseline,
                    "onsite_loss_weight": None,
                    "separate_descriptor_projections": True,
                    "resource_band": resource_band,
                    **resource_bands[resource_band],
                }
            )
    onsite_manifest = _finalize(
        {
            **common,
            "convention": "mandala-stage6-independent-onsite-cg-residual-v1",
            "steps": 6000,
            "eval_interval": 250,
            "early_stopping_evaluations": 10,
            "fixed_settings": {
                "hidden_l_max": 4,
                "bond_radial_count": 1,
                "bond_l_max": 0,
                "bond_cutoff_angstrom": 6.5,
                "weight_decay": 1.0e-6,
                "onsite_batch_size": 512,
                "offsite_batch_size": 1,
                "evaluation_batch_size": 2048,
                "envelope_floor_hartree": 1.0e-8,
                "distance_bin_width_angstrom": 0.5,
                "float32_symmetry_tolerance": 2.0e-5,
                "device": "cuda",
            },
            "tasks": onsite_tasks,
        }
    )
    _atomic_json(offsite_dir / "calibration_manifest.json", offsite_manifest)
    _atomic_json(onsite_dir / "calibration_manifest.json", onsite_manifest)
    summary = {
        "completed": True,
        "passed": len(offsite_tasks) == 2 and len(onsite_tasks) == 12,
        "offsite_manifest_hash": offsite_manifest["manifest_hash"],
        "onsite_neural_manifest_hash": onsite_manifest["manifest_hash"],
        "offsite_task_count": len(offsite_tasks),
        "onsite_neural_task_count": len(onsite_tasks),
        "test_shards_read": False,
    }
    _atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Stage 6 independent onsite/offsite freeze\n\n"
        "The two selected M3/M5 approaches are retrained only on all directed offsite "
        "records. Onsite fitting is a separate validation grid: invariant species means, "
        "linear and affine equivariant ridge screens (separate CPU launcher), and "
        "onsite-only quadratic-CG residual networks across all four descriptor families. "
        "No parameters, losses, or training batches are shared between scopes. Physical "
        "predictions are projected only during validation. Test targets are never read.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
