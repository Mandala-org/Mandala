#!/usr/bin/env python3
"""Run one frozen Stage-7 offsite descriptor/continuation task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from pair_hamiltonian.stage6_validation import assess_stage6_scoped_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--stage7-root", type=Path, required=True)
    parser.add_argument("--stage6-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.stage7_root / "offsite" / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get(
        "convention"
    ) != "mandala-stage7-offsite-descriptor-decay-v1" or manifest.get(
        "test_shards_read"
    ):
        raise ValueError("Stage-7 offsite manifest is invalid")
    tasks = manifest["tasks"]
    if not 0 <= args.task_index < len(tasks):
        raise ValueError(f"task-index must lie in [0, {len(tasks) - 1}]")
    task = tasks[args.task_index]
    output = args.stage7_root / "offsite" / "runs" / task["task_id"]
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text())
        config = json.loads((output / "config.json").read_text())
        assessment = assess_stage6_scoped_run(
            payload, float(config["float32_symmetry_tolerance"])
        )
        if not assessment["accepted"]:
            raise ValueError(f"existing task summary is invalid: {summary}")
        print(json.dumps(assessment, indent=2, sort_keys=True))
        return

    runtime_mode: list[str]
    if (output / "config.json").is_file():
        runtime_mode = ["--resume"]
    elif task["warm_start_task_id"] is not None:
        runtime_mode = [
            "--warm-start-run-dir",
            str(args.stage6_root / "offsite" / "runs" / task["warm_start_task_id"]),
        ]
    else:
        runtime_mode = []

    family = task["family"]
    command = [
        sys.executable,
        "-u",
        "scripts/pair_mappers/train_stage5_neural_calibration.py",
        "--family",
        family,
        "--descriptor-key",
        task["descriptor_key"],
        "--descriptor-cache-dir",
        f"/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/{family}_sio2_v2",
        "--descriptor-summary",
        f"artifacts/pair_stage2/{family}_sio2_precompute_v2/summary.json",
        "--descriptor-schemas",
        f"artifacts/pair_stage2/{family}_sio2_precompute_v2/descriptor_schemas.json",
        "--normalization",
        "artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json",
        "--normalization-summary",
        "artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json",
        "--promotions",
        "artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json",
        "--calibration-manifest",
        str(manifest_path),
        "--task-id",
        task["task_id"],
        "--baseline-cache-dir",
        "/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2",
        "--baseline-summary",
        "artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json",
        "--shard-registry",
        "artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv",
        "--range-envelope",
        "artifacts/pair_stage1/sio2_baseline_cache_v2/range_envelope.json",
        "--output-dir",
        str(output),
        "--architecture",
        task["architecture"],
        "--resource-band",
        task["resource_band"],
        "--descriptor-multiplicity-cap",
        str(task["descriptor_multiplicity_cap"]),
        "--generator-multiplicity",
        str(task["generator_multiplicity"]),
        "--hidden-multiplicity",
        str(task["hidden_multiplicity"]),
        "--hidden-l-max",
        "4",
        "--invariant-hidden",
        str(task["invariant_hidden"]),
        "--factorization-rank",
        str(task["factorization_rank"]),
        "--bond-radial-count",
        "2",
        "--bond-l-max",
        "4",
        "--bond-cutoff-angstrom",
        "6.5",
        "--learning-rate",
        "1e-3",
        "--learning-rate-schedule",
        manifest["learning_rate_schedule"],
        "--minimum-learning-rate",
        str(manifest["minimum_learning_rate"]),
        "--decay-start-step",
        str(manifest["decay_start_step"]),
        "--weight-decay",
        "1e-6",
        "--range-loss-mode",
        "physical_mse",
        "--steps",
        str(manifest["steps"]),
        "--eval-interval",
        str(manifest["eval_interval"]),
        "--early-stopping-evaluations",
        str(manifest["early_stopping_evaluations"]),
        "--onsite-batch-size",
        "256",
        "--offsite-batch-size",
        "512",
        "--evaluation-batch-size",
        "2048",
        "--train-fraction",
        "1.0",
        "--target-scope",
        "offsite",
        "--onsite-baseline",
        "none",
        "--separate-descriptor-projections",
        "--seed",
        str(manifest["seed"]),
        "--device",
        "cuda",
        "--envelope-floor-hartree",
        "1e-8",
        "--distance-bin-width-angstrom",
        "0.5",
        "--float32-symmetry-tolerance",
        "2e-5",
        *runtime_mode,
    ]
    log_path = output / "launcher.log"
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    payload = json.loads(summary.read_text())
    config = json.loads((output / "config.json").read_text())
    assessment = assess_stage6_scoped_run(
        payload, float(config["float32_symmetry_tolerance"])
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    if not assessment["accepted"]:
        raise SystemExit("Stage-7 scoped run failed numerical certification")


if __name__ == "__main__":
    main()
