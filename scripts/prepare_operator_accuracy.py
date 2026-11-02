"""Prepare an immutable, validation-only common-split accuracy investigation.

This local preparation reads small pilot manifests, never datasets/checkpoints.
The registration lists every candidate and its budget before cluster execution.
"""

from __future__ import annotations

# Exact integer checks reject JSON booleans in frozen seeds and budgets.

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shlex

from omegaconf import OmegaConf


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    evidence,
    destination,
    remote_records,
    source_manifest_path=None,
    *,
    seeds=(42,),
    max_epochs=48,
    segments=1,
    allocation_hours=6,
    internal_hours=5,
    endpoints=(12, 24, 48),
    namespace="mandala_accuracy_v1",
    runtime_profile="mandala",
    concurrency=2,
):
    seeds, endpoints = tuple(seeds), tuple(endpoints)
    if runtime_profile not in {"mandala", "mandala_legacy"}:
        raise ValueError("runtime_profile must be mandala or mandala_legacy")
    if type(concurrency) is not int or concurrency < 1:  # noqa: E721
        raise ValueError("concurrency must be a positive integer")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            type(seed) is not int or not 0 <= seed < 2**32  # noqa: E721
            for seed in seeds
        )
    ):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if type(max_epochs) is not int or max_epochs <= 0:  # noqa: E721
        raise ValueError("max_epochs must be a positive integer")
    if type(segments) is not int or segments <= 0:  # noqa: E721
        raise ValueError("segments must be a positive integer")
    if (
        type(allocation_hours) is not int  # noqa: E721
        or not 0 < allocation_hours <= 48
    ):
        raise ValueError("allocation_hours must be an integer in [1, 48]")
    if not 0 < internal_hours < allocation_hours:
        raise ValueError("internal_hours must be positive and below allocation_hours")
    internal_seconds = int(internal_hours * 3600)
    if internal_seconds != internal_hours * 3600:
        raise ValueError("internal_hours must specify a whole number of seconds")
    if (
        not endpoints
        or any(
            type(epoch) is not int or not 0 < epoch <= max_epochs  # noqa: E721
            for epoch in endpoints
        )
        or list(endpoints) != sorted(set(endpoints))
    ):
        raise ValueError(
            "endpoints must be strictly increasing epochs within max_epochs"
        )
    if (
        not isinstance(namespace, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace) is None
    ):
        raise ValueError("namespace must be a single safe nonempty path component")
    if destination.exists():
        raise FileExistsError(destination)
    source_manifest = json.loads(
        (source_manifest_path or evidence / "source_manifest.json").read_text()
    )
    # Check the frozen inputs before claiming an immutable destination.
    for material in ("graphene", "mos2"):
        original = Path("configs/pilots") / f"{material}_mandala.yaml"
        if sha256(original) != source_manifest["files"][str(original)]:
            raise ValueError(
                f"Local pilot config differs from frozen remote: {original}"
            )
        for name in ("split_manifest.json", "run_manifest.json"):
            json.loads((evidence / material / name).read_text())
    destination.mkdir(parents=True)
    candidates = {
        "baseline": {},
        "mse": {"loss_l1_fraction": 0.0},
        "rollback_off": {"revert_on_spike": False},
        "mse_rollback_off": {"loss_l1_fraction": 0.0, "revert_on_spike": False},
        "adamw_decay0": {"optimization.optimizer.weight_decay": 0.0},
        "radial_copy": {"radial_weight_granularity": "copy"},
    }
    jobs, audits, reserved = [], [], {}
    checkpoint_root = (
        Path("/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints") / namespace
    )
    for material in ("graphene", "mos2"):
        original = Path("configs/pilots") / f"{material}_mandala.yaml"
        split_path = evidence / material / "split_manifest.json"
        split = json.loads(split_path.read_text())
        pilot_manifest = evidence / material / "run_manifest.json"
        expected_path = destination / f"{material}_pilot_run_manifest.json"
        expected_path.write_bytes(pilot_manifest.read_bytes())
        reserved[material] = {
            "pilot_split_hash_sha256": split["split_hash_sha256"],
            "reserved_test": split["splits"]["test"],
        }
        base = OmegaConf.to_container(OmegaConf.load(original), resolve=True)
        base["data"]["splits"] = {
            name: [str(Path(row["matrix"]).parent) for row in split["splits"][name]]
            for name in ("train", "val")
        }
        base["data"]["splits"]["test"] = []
        base["training"].pop("split", None)
        base["matrix_loss_reduction"] = "pair_sum"
        base["radial_weight_granularity"] = "term"
        base["save_dir"] = str(checkpoint_root)
        base["optimization"] = {
            "optimizer": {
                "kind": "adamw",
                "lr": base["lr"],
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": {
                "kind": "plateau",
                "ema_alpha": 1.0,
                "factor": base["lr_scheduler_factor"],
                "patience": base["lr_scheduler_patience"],
                "min_lr": base["lr_scheduler_min_lr"],
                "monitor": base["lr_scheduler_target"],
            },
        }
        audit_path = destination / f"{material}_audit.yaml"
        OmegaConf.save(OmegaConf.create(base), audit_path)
        audits.append(
            {
                "material": material,
                "config": audit_path.name,
                "expected_manifest": expected_path.name,
            }
        )
        settings = deepcopy(candidates)
        if material == "mos2":
            settings.update(
                pair_mean={"matrix_loss_reduction": "pair_mean"},
                elements={"matrix_loss_reduction": "elements"},
            )
        for seed in seeds:
            for name, changes in settings.items():
                run_id = f"{material}_{name}_s{seed}"
                document = OmegaConf.create(base)
                for key, value in changes.items():
                    OmegaConf.update(document, key, value)
                document.max_epochs = max_epochs
                document.training.max_epochs = max_epochs
                document.seed = document.training.seed = seed
                document.data_split_seed = document.training.data_split_seed = 42
                document.max_wall_clock_seconds = internal_seconds
                document.run_name = run_id
                config_path = destination / f"{run_id}.yaml"
                OmegaConf.save(document, config_path)
                launchers = (
                    [f"{run_id}.sh"]
                    if segments == 1
                    else [
                        f"{run_id}_segment_{index}.sh"
                        for index in range(1, segments + 1)
                    ]
                )
                jobs.append(
                    {
                        "id": run_id,
                        "namespace": namespace,
                        "material": material,
                        "hardware_block": f"{material}_s{seed}",
                        "treatment": name,
                        "changes_from_baseline": changes,
                        "config": config_path.name,
                        "launchers": launchers,
                        "seed": seed,
                        "data_split_seed": 42,
                        "max_epochs": max_epochs,
                        "segments": segments,
                        "internal_seconds": internal_seconds,
                        "allocation_hours": allocation_hours,
                        "max_gpu_hours": allocation_hours * segments,
                        "max_internal_seconds": internal_seconds * segments,
                        "primary_endpoints_epochs": list(endpoints),
                    }
                )
    registration = {
        "schema_version": 1,
        "id": namespace,
        "purpose": (
            "Single-seed early-learning screen, not convergence or confirmed component attribution"
            if len(seeds) == 1 and max_epochs <= 48
            else "Preregistered independent component grid across all declared seeds; no validation-selected promotion and no convergence guarantee"
        ),
        "initialization_seeds": list(seeds),
        "data_split_seed": 42,
        "runtime_profile": runtime_profile,
        "primary_endpoints_epochs": list(endpoints),
        "hardware_policy": "Record hardware assignment separately before execution; block treatment/control comparisons by material and initialization seed. Report remaining hardware confounding.",
        "pilot_source_snapshot_sha256": source_manifest["source_snapshot_sha256"],
        "selection": "minimum raw global structure-mean observed-entry validation MAE; report projected at same checkpoint",
        "test_policy": "No test input paths in executable configs; reserved membership is metadata only. Do not inspect existing pilot test results for tuning.",
        "budget": {
            "candidates": len(jobs),
            "max_gpu_hours": sum(j["max_gpu_hours"] for j in jobs),
            "max_internal_gpu_hours": sum(j["max_internal_seconds"] for j in jobs)
            / 3600,
            "array_concurrency": concurrency if segments == 1 else None,
            "max_simultaneous_candidates": concurrency,
            "segments_per_candidate": segments,
            "max_epochs_per_candidate": max_epochs,
            "allocation_hours_per_segment": allocation_hours,
            "internal_seconds_per_segment": internal_seconds,
            "hardware_gpu_count_per_segment": 1,
            "actual_consumption": "Record completed epochs, optimizer updates, elapsed time and allocated GPU hours per segment; upper caps are not actual consumption",
            "no_automatic_retry": True,
        },
        "contrasts": [
            "mse-baseline",
            "rollback_off-baseline",
            "mse_rollback_off-mse",
            "mse_rollback_off-rollback_off",
            "adamw_decay0-baseline",
            "radial_copy-baseline",
            "mos2_pair_mean-baseline (scale)",
            "mos2_elements-pair_mean (weighting)",
        ],
        "interpretation": [
            "Matched completed endpoints only; report missing endpoints as resource-incomplete",
            (
                "48 epochs cannot diagnose a plateau scheduler with patience 60"
                if max_epochs <= 48
                else "Report complete plateau/rollback/update histories; a fixed epoch/time cap does not certify convergence"
            ),
            "MAE/MSE is at fixed optimizer settings; objective scaling/Adam epsilon/clipping remain mechanisms",
            "Pair_mean versus elements controls global normalization, with distinct species weighting",
            "Identical parameter shapes/init seeds; copy mode changes gating map, not tensor counts",
            "Every listed arm/seed is independent and preregistered; do not promote, omit or extend arms based on interim validation. Unlisted treatments, longer budgets and recipe/architecture bridges require a new registration",
        ],
        "audits": audits,
        "jobs": jobs,
        "files": {
            p.name: sha256(p) for p in sorted(destination.glob("*")) if p.is_file()
        },
    }
    (destination / "registration.json").write_text(
        json.dumps(registration, indent=2) + "\n"
    )
    (destination / "reserved_membership.json").write_text(
        json.dumps(reserved, indent=2) + "\n"
    )
    # Source is placed at remote_records/source; launchers accept no arguments.
    quoted_records = shlex.quote(str(remote_records))
    frozen_pythonpath = shlex.quote(
        f"{remote_records / 'source' / 'src'}:{remote_records / 'source'}"
    )
    header = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'if [[ $# != 0 ]]; then echo "This frozen launcher takes no arguments." >&2; exit 2; fi\n'
        f"source ~/scripts/profiles/{runtime_profile}.sh\n"
        f'export PYTHONPATH={frozen_pythonpath}"${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1\n"
    )
    verify = f"python {quoted_records}/source/scripts/prepare_operator_accuracy.py verify --records {quoted_records}\n"
    for item in audits:
        material = item["material"]
        cmd = [
            "python",
            "-u",
            "-m",
            "scripts.audit_operator_accuracy",
            "--config",
            str(remote_records / "batch" / item["config"]),
            "--expected-manifest",
            str(remote_records / "batch" / item["expected_manifest"]),
            "--checkpoint",
            f"/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/operator_benchmark_pilots_v1/{material}/best_model.pt",
            "--checkpoint-copy",
            str(checkpoint_root / "pilot_snapshots" / f"{material}.pt"),
            "--accelerator",
            "gpu",
            "--output",
            str(remote_records / "results" / f"{material}_audit.json"),
        ]
        (destination / f"audit_{material}.sh").write_text(
            header
            + "export PYTHONHASHSEED=42\n"
            + verify
            + "exec "
            + shlex.join(cmd)
            + "\n"
        )
    for item in jobs:
        run_id = item["id"]
        for segment, launcher in enumerate(item["launchers"], start=1):
            cmd = [
                "python",
                "-u",
                "-m",
                "scripts.run_training_segment",
                "--config",
                str(remote_records / "batch" / item["config"]),
                "--output",
                str(checkpoint_root / run_id),
                "--records",
                str(remote_records / "results" / run_id),
                "--segments",
                str(segments),
                "--segment",
                str(segment),
                "--budget-seconds",
                str(item["internal_seconds"]),
                "--accelerator",
                "gpu",
            ]
            (destination / launcher).write_text(
                header
                + f"export PYTHONHASHSEED={item['seed']}\n"
                + verify
                + f"python {quoted_records}/source/scripts/prepare_operator_accuracy.py gate --records {quoted_records}\n"
                + "exec "
                + shlex.join(cmd)
                + "\n"
            )
    if segments == 1:
        (destination / "screen_array.sh").write_text(
            header
            + verify
            + f"python {quoted_records}/source/scripts/prepare_operator_accuracy.py gate --records {quoted_records}\n"
            + "case ${SLURM_ARRAY_TASK_ID:?} in\n"
            + "".join(
                f"{i}) exec bash {shlex.quote(str(remote_records / 'batch' / (item['id'] + '.sh')))} ;;\n"
                for i, item in enumerate(jobs)
            )
            + "*) exit 2 ;;\nesac\n"
        )
    # Include every executable and reserved metadata in final immutable digest.
    registration["files"] = {
        p.name: sha256(p)
        for p in sorted(destination.glob("*"))
        if p.is_file() and p.name != "registration.json"
    }
    (destination / "registration.json").write_text(
        json.dumps(registration, indent=2) + "\n"
    )
    print(json.dumps(registration["budget"], indent=2))


def verify(records):
    source = records / "source"
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    if not manifest.get("files"):
        raise ValueError("Frozen source manifest has no files")
    for name, expected in manifest["files"].items():
        if not (source / name).resolve().is_relative_to(source.resolve()):
            raise ValueError("Invalid frozen source path: " + name)
        if sha256(source / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    registration = json.loads((records / "batch/registration.json").read_text())
    if not registration.get("files"):
        raise ValueError("Frozen batch registration has no files")
    for name, expected in registration["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid frozen batch path: " + name)
        if sha256(records / "batch" / name) != expected:
            raise ValueError("Frozen batch changed: " + name)


def gate(records):
    """Requires the explicit human/agent review artifact, not elapsed time."""
    verify(records)
    review = json.loads((records / "workflow/certification_review.json").read_text())
    if review.get("decision") != "release_registered_screen":
        raise ValueError("Certification review has not released the screen")
    registration_path = records / "batch/registration.json"
    if review["registration_sha256"] != sha256(registration_path):
        raise ValueError("Registration changed after review")
    for material in ("graphene", "mos2"):
        path = records / "results" / f"{material}_audit.json"
        if (
            sha256(path) != review["audit_sha256"][material]
            or json.loads(path.read_text()).get("passed") is not True
        ):
            raise ValueError("Certification changed or failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--evidence", type=Path, required=True)
    prep.add_argument("--destination", type=Path, required=True)
    prep.add_argument("--remote-records", type=Path, required=True)
    prep.add_argument(
        "--source-manifest",
        type=Path,
        help="Explicit authoritative manifest; defaults to EVIDENCE/source_manifest.json",
    )
    prep.add_argument("--seeds", type=int, nargs="+", default=[42])
    prep.add_argument("--max-epochs", type=int, default=48)
    prep.add_argument("--segments", type=int, default=1)
    prep.add_argument("--allocation-hours", type=int, default=6)
    prep.add_argument("--internal-hours", type=float, default=5)
    prep.add_argument("--endpoints", type=int, nargs="+", default=[12, 24, 48])
    prep.add_argument("--namespace", default="mandala_accuracy_v1")
    prep.add_argument(
        "--runtime-profile", choices=("mandala", "mandala_legacy"), default="mandala"
    )
    prep.add_argument("--concurrency", type=int, default=2)
    for name in ("verify", "gate"):
        commands.add_parser(name).add_argument("--records", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.evidence,
            args.destination,
            args.remote_records,
            args.source_manifest,
            seeds=args.seeds,
            max_epochs=args.max_epochs,
            segments=args.segments,
            allocation_hours=args.allocation_hours,
            internal_hours=args.internal_hours,
            endpoints=args.endpoints,
            namespace=args.namespace,
            runtime_profile=args.runtime_profile,
            concurrency=args.concurrency,
        )
    elif args.command == "verify":
        verify(args.records)
    else:
        gate(args.records)


if __name__ == "__main__":
    main()
