"""Queue every registered segmented arm with hardware blocks and a data gate.

Run on the ROSI login node: only JSON, source hashes and Slurm commands are used.
First segments remain held until the afterok certification-review job releases
them. Continuations depend on successful first segments. No automatic retry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from scripts.prepare_operator_accuracy import sha256, verify


def planned_command(records, job, segment, *, previous=None):
    allocation = job["allocation"]
    command = [
        "sbatch",
        "--parsable",
        f"--account={allocation['account']}",
        f"--partition={allocation['partition']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={allocation['cpus']}",
        f"--mem={allocation['mem']}",
        "--gres=gpu:1",
        f"--time={job['allocation_hours']:02d}:00:00",
        "--signal=INT@600",
        f"--chdir={records / 'source'}",
        f"--job-name=ma-{job['id']}-s{segment}",
        f"--output={records / 'logs' / (job['id'] + '_segment' + str(segment) + '_%j.out')}",
    ]
    if allocation.get("exclude"):
        command.append("--exclude=" + allocation["exclude"])
    command.append(
        "--hold" if segment == 1 else "--dependency=afterok:" + str(previous)
    )
    command.append(str(records / "batch" / job["launchers"][segment - 1]))
    return command


def submit(
    records, certification_records, dependencies, runtime_probes=(), *, dry_run=False
):
    verify(records)
    registration = json.loads((records / "batch/registration.json").read_text())
    for job in registration["jobs"]:
        planned_command(records, job, 1)
    if dry_run:
        print(
            json.dumps(
                {
                    "arms": len(registration["jobs"]),
                    "allocations": sum(job["segments"] for job in registration["jobs"]),
                    "first_segments_held": True,
                    "max_gpu_hours": registration["budget"]["max_gpu_hours"],
                }
            )
        )
        return
    for name in ("workflow", "logs", "results"):
        (records / name).mkdir(exist_ok=True)
    path = records / "workflow/submissions_grid.json"
    ledger = {
        "registration_sha256": sha256(records / "batch/registration.json"),
        "submissions": [],
    }
    with path.open("x") as stream:
        json.dump(ledger, stream, indent=2)

    def run(command, item):
        ledger["submissions"].append(item)
        path.write_text(json.dumps(ledger, indent=2) + "\n")
        result = subprocess.run(command, text=True, capture_output=True)
        item.update(
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            status="submitted" if result.returncode == 0 else "failed",
        )
        if result.returncode == 0:
            item["job_id"] = result.stdout.strip().split(";")[0]
        path.write_text(json.dumps(ledger, indent=2) + "\n")
        result.check_returncode()
        return item["job_id"]

    for job in registration["jobs"]:
        previous = None
        for segment in range(1, job["segments"] + 1):
            previous = run(
                planned_command(records, job, segment, previous=previous),
                {"arm": job["id"], "segment": segment, "status": "submitting"},
            )
    review_cli = [
        "python",
        "-u",
        "-m",
        "scripts.release_operator_accuracy",
        "--records",
        str(records),
        "--certification-records",
        str(certification_records),
        "--submissions",
        str(path),
    ]
    if runtime_probes:
        review_cli.extend(["--runtime-probes", *map(str, runtime_probes)])
    prefix = f"source ~/scripts/profiles/{registration['runtime_profile']}.sh; export PYTHONPATH={shlex.quote(str(records / 'source/src') + ':' + str(records / 'source'))}; "
    review_command = [
        "sbatch",
        "--parsable",
        "--account=default",
        "--partition=cpu-skylake",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:10:00",
        "--job-name=ma-certification-review",
        f"--chdir={records / 'source'}",
        f"--dependency=afterok:{':'.join(dependencies)}",
        f"--output={records / 'logs/certification_review_%j.out'}",
        "--wrap",
        "bash -c " + shlex.quote(prefix + shlex.join(review_cli)),
    ]
    result = subprocess.run(review_command, text=True, capture_output=True)
    ledger["review_submission"] = {
        "command": review_command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    result.check_returncode()
    print(
        json.dumps(
            {
                "arms": len(registration["jobs"]),
                "allocations": len(ledger["submissions"]),
                "review_job": result.stdout.strip(),
            }
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--certification-records", type=Path, required=True)
    parser.add_argument("--dependencies", nargs="+", required=True)
    parser.add_argument("--runtime-probes", type=Path, nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    submit(
        args.records.resolve(),
        args.certification_records.resolve(),
        args.dependencies,
        args.runtime_probes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
