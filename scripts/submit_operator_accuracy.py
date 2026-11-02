"""Submit the registered investigation on ROSI; no scientific work on login.

Audit jobs come first. Screening requires an explicit certification review
binding both successful audit result hashes to the immutable registration.
Existing pilot jobs/checkpoints are never modified. No automatic retries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from scripts.prepare_operator_accuracy import gate, sha256, verify


def commands(
    records,
    phase,
    *,
    partition="gpu-h100-casus",
    nodelist=None,
    account="casus",
):
    registration = json.loads((records / "batch/registration.json").read_text())
    common = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--partition={partition}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=32",
        "--mem=192G",
        "--gres=gpu:1",
        f"--chdir={records / 'source'}",
    ]
    if nodelist is not None:
        common.append(f"--nodelist={nodelist}")
    if phase == "audits":
        return [
            common
            + [
                "--time=02:00:00",
                f"--job-name=m-accuracy-audit-{item['material']}",
                f"--output={records / 'logs' / (item['material'] + '_audit_%j.out')}",
                str(records / "batch" / f"audit_{item['material']}.sh"),
            ]
            for item in registration["audits"]
        ]
    if registration["budget"].get("segments_per_candidate", 1) != 1:
        raise ValueError(
            "Multi-segment registrations require individually scheduled frozen "
            "segment launchers; screen arrays are disabled"
        )
    hours = registration["budget"].get("allocation_hours_per_segment", 6)
    concurrency = registration["budget"].get("array_concurrency", 2)
    return [
        common
        + [
            f"--time={hours:02d}:00:00",
            "--signal=INT@600",
            "--job-name=m-accuracy-screen",
            f"--array=0-{len(registration['jobs'])-1}%{concurrency}",
            f"--output={records / 'logs/screen_%A_%a.out'}",
            str(records / "batch/screen_array.sh"),
        ]
    ]


def submit(
    records,
    phase,
    *,
    dry_run=False,
    partition="gpu-h100-casus",
    nodelist=None,
    account="casus",
):
    verify(records)
    if phase == "screen":
        gate(records)
    planned = commands(
        records, phase, partition=partition, nodelist=nodelist, account=account
    )
    if dry_run:
        print(json.dumps(planned, indent=2))
        return
    (records / "workflow").mkdir(exist_ok=True)
    (records / "logs").mkdir(exist_ok=True)
    (records / "results").mkdir(exist_ok=True)
    path = records / "workflow" / f"submissions_{phase}.json"
    record = {
        "phase": phase,
        "registration_sha256": sha256(records / "batch/registration.json"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "submissions": [],
    }
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    for command in planned:
        item = {"command": command, "status": "submitting"}
        record["submissions"].append(item)
        path.write_text(json.dumps(record, indent=2) + "\n")
        result = subprocess.run(command, text=True, capture_output=True)
        item.update(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            status="submitted" if result.returncode == 0 else "failed",
        )
        if result.returncode == 0:
            item["job_id"] = result.stdout.strip().split(";")[0]
        path.write_text(json.dumps(record, indent=2) + "\n")
        result.check_returncode()
    print(json.dumps(record, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--phase", choices=("audits", "screen"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--partition", default="gpu-h100-casus")
    parser.add_argument("--nodelist")
    parser.add_argument("--account", default="casus")
    args = parser.parse_args(argv)
    submit(
        args.records.resolve(),
        args.phase,
        dry_run=args.dry_run,
        partition=args.partition,
        nodelist=args.nodelist,
        account=args.account,
    )


if __name__ == "__main__":
    main()
