"""Submit three independent four-segment publication retrain chains on ROSI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--materials", nargs="+", choices=("silicon", "sio2", "zncusnses"), default=["silicon", "sio2", "zncusnses"])
    parser.add_argument("--append", action="store_true", help="Preserve prior submissions and append replacement chains")
    parser.add_argument("--first-segment", type=int, default=1, choices=(1, 2, 3, 4), help="Resume a chain at this segment (earlier segments already ran)")
    parser.add_argument("--partition", help="Skip the availability probe and use this partition (account casus for -casus partitions)")
    args = parser.parse_args()
    destination = RUN / "submissions.json"
    if destination.exists() and not args.plan and not args.append:
        raise SystemExit("submissions.json already exists; refusing duplicate chains")
    inventory = output(["sinfo", "-h", "-o", "%P|%a|%T"])
    choices = (
        ("gpu-b200-casus", "casus"),
        ("gpu-h100-casus", "casus"),
        ("gpu-h100", "default"),
    )
    for partition, account in choices:
        if args.partition:
            partition = args.partition
            account = "casus" if partition.endswith("-casus") else "default"
            break
        if any(
            name.rstrip("*") == partition and available == "up"
            and not any(word in state.lower() for word in
                        ("drain", "down", "fail", "inval", "maint"))
            for name, available, state in
            (line.split("|") for line in inventory.splitlines())
        ):
            break
    else:
        raise SystemExit("No eligible GPU partition is available")
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(ROOT),
        "source_commit": output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "partition": partition,
        "account": account,
        "sinfo": inventory,
        "jobs": [],
    }
    if args.append and destination.exists():
        previous_record = json.loads(destination.read_text())
        record["previous_submissions"] = previous_record.get("previous_submissions", []) + [{key: value for key, value in previous_record.items() if key not in ("jobs", "previous_submissions")} ]
        record["jobs"] = previous_record["jobs"]
    if not args.plan:
        (RUN / "logs").mkdir(exist_ok=True)
        destination.write_text(json.dumps(record, indent=2) + "\n")
    for material in dict.fromkeys(args.materials):
        previous = None
        for segment in range(args.first_segment, 5):
            name = f"r6-{material}-{segment}"
            argv = [
                "sbatch", "--parsable", f"--job-name={name}",
                f"--partition={partition}", f"--account={account}",
                "--nodes=1", "--ntasks=1", "--gres=gpu:1",
                "--cpus-per-task=4", "--mem=128G", "--time=45:00:00",
                "--nice=1000", "--signal=INT@600", f"--chdir={ROOT}",
                f"--output={RUN / 'logs' / (name + '-%j.log')}",
                "--open-mode=append",
            ]
            if previous:
                argv.append(f"--dependency=afterok:{previous}")
            argv.extend([str(RUN / "segment.sh"), material])
            job_id = f"PLAN-{material}-{segment}" if args.plan else output(argv).split(";")[0]
            if not args.plan and not job_id.isdigit():
                raise RuntimeError(f"Unrecognized sbatch job id: {job_id}")
            record["jobs"].append({
                "material": material, "segment": segment, "job_id": job_id,
                "dependency": f"afterok:{previous}" if previous else None,
                "sbatch_argv": argv,
            })
            if not args.plan:
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, indent=2) + "\n")
                temporary.replace(destination)
            previous = job_id
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
