"""Copy only small authoritative ROSI pilot records; never open checkpoints/data.

Run locally. Each collection gets a new immutable directory. CSVs and bounded
log tails are evidence; scheduler state is recorded separately as workflow.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess


REMOTE = r"""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, subprocess
root = Path.home() / "casus/mandala"
pilots = root / "researchops/runs/operator_benchmark_pilots_v1"
files = {}
def retain(path, label, tail=None, required=False):
    if not path.is_file():
        if required:
            raise FileNotFoundError("Missing required pilot evidence: " + str(path))
        return
    if path.stat().st_size > 4 * 1024**2 and tail is None:
        raise ValueError("Refusing large file: " + str(path))
    with path.open("rb") as stream:
        if tail:
            stream.seek(max(0, path.stat().st_size-tail))
        value = stream.read()
    files[label] = {"source": str(path), "sha256": hashlib.sha256(value).hexdigest(),
                    "content_base64": base64.b64encode(value).decode("ascii"), "tail_bytes": tail}
for name in ("jobs.json", "datasets.json"):
    retain(pilots / name, name, required=True)
jobs = json.loads((pilots / "jobs.json").read_text())
source_manifest = pilots / "source_manifest.json"
if not source_manifest.is_file():
    source_manifest = Path(jobs["pilot_source"]["path"]) / "SOURCE_MANIFEST.json"
retain(source_manifest, "source_manifest.json", required=True)
if json.loads(source_manifest.read_text())["source_snapshot_sha256"] != jobs["pilot_source"]["source_snapshot_sha256"]:
    raise ValueError("Pilot source identity disagrees with jobs manifest")
for name in ("siox_comparison.json",
             "siox_mandala_best_evaluation.json", "siox_mandala_latest_evaluation.json",
             "siox_mandala_metrics.csv"):
    retain(pilots / name, name)
for material in ("graphene", "mos2"):
    for path in sorted((pilots / material / "metrics").glob("version_*/metrics.csv")):
        retain(path, str(path.relative_to(pilots)))
    retain(pilots / material / "metrics/version_0/metrics.csv", material + "/metrics/version_0/metrics.csv", required=True)
    retain(pilots / material / "segment_1.json", material + "/segment_1.json", required=True)
    for path in sorted((pilots / material).glob("segment_*.json")):
        retain(path, str(path.relative_to(pilots)))
    retain(pilots / material / "best_val_evaluation.json", material + "/best_val_evaluation.json")
    models = Path("/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/operator_benchmark_pilots_v1") / material
    for name in ("split_manifest.json", "run_manifest.json"):
        retain(models / name, material + "/" + name, required=True)
logs = set(pilots.glob("*_s1_*.out")) | set(pilots.glob("siox_*.out")) | set(pilots.glob("prepare_*.out"))
for path in sorted(logs):
    retain(path, "log_tails/" + path.name + ".txt", tail=32768)
commands = {
    "startup": ["python3", str(pilots / "check_startup.py")],
    "sacct": ["sacct", "-X", "-j", "786230,786243,786244,786245,786246",
              "--parsable2", "--format=JobIDRaw,State,Elapsed,Timelimit,ExitCode,Partition"],
    "squeue": ["squeue", "-j", "786230,786243,786244,786245,786246", "-o", "%.18i %.18P %.32j %.12T %.12M %.12l %R"],
}
workflow = {}
for label, command in commands.items():
    result = subprocess.run(command, text=True, capture_output=True)
    workflow[label] = {"command": command, "returncode": result.returncode,
                       "stdout": result.stdout, "stderr": result.stderr}
print(json.dumps({"captured_utc": datetime.now(timezone.utc).isoformat(),
                  "files": files, "workflow": workflow}))
"""


def collect(host, destination):
    if destination.exists():
        raise FileExistsError(destination)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "python3 -"],
        input=REMOTE,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    destination.mkdir(parents=True)
    provenance = {"captured_utc": payload["captured_utc"], "host": host, "files": {}}
    for label, record in payload["files"].items():
        path = destination / "evidence" / label
        if not path.resolve().is_relative_to((destination / "evidence").resolve()):
            raise ValueError("Invalid remote label")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(record.pop("content_base64"), validate=True)
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Remote evidence checksum mismatch: " + label)
        path.write_bytes(data)
        provenance["files"][label] = record
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (destination / "workflow.json").write_text(
        json.dumps(payload["workflow"], indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "captured_utc": payload["captured_utc"],
                "files": len(provenance["files"]),
            },
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="rosi4")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    collect(args.host, args.destination)


if __name__ == "__main__":
    main()
