#!/usr/bin/env python3
"""Run pytest node IDs in isolated processes and record timing audit results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["tests"])
    parser.add_argument("--output-dir", type=Path, default=Path("test_audit"))
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pattern", help="Regex applied to collected node IDs")
    parser.add_argument("--max-tests", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed records from an existing timings.json file",
    )
    return parser.parse_args()


def _pytest_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-o", "addopts=", *args]


def _collect_node_ids(paths: list[str]) -> list[str]:
    result = subprocess.run(
        _pytest_command("--collect-only", "-q", *paths),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pytest collection failed:\n{result.stdout}")
    node_ids = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if not node_ids:
        raise RuntimeError("pytest collection returned no test node IDs")
    return node_ids


def _run_node(node_id: str, timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            _pytest_command("-q", node_id),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        output = result.stdout
        if result.returncode == 0:
            status = "skipped" if re.search(r"\bskipped\b", output) else "passed"
        else:
            status = "failed"
        return {
            "node_id": node_id,
            "status": status,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
            "output": output if status != "passed" else "",
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "node_id": node_id,
            "status": "timeout",
            "elapsed_seconds": elapsed,
            "returncode": None,
            "output": output,
        }


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {record["node_id"]: record for record in payload.get("records", [])}


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    within_60 = sum(record["elapsed_seconds"] <= 60 for record in records)
    within_300 = sum(record["elapsed_seconds"] <= 300 for record in records)
    statuses: dict[str, int] = {}
    for record in records:
        status = record["status"]
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "total": total,
        "statuses": statuses,
        "within_60_seconds": within_60,
        "within_60_percent": 100.0 * within_60 / total if total else 0.0,
        "within_300_seconds": within_300,
        "within_300_percent": 100.0 * within_300 / total if total else 0.0,
    }


def _write_results(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda record: record["node_id"])
    payload = {"summary": _summary(records), "records": records}
    (output_dir / "timings.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output_dir / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "status", "elapsed_seconds", "returncode"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        raise ValueError("--timeout-seconds must be in the interval (0, 300]")

    node_ids = _collect_node_ids(args.paths)
    if args.pattern:
        regex = re.compile(args.pattern)
        node_ids = [node_id for node_id in node_ids if regex.search(node_id)]
    if args.max_tests is not None:
        node_ids = node_ids[: args.max_tests]

    output_path = args.output_dir / "timings.json"
    existing = _load_existing(output_path) if args.resume else {}
    records = [existing[node_id] for node_id in node_ids if node_id in existing]
    pending = [node_id for node_id in node_ids if node_id not in existing]
    print(
        f"Collected {len(node_ids)} test(s): {len(records)} resumed, "
        f"{len(pending)} pending; workers={args.workers}, "
        f"timeout={args.timeout_seconds:.0f}s",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_node, node_id, args.timeout_seconds): node_id
            for node_id in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(pending)}] {record['status']:7s} "
                f"{record['elapsed_seconds']:8.2f}s {record['node_id']}",
                flush=True,
            )
            _write_results(args.output_dir, records)

    _write_results(args.output_dir, records)
    summary = _summary(records)
    print(json.dumps(summary, indent=2))
    return (
        1
        if summary["statuses"].get("failed", 0) or summary["statuses"].get("timeout", 0)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
