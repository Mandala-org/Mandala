#!/usr/bin/env python3
"""Summarize frozen common-split runs with the standard library only.

Local: script.py /tmp/mf_snapshot --output-dir OUTPUT --interim
Remote: script.py user@host:/run/path --snapshot-dir /tmp/mf_snapshot --interim
Only segment/terminal JSON, horizon, and metrics CSV are rsynced. Remote logs
are reduced in place to completed-epoch timings; no checkpoints are copied.

The shared trainer's val/hamiltonian_mae_elementwise is a per-structure
observed-element MAE, epoch-averaged in eV (src/net/e3gnn.py and
src/net/operator_metrics.py). Legacy val/hamiltonian_mae has different reductions
and is deliberately never a fallback. Explicit _mev/_ev/_hartree variants are
supported. Projected metrics are independent series, not inferred from raw MAE.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

ARMS = [
    f"{model}_{material}"
    for material in ("graphene", "mos2")
    for model in ("mandala", "deeph_e3", "mace_h")
]
# This source is also executed remotely, producing a bounded JSON timing file.
TIMING_CODE = r"""
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
output = {}
for p in sorted((root / 'logs').glob('*.out')):
    if not re.search(r'_[12]_\d+\.out$', p.name):
        continue
    times = {}
    with p.open(errors='replace') as f:
        for line in f:
            for m in re.finditer(r'Epoch\s+(\d+):\s+100%[^\r\n]*?\[([\d:]+)<', line):
                seconds = 0
                for part in m[2].split(':'):
                    seconds = 60 * seconds + int(part)
                times[m[1]] = max(times.get(m[1], 0), seconds)
    output[p.name] = times
print(json.dumps(output))
"""


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def collect_remote(source, destination):
    host, remote = source.split(":", 1)
    if not host or host.startswith("-") or not remote:
        raise ValueError("Expected user@host:/absolute/run/path")
    destination.mkdir(parents=True, exist_ok=True)
    includes = [
        "*/",
        "HORIZON_FREEZE.json",
        "segment_[12].json",
        "terminal_test.json",
        "terminal_comparison.json",
        "metrics.csv",
    ]
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--prune-empty-dirs",
            "--exclude=/logs/",
            "--exclude=/results/probe/",
            *[f"--include={x}" for x in includes],
            "--exclude=*",
            "--",
            source.rstrip("/") + "/",
            str(destination) + "/",
        ],
        check=True,
    )
    # Reuse configured SSH ControlMaster. Never fetch full multi-MB tqdm logs.
    command = "python3 -c " + shlex.quote(TIMING_CODE) + " " + shlex.quote(remote)
    result = subprocess.run(
        ["ssh", host, command], check=True, capture_output=True, text=True
    )
    timing = json.loads(result.stdout)
    (destination / "epoch_timings.json").write_text(json.dumps(timing, indent=2) + "\n")


def timings(root):
    saved = read_json(root / "epoch_timings.json")
    if saved is not None:
        return saved
    result = subprocess.run(
        ["python3", "-c", TIMING_CODE, str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def metric_column(headers, projected):
    base = "val/hamiltonian_mae_elementwise" + ("_projected" if projected else "")
    for suffix, factor in [
        ("", 1000.0),
        ("_mev", 1.0),
        ("_ev", 1000.0),
        ("_hartree", 27211.4),
    ]:
        if base + suffix in headers:
            return base + suffix, factor
    return None, None


def series_summary(points, targets):
    if not points:
        return None
    points = sorted(points.values(), key=lambda x: x["epoch"])
    matched = {}
    for target in targets:
        nearest = min(points, key=lambda x: (abs(x["epoch"] - target), x["epoch"]))
        matched[str(target)] = dict(
            nearest,
            requested_epoch=target,
            epoch_delta=nearest["epoch"] - target,
            target_reached=points[-1]["epoch"] >= target,
        )
    return dict(
        best=min(points, key=lambda x: (x["mae_mev"], x["epoch"])),
        last=points[-1],
        matched_epochs=matched,
        logged_epochs=len(points),
    )


def terminal_scores(document):
    if document is None:
        return None
    # run_fixed_recipe_benchmark.score writes metrics.<view>.structure_mean_mae_mev.
    metrics = document.get("metrics", {})
    result = {}
    for view in ("raw", "projected"):
        value = metrics.get(view, {}).get("structure_mean_mae_mev")
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Nonfinite terminal MAE")
            result[view] = value
    if not result:
        raise ValueError(
            "Unrecognized terminal_test.json schema; refusing to guess units"
        )
    return dict(
        epoch=document.get("epoch"), updates=document.get("updates"), mae_mev=result
    )


def summarize(root, interim=False, timestamp=None):
    timing = timings(root)
    result = dict(
        schema_version=1,
        snapshot_utc=timestamp or datetime.now(timezone.utc).isoformat(),
        status="interim" if interim else "snapshot",
        horizon_hours=90,
        horizon_freeze=read_json(root / "HORIZON_FREEZE.json"),
        arms={},
        sources={},
    )
    for arm in ARMS:
        directory = root / "results" / arm
        targets = [350, 700] if arm.endswith("graphene") else [175, 350]
        segments = {str(i): read_json(directory / f"segment_{i}.json") for i in (1, 2)}
        endpoint = segments["2"]["epochs"] if segments["2"] else None
        series = {"raw": {}, "projected": {}}
        metric_sources = []
        paths = sorted(
            directory.glob("metrics/version_*/metrics.csv"),
            key=lambda p: int(p.parent.name.split("_")[-1]),
        )
        for path in paths:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                columns = {
                    view: metric_column(headers, view == "projected") for view in series
                }
                if columns["raw"][0] is None:
                    raise ValueError(f"No common elementwise Hamiltonian MAE in {path}")
                metric_sources.append(
                    dict(
                        path=str(path.relative_to(root)),
                        columns={
                            v: dict(column=c, multiplier_to_mev=m)
                            for v, (c, m) in columns.items()
                        },
                    )
                )
                for row in reader:
                    if not row.get("epoch"):
                        continue
                    # CSV epochs are zero-based; segment epochs count completed epochs.
                    epoch = int(float(row["epoch"])) + 1
                    if endpoint is not None and epoch > endpoint:
                        continue
                    for view, (column, multiplier) in columns.items():
                        if column is None or not row.get(column):
                            continue
                        value = float(row[column]) * multiplier
                        if not math.isfinite(value):
                            raise ValueError(f"Nonfinite {column} at {path}:{epoch}")
                        series[view][epoch] = dict(
                            epoch=epoch,
                            logged_epoch=epoch - 1,
                            mae_mev=value,
                            step=row.get("step"),
                            source=str(path.relative_to(root)),
                        )
        timing_endpoint = (
            endpoint if endpoint is not None else max(series["raw"], default=0)
        )
        duration_by_epoch = {}
        for name, values in timing.items():
            if re.fullmatch(re.escape(arm) + r"_[12]_\d+\.out", name):
                for epoch, seconds in values.items():
                    completed = int(epoch) + 1
                    if completed <= timing_endpoint:
                        duration_by_epoch[completed] = max(
                            duration_by_epoch.get(completed, 0), seconds
                        )
        durations = list(duration_by_epoch.values())
        result["arms"][arm] = dict(
            segments=segments,
            epochs_at_90h=endpoint,
            last_logged_completed_epoch=max(series["raw"], default=None),
            mean_minutes_per_epoch=(
                sum(durations) / len(durations) / 60 if durations else None
            ),
            timing_epoch_count=len(durations),
            timing_source="maximum full-epoch tqdm elapsed time per epoch; includes validation",
            validation={v: series_summary(p, targets) for v, p in series.items()},
            metric_sources=metric_sources,
            terminal_test=terminal_scores(read_json(directory / "terminal_test.json")),
        )
        for path in [
            *paths,
            *(directory.glob("segment_[12].json")),
            directory / "terminal_test.json",
        ]:
            if path.exists():
                result["sources"][str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    result["terminal_comparison"] = read_json(root / "results/terminal_comparison.json")
    for path in [root / "HORIZON_FREEZE.json", root / "epoch_timings.json"]:
        if path.exists():
            result["sources"][str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def fmt(value):
    return "pending" if value is None else f"{value:.4g}"


def score_cell(point):
    return (
        "unavailable" if point is None else f"{point['mae_mev']:.4g} @ {point['epoch']}"
    )


def markdown(result):
    lines = [
        f"# Fixed-recipe benchmark — {result['status']} snapshot",
        "",
        f"Snapshot UTC: {result['snapshot_utc']}",
        "",
        "Frozen horizon: segment 2, 90 H100 hours per arm. Pending endpoints are not "
        "estimated from the freeze-decision epoch. No test scores were used to select a checkpoint.",
        "",
        "All MAEs below are Hamiltonian errors in meV. Validation uses "
        "`val/hamiltonian_mae_elementwise` (eV × 1000), not the legacy pair-summed "
        "`val/hamiltonian_mae`. Explicit unit-suffixed columns use meV × 1, eV × 1000, "
        "or Hartree × 27211.4; selected columns and conversion factors are in JSON. "
        "Elementwise errors are averaged per structure, then over the validation epoch. "
        "Projected validation is shown only where explicitly logged. Terminal scores "
        "use `metrics.{raw,projected}.structure_mean_mae_mev`.",
        "",
        "Epochs below count completed epochs (CSV epoch + 1); segment JSON already uses "
        "completed epochs. Values are `MAE @ completed epoch`. Nearest matches break ties "
        "toward the earlier epoch; † means the target has not yet been reached. "
        "Best validation is descriptive, not terminal checkpoint selection. Timing averages "
        "deduplicated full-epoch tqdm elapsed times, including validation and excluding "
        "startup/checkpoint overhead; no timing is inferred from untimestamped CSV rows.",
        "",
    ]
    for material, targets in [("graphene", [350, 700]), ("mos2", [175, 350])]:
        lines += [
            f"## {material}",
            "",
            "| Arm | Segment 1 epochs | Segment 2 / 90 h epochs | Last logged epoch | min/epoch (n) | Best raw val | Last raw val | "
            + " | ".join(f"Raw val ~{t}" for t in targets)
            + " | Best projected val | Last projected val | "
            + " | ".join(f"Projected val ~{t}" for t in targets)
            + " | Test projected | Test raw |",
            "| " + " | ".join(["---"] * 15) + " |",
        ]
        for arm in ARMS:
            if not arm.endswith(material):
                continue
            a = result["arms"][arm]
            raw, projected = a["validation"]["raw"], a["validation"]["projected"]

            def matches(s):
                return [
                    (
                        "unavailable"
                        if s is None
                        else score_cell(s["matched_epochs"][str(t)])
                        + (
                            ""
                            if s["matched_epochs"][str(t)]["target_reached"]
                            else " †"
                        )
                    )
                    for t in targets
                ]

            test = (a["terminal_test"] or {}).get("mae_mev", {})
            cells = [
                arm,
                fmt((a["segments"]["1"] or {}).get("epochs")),
                fmt(a["epochs_at_90h"]),
                fmt(a["last_logged_completed_epoch"]),
                f"{fmt(a['mean_minutes_per_epoch'])} ({a['timing_epoch_count']})",
                score_cell(raw["best"]) if raw else "unavailable",
                score_cell(raw["last"]) if raw else "unavailable",
                *matches(raw),
                score_cell(projected["best"]) if projected else "unavailable",
                score_cell(projected["last"]) if projected else "unavailable",
                *matches(projected),
                fmt(test.get("projected")),
                fmt(test.get("raw")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--interim", action="store_true")
    parser.add_argument(
        "--timestamp", help="UTC snapshot timestamp, for reproducible rendering"
    )
    args = parser.parse_args()
    with TemporaryDirectory(prefix="mandala-benchmark-") as temporary:
        root = Path(args.run_directory)
        if ":" in args.run_directory:
            root = args.snapshot_dir or Path(temporary)
            collect_remote(args.run_directory, root)
        result = summarize(root, args.interim, args.timestamp)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "benchmark_summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        (args.output_dir / "benchmark_summary.md").write_text(markdown(result))


if __name__ == "__main__":
    main()
