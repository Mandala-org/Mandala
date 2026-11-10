#!/usr/bin/env python
"""Aggregate bounded structure diagnostics; retain failures and denominators."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from analysis.physical_constraints import summarize_records


def summarize(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError(f"Incomplete result: {directory}")
    records, structures = defaultdict(list), defaultdict(list)
    for path in sorted(directory.glob("structure_*.json")):
        row = json.loads(path.read_text())
        for label, scenario in row["scenarios"].items():
            records[label].extend(scenario["kpoints"])
            structures[label].append((scenario, row["input_metadata"]["num_electrons"]))
    if len(list(directory.glob("structure_*.json"))) != summary["structures"]:
        raise ValueError("Missing structure records")
    result = {
        "material": summary["material"],
        "structures": summary["structures"],
        "scenarios": {},
    }
    for label, values in records.items():
        scenarios = structures[label]
        result["scenarios"][label] = summarize_records(values)
        target = result["scenarios"][label]
        target["structures_with_invalid_metric"] = sum(
            s["summary"]["invalid_kpoint_count"] > 0 for s, _ in scenarios
        )
        target["structures_with_occupation_violation"] = sum(
            any(
                r.get("density", {}).get("negative_count", 0)
                or r.get("density", {}).get("overoccupied_count", 0)
                for r in s["kpoints"]
            )
            for s, _ in scenarios
        )
        errors = [
            abs(s["hermitian_realspace_electron_trace"]["real"] - n)
            for s, n in scenarios
        ]
        target["electron_count_absolute_error"] = {
            "count": len(errors),
            "mean": sum(errors) / len(errors),
            "max": max(errors),
        }
        # This is a separate quantity from the exact unrepaired real-space
        # trace above. In overlap-repair controls it uses S_rep on the mesh.
        mesh_errors = [
            abs(s["mesh_electron_count_error"])
            for s, _ in scenarios
            if "mesh_electron_count_error" in s
        ]
        if mesh_errors:
            target["mesh_electron_count_absolute_error"] = {
                "count": len(mesh_errors),
                "mean": sum(mesh_errors) / len(mesh_errors),
                "max": max(mesh_errors),
            }
        fd = [
            s["fermi_consistency"]["density_relative_frobenius"]
            for s, _ in scenarios
            if "density_relative_frobenius" in s["fermi_consistency"]
        ]
        target["finite_temperature_density_relative_frobenius"] = {
            "count": len(fd),
            "mean": sum(fd) / len(fd) if fd else None,
            "max": max(fd) if fd else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    result = {
        "schema_version": 1,
        "evaluations": {str(path): summarize(path) for path in args.inputs},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
