#!/usr/bin/env python
"""Reaudit frozen silicon H/S/D exports after overlap-only eigenvalue flooring.

No inference, occupation projection, upper eigenvalue clamp or jitter is used.
The producer contract binds ordered inputs; export manifests bind raw operators.
The unrepaired scenario is repeated and checked against the historical audit.
Only S^h(k) on reference support is repaired, after the usual Fourier assembly.
Real-space MAEs and exact Hermitian Tr(DS) keep the original, unrepaired S.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from analysis.physical_constraints import (
    BlockFourier,
    diagnose_kpoint,
    fermi_consistency,
    hermitian,
    public_record,
    repair_overlap,
    summarize_records,
    uniform_kpoints,
)
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot
from scripts.report.evaluate_physical_constraints import (
    matrix_error,
    restrict_support,
    sha256,
    sparse_trace,
    write_json,
)
from scripts.report.summarize_physical_constraints import summarize

BEFORE = "joint_predicted_reference_support"
AFTER = BEFORE + "_overlap_repaired"


def assert_reproduced(actual, expected, path=""):
    """Check the archived scalars, including exact counts and failure rates."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_reproduced(actual[key], value, path + "." + key)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(
                f"Historical reproduction failed at {path}: {actual} != {expected}"
            )
    elif actual != expected:
        raise ValueError(
            f"Historical reproduction failed at {path}: {actual} != {expected}"
        )


def parse_repair(value):
    try:
        name, floor = value.split("=")
        floor = float(floor)
        if name != "eig_floor" or not math.isfinite(floor) or floor <= 0:
            raise ValueError
        return floor
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Expected eig_floor=<positive finite value>"
        ) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-contract", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-aggregate", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Prefix for relocated absolute producer input paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap-repair", type=parse_repair, default="eig_floor=1e-6")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.threads <= 32:
        parser.error("Use 1 to 32 CPU threads")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory: " + str(args.output_dir))
    producer = json.loads(args.producer_contract.read_text())
    baseline = json.loads(args.baseline_summary.read_text())
    aggregates = json.loads(args.baseline_aggregate.read_text())["evaluations"]
    selected = [v for k, v in aggregates.items() if k.endswith("/silicon_historical")]
    if producer["arguments"]["material"] != "silicon" or len(selected) != 1:
        raise ValueError("This control requires the joint historical silicon audit")
    baseline_aggregate = selected[0]
    pairs = producer["split"]["validation"]
    if baseline["status"] != "complete" or baseline["structures"] != len(pairs):
        raise ValueError("Incomplete historical audit")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    source_paths = [
        "src/analysis/physical_constraints.py",
        "scripts/report/evaluate_overlap_repair.py",
        "scripts/report/evaluate_physical_constraints.py",
        "scripts/report/summarize_physical_constraints.py",
        "src/data/block_matrix.py",
        "src/data/snapshot.py",
        "src/data/openmx_parser.py",
    ]
    contract = {
        "schema_version": 1,
        "arguments": vars(args),
        "producer_contract_sha256": sha256(args.producer_contract),
        "baseline_summary_sha256": sha256(args.baseline_summary),
        "baseline_aggregate_sha256": sha256(args.baseline_aggregate),
        "checkpoint_sha256": producer["checkpoint_sha256"],
        "source_sha256": {p: sha256(ROOT / p) for p in source_paths},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "no_inference_or_model_selection": True,
        "scenario": AFTER,
        "repair": {
            "eig_floor": args.overlap_repair,
            "upper_clamp": False,
            "jitter": False,
        },
        "occupation_projection": False,
        "temperature_note": "300 K Fermi-Dirac data; no fixed-charge zero-temperature occupation projection",
        "realspace_trace": "unrepaired Hermitian real-space S and D; no back-transform",
        "reference_columns": "unchanged historical reference_full; overlap repair control on structure_0000 only",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.threads,
        },
    }
    write_json(args.output_dir / "contract.json", contract)

    def local(path):
        return args.input_root / path.lstrip("/") if args.input_root else Path(path)

    results, correction_rows = [], []
    reference_control = None
    for index, (matrix_path, info_path) in enumerate(pairs):
        print(f"Overlap repair {index + 1}/{len(pairs)}: {matrix_path}", flush=True)
        old = baseline["results"][index]
        if old["matrix_path"] != matrix_path or old["info_path"] != info_path:
            raise ValueError("Producer and baseline structure order differ")
        for path, digest in producer["input_sha256"][index].items():
            if sha256(local(path)) != digest:
                raise ValueError("Producer input changed: " + path)
        cache = args.operator_cache / f"structure_{index:04d}"
        exported = json.loads((cache / "export.json").read_text())
        if not exported["strict_load"] or not exported["implementation"].startswith(
            producer["arguments"]["prediction_source_root"] + "/"
        ):
            raise ValueError(
                "Export does not identify the producer's historical implementation"
            )
        predictions = {}
        for key in ("hamiltonian", "overlap", "density"):
            path = cache / f"{key}.pt"
            if sha256(path) != exported["operator_sha256"][key]:
                raise ValueError("Frozen operator changed: " + str(path))
            predictions[key] = BlockMatrix.load(path)
        reference = Snapshot.from_openmx(local(matrix_path), local(info_path))
        refs = {
            "hamiltonian": reference.hamiltonian,
            "overlap": reference.overlap,
            "density": reference.density * 2,
        }
        # Reproduce the archived scalar element errors, but never repair real-space blocks.
        for key, value in predictions.items():
            errors = matrix_error((value + value.adjoint()) * 0.5, refs[key])
            assert_reproduced(
                errors["mae"],
                old["matrix_errors_hermitian_common_support"][key]["mae"],
                key + ".mae",
            )
        predictions = {
            key: restrict_support(value, refs[key])
            for key, value in predictions.items()
        }
        assemblers = {key: BlockFourier(value) for key, value in predictions.items()}
        reference_s = BlockFourier(reference.overlap) if index == 0 else None
        metadata = old["input_metadata"]
        kpoints = uniform_kpoints(old["mesh"])
        records = {BEFORE: [], AFTER: []}
        control_records, corrections, unrepaired_traces = [], [], []
        for k in kpoints:
            h, s, d = (
                assemblers[key].at(k) for key in ("hamiltonian", "overlap", "density")
            )
            unrepaired_traces.append(
                float(torch.trace(hermitian(d) @ hermitian(s)).real)
            )
            for label, floor in ((BEFORE, None), (AFTER, args.overlap_repair)):
                record = diagnose_kpoint(h, s, d, overlap_eig_floor=floor)
                record["fractional_k"] = k
                if "density" in record:
                    error = (
                        record["density"]["electron_trace"] - metadata["num_electrons"]
                    )
                    record["density"]["electron_count_error"] = error
                    record["density"]["electron_count_absolute_error"] = abs(error)
                records[label].append(record)
            corrections.append(
                {
                    "status": "ok",
                    "fractional_k": k,
                    **records[AFTER][-1]["overlap_repair"],
                }
            )
            if reference_s is not None:
                _, control = repair_overlap(
                    reference_s.at(k), eig_floor=args.overlap_repair
                )
                control_records.append({"status": "ok", "fractional_k": k, **control})
        realspace_trace = sparse_trace(
            (predictions["density"] + predictions["density"].adjoint()) * 0.5,
            (predictions["overlap"] + predictions["overlap"].adjoint()) * 0.5,
        )
        row = {
            "matrix_path": matrix_path,
            "info_path": info_path,
            "input_metadata": metadata,
            "mesh": old["mesh"],
            "kpoint_weight": 1 / len(kpoints),
            "matrix_errors_raw_common_support": old["matrix_errors_raw_common_support"],
            "matrix_errors_hermitian_common_support": old[
                "matrix_errors_hermitian_common_support"
            ],
            "operator_sha256": exported["operator_sha256"],
            "export_manifest_sha256": sha256(cache / "export.json"),
            "scenarios": {},
        }
        for label, values in records.items():
            scenario = {
                "operators": {
                    key: "prediction_on_reference_support" for key in predictions
                },
                "summary": summarize_records(values),
                "hermitian_realspace_electron_trace": realspace_trace,
                "realspace_overlap_repaired": False,
                "fermi_consistency": fermi_consistency(
                    values,
                    num_electrons=metadata["num_electrons"],
                    temperature_kelvin=metadata["temperature_kelvin"],
                ),
                "kpoints": [public_record(r) for r in values],
            }
            if label == BEFORE:
                assert_reproduced(
                    scenario["summary"], old["scenarios"][BEFORE]["summary"]
                )
                assert_reproduced(
                    realspace_trace,
                    old["scenarios"][BEFORE]["hermitian_realspace_electron_trace"],
                )
                assert_reproduced(
                    scenario["fermi_consistency"],
                    old["scenarios"][BEFORE]["fermi_consistency"],
                )
                traces = unrepaired_traces
            else:
                scenario["operators"]["overlap"] += "_overlap_repaired"
                traces = (
                    [r["density"]["electron_trace"] for r in values]
                    if all("density" in r for r in values)
                    else []
                )
            if traces:
                scenario["mesh_electron_count"] = sum(traces) / len(traces)
                scenario["mesh_electron_count_error"] = (
                    scenario["mesh_electron_count"] - metadata["num_electrons"]
                )
            row["scenarios"][label] = scenario
        if control_records:
            reference_control = {
                "structure_index": index,
                "matrix_path": matrix_path,
                "summary": summarize_records(control_records),
                "kpoints": control_records,
            }
            if any(r["floored_eigenvalues"] for r in control_records):
                raise ValueError("Reference overlap unexpectedly requires repair")
        correction_rows.append(
            {
                "structure_index": index,
                "matrix_path": matrix_path,
                "summary": summarize_records(corrections),
                "kpoints": corrections,
            }
        )
        write_json(args.output_dir / f"structure_{index:04d}.json", row)
        results.append(
            {
                **row,
                "scenarios": {
                    key: {k: v for k, v in scenario.items() if k != "kpoints"}
                    for key, scenario in row["scenarios"].items()
                },
            }
        )
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "material": "silicon",
            "structures": len(results),
            "elapsed_seconds": time.monotonic() - start,
            "results": results,
        },
    )
    aggregate = summarize(args.output_dir)
    # Historical reference columns are deliberately kept byte-for-byte as values.
    aggregate["scenarios"]["reference_full"] = deepcopy(
        baseline_aggregate["scenarios"]["reference_full"]
    )
    assert_reproduced(
        aggregate["scenarios"][BEFORE], baseline_aggregate["scenarios"][BEFORE]
    )
    all_corrections = [r for row in correction_rows for r in row["kpoints"]]
    write_json(
        args.output_dir / "corrections.json",
        {
            "scenario": AFTER,
            "eig_floor": args.overlap_repair,
            "summary": summarize_records(all_corrections),
            "total_floored_eigenvalues": sum(
                r["floored_eigenvalues"] for r in all_corrections
            ),
            "structures": correction_rows,
            "reference_control": reference_control,
        },
    )
    write_json(
        args.output_dir / "aggregate.json",
        {"schema_version": 1, "evaluations": {str(args.output_dir): aggregate}},
    )
    print(
        f"Completed {len(results)} structures in {time.monotonic() - start:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
