#!/usr/bin/env python
"""Postprocess immutable raw density exports; no new inference or model choice.

The producer contract fixes checkpoint, ordered structures and operator hashes.
This control tests predicted D on observed support with reference H and S, and
repeats the reference baseline to verify consistency with the initial audit.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from analysis.physical_constraints import (
    BlockFourier,
    diagnose_kpoint,
    fermi_consistency,
    public_record,
    summarize_records,
    uniform_kpoints,
)
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot
from scripts.report.evaluate_physical_constraints import (
    input_metadata,
    restrict_support,
    sha256,
    sparse_trace,
    write_json,
)
from scripts.report.summarize_physical_constraints import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-contract", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # The fixed 104-AO problem is dominated by small eigensolves; a measured
    # Genoa run showed severe four-thread overhead. GPU inference is absent.
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    producer = json.loads(args.producer_contract.read_text())
    if producer["arguments"]["material"] != "silicon":
        raise ValueError(
            "This registered control is restricted to the joint silicon model"
        )
    pairs = [(Path(a), Path(b)) for a, b in producer["split"]["validation"]]
    contract = {
        "executor": "ROSI compute",
        "kind": "calculate",
        "description": "Density-only native-support control with reference metric",
        "producer_contract_sha256": sha256(args.producer_contract),
        "checkpoint_sha256": producer["checkpoint_sha256"],
        "input_sha256": producer["input_sha256"],
        "arguments": vars(args),
        "no_model_inference_or_selection": True,
        "source_manifest_sha256": (
            sha256(ROOT / "source_manifest.json")
            if (ROOT / "source_manifest.json").is_file()
            else None
        ),
        "torch": torch.__version__,
        "structures": len(pairs),
    }
    write_json(args.output_dir / "contract.json", contract)
    results = []
    for index, (matrix_path, info_path) in enumerate(pairs):
        print(
            f"Density support control {index + 1}/{len(pairs)}: {matrix_path}",
            flush=True,
        )
        for path, digest in producer["input_sha256"][index].items():
            if sha256(path) != digest:
                raise RuntimeError("Producer input changed: " + path)
        cache = args.operator_cache / f"structure_{index:04d}"
        exported = json.loads((cache / "export.json").read_text())
        density_path = cache / "density.pt"
        if sha256(density_path) != exported["operator_sha256"]["density"]:
            raise RuntimeError("Frozen density export changed")
        predicted = BlockMatrix.load(density_path)
        reference = Snapshot.from_openmx(matrix_path, info_path)
        metadata = input_metadata(info_path)
        density_reference = reference.density * 2
        densities = {
            "reference_full": density_reference,
            "reference_HS_predicted_D_reference_support": restrict_support(
                predicted, density_reference
            ),
        }
        assemblers = {
            "hamiltonian": BlockFourier(reference.hamiltonian),
            "overlap": BlockFourier(reference.overlap),
        }
        assemblers.update({key: BlockFourier(d) for key, d in densities.items()})
        kpoints = uniform_kpoints(metadata["native_kmesh"])
        records = {key: [] for key in densities}
        for k in kpoints:
            h, s = assemblers["hamiltonian"].at(k), assemblers["overlap"].at(k)
            for key in densities:
                result = diagnose_kpoint(h, s, assemblers[key].at(k))
                result["fractional_k"] = k
                records[key].append(result)
        row = {
            "matrix_path": str(matrix_path),
            "info_path": str(info_path),
            "input_metadata": metadata,
            "scenarios": {},
        }
        for key, density in densities.items():
            row["scenarios"][key] = {
                "operators": {
                    "hamiltonian": "reference",
                    "overlap": "reference",
                    "density": (
                        "reference"
                        if key == "reference_full"
                        else "prediction_on_reference_support"
                    ),
                },
                "summary": summarize_records(records[key]),
                "hermitian_realspace_electron_trace": sparse_trace(
                    (density + density.adjoint()) * 0.5, reference.overlap
                ),
                "fermi_consistency": fermi_consistency(
                    records[key],
                    num_electrons=metadata["num_electrons"],
                    temperature_kelvin=metadata["temperature_kelvin"],
                ),
                "kpoints": [public_record(r) for r in records[key]],
            }
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
            "structures": len(pairs),
            "results": results,
        },
    )
    write_json(args.output_dir / "aggregate.json", summarize(args.output_dir))


if __name__ == "__main__":
    main()
