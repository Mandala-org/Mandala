#!/usr/bin/env python
"""Join a resource-only partition after verifying every scientific identity."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.report.evaluate_physical_constraints import sha256, write_json
from scripts.report.summarize_physical_constraints import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    evaluations = {}
    for mode in ("develop", "historical"):
        source = args.records_root / "results" / f"zncusnses_{mode}"
        original_contract = source / "contract.json"
        if sha256(original_contract) != registration["original_contract_sha256"][mode]:
            raise RuntimeError("Original producer contract changed")
        contract = json.loads(original_contract.read_text())
        output = args.output_root / f"zncusnses_{mode}"
        output.mkdir()
        rows, provenance = [], []
        for i, expected in enumerate(registration["validation"]):
            completed = registration["completed"][mode].get(str(i))
            if completed:
                row_path = source / f"structure_{i:04d}.json"
                if sha256(row_path) != completed["sha256"]:
                    raise RuntimeError("Preserved completed row changed")
                producer = original_contract
                identity = contract
            else:
                producer = (
                    args.records_root
                    / "zn_chunks/results"
                    / f"{mode}_{i:04d}"
                    / "contract.json"
                )
                row_path = producer.parent / "structure_0000.json"
                identity = json.loads(producer.read_text())
                for key in (
                    "checkpoint_sha256",
                    "source_manifest_sha256",
                    "envelope_sha256",
                    "historical_source_manifest_sha256",
                    "tolerances",
                    "postprocessing",
                ):
                    if identity.get(key) != contract.get(key):
                        raise RuntimeError(
                            f"Changed scientific identity {mode} {i}: {key}"
                        )
                if identity["arguments"]["mesh"] != contract["arguments"]["mesh"]:
                    raise RuntimeError("Changed analysis mesh")
                if identity["input_sha256"] != [contract["input_sha256"][i]]:
                    raise RuntimeError("Changed structure/geometry identities")
            row = json.loads(row_path.read_text())
            if [row["matrix_path"], row["info_path"]] != expected:
                raise RuntimeError("Structure is outside its registered ordinal")
            row = deepcopy(row)
            # V1 used Python identity to label a newly restricted matrix. Fix
            # that provenance label in this derived artifact; no numbers change.
            row["scenarios"]["predicted_H_reference_SD_reference_support"]["operators"][
                "hamiltonian"
            ] = "prediction_on_reference_support"
            write_json(output / f"structure_{i:04d}.json", row)
            rows.append(
                {
                    **row,
                    "scenarios": {
                        key: {k: v for k, v in scenario.items() if k != "kpoints"}
                        for key, scenario in row["scenarios"].items()
                    },
                }
            )
            provenance.append(
                {
                    "ordinal": i,
                    "row": str(row_path.relative_to(args.records_root)),
                    "row_sha256": sha256(row_path),
                    "producer_contract_sha256": sha256(producer),
                }
            )
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "material": "zncusnses",
                "structures": len(rows),
                "results": rows,
            },
        )
        contract["resource_partition_join"] = {
            "registration_sha256": sha256(args.registration),
            "aggregation_script_sha256": sha256(Path(__file__)),
            "components": provenance,
            "metadata_correction": "V1 support-control H origin label corrected; all numeric values preserved",
        }
        write_json(output / "contract.json", contract)
        evaluations[str(output)] = summarize(output)
    write_json(
        args.output_root / "zncusnses_aggregate.json",
        {"schema_version": 1, "evaluations": evaluations},
    )


if __name__ == "__main__":
    main()
