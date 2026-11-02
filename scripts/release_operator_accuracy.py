"""Release only preregistered jobs after deterministic data/runtime certification.

This workflow reads small provenance/results JSON only. No validation score
selects configurations, budgets or seeds. Failed certification leaves jobs held.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from scripts.prepare_operator_accuracy import sha256, verify


def certify(records, certification_records, runtime_probes=()):
    verify(records)
    source = json.loads((records / "source/SOURCE_MANIFEST.json").read_text())
    original = json.loads(
        (certification_records / "source/SOURCE_MANIFEST.json").read_text()
    )
    scientific = [
        name
        for name in original["files"]
        if name.startswith("src/")
        or name
        in {
            "scripts/train.py",
            "scripts/run_training_segment.py",
            "scripts/evaluate_operator_checkpoint.py",
        }
    ]
    if any(source["files"].get(name) != original["files"][name] for name in scientific):
        raise ValueError("Scientific source changed after data certification")
    registration = json.loads((records / "batch/registration.json").read_text())
    audits = {}
    for material, counts in (("graphene", (270, 90)), ("mos2", (300, 100))):
        path = certification_records / "results" / f"{material}_audit.json"
        document = json.loads(path.read_text())
        if (
            not all(
                document.get(key) is True
                for key in (
                    "passed",
                    "input_hashes_match_pilot",
                    "predictor_configuration_matches_pilot",
                )
            )
            or document.get("test_inputs") != 0
        ):
            raise ValueError(f"{material} certification failed or incomplete")
        if (
            tuple(document["splits"][split]["structures"] for split in ("train", "val"))
            != counts
        ):
            raise ValueError("Certified split membership counts changed")
        expected_manifest = records / "batch" / f"{material}_pilot_run_manifest.json"
        if document["pilot_run_manifest_sha256"] != sha256(expected_manifest):
            raise ValueError("Original pilot inputs differ from certification")
        for key in (
            "roundtrip_ev",
            "pipeline_target_change_ev",
            "energy_roundtrip_ev",
            "cartesian_once_angstrom",
            "pipeline_positions_angstrom",
            "raw_target_hermiticity_max_ev",
        ):
            value = document["checks"][key]
            if not math.isfinite(value) or value > (
                5e-6 if key == "pipeline_target_change_ev" else 5e-5
            ):
                raise ValueError("Convention check failed: " + key)
        for name, origin in document["module_origins"].items():
            expected = certification_records / "source/src" / Path(*name.split("."))
            if Path(origin).resolve() != expected.with_suffix(".py").resolve():
                raise ValueError("Certification imported the wrong source")
        destination = records / "results" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(path):
            raise FileExistsError(destination)
        if not destination.exists():
            shutil.copyfile(path, destination)
        audits[material] = sha256(path)
    probes = []
    probe_configurations = set()
    for path in runtime_probes:
        document = json.loads(path.read_text())
        model = document.get("model_probe", {})
        if (
            document.get("passed") is not True
            or document.get("test_inputs") != 0
            or model.get("training_backward_finite") is not True
            or model.get("raw_validation_outputs_finite") is not True
            or model.get("optimizer_steps") != 0
        ):
            raise ValueError("Runtime compatibility/memory probe failed")
        if registration.get("runtime_profile") == "mandala_legacy":
            runtime = document["runtime"]
            if (
                not runtime["torch"].startswith("2.5.1")
                or not runtime["cuda"].startswith("12.1")
                or "V100" not in runtime["gpu"]["name"]
            ):
                raise ValueError("Incorrect V100 runtime")
        probes.append({"path": str(path), "sha256": sha256(path)})
        probe_configurations.add(document.get("config_sha256"))
    if registration.get("runtime_profile") == "mandala_legacy" and len(probes) != 2:
        raise ValueError("Legacy campaign requires both material runtime probes")
    if registration.get("runtime_profile") == "mandala_legacy":
        expected = {
            sha256(records / "batch" / item["config"])
            for item in registration["jobs"]
            if item["treatment"] == "baseline"
        }
        if len(expected) != 2 or probe_configurations != expected:
            raise ValueError("Runtime probes do not cover both registered baselines")
    return {
        "decision": "release_registered_screen",
        "review_kind": "preregistered deterministic convention/support/runtime checks; no score-dependent promotion",
        "registration_sha256": sha256(records / "batch/registration.json"),
        "audit_sha256": audits,
        "certification_records": str(certification_records),
        "scientific_source_files_identical": len(scientific),
        "runtime_probes": probes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--certification-records", type=Path, required=True)
    parser.add_argument("--runtime-probes", type=Path, nargs="*", default=[])
    parser.add_argument("--submissions", type=Path, required=True)
    args = parser.parse_args(argv)
    review = certify(args.records, args.certification_records, args.runtime_probes)
    ledger = json.loads(args.submissions.read_text())
    registration = json.loads((args.records / "batch/registration.json").read_text())
    if {
        item["arm"]
        for item in ledger["submissions"]
        if item["segment"] == 1 and item["status"] == "submitted"
    } != {item["id"] for item in registration["jobs"]}:
        raise ValueError("Submission ledger is incomplete")
    path = args.records / "workflow/certification_review.json"
    with path.open("x") as stream:
        json.dump(review, stream, indent=2)
        stream.write("\n")
    released = []
    for item in ledger["submissions"]:
        if item["segment"] == 1:
            subprocess.run(["scontrol", "release", item["job_id"]], check=True)
            released.append(item["job_id"])
            (args.records / "workflow/released_jobs.json").write_text(
                json.dumps(released, indent=2) + "\n"
            )
    print(json.dumps({"decision": review["decision"], "released": len(released)}))


if __name__ == "__main__":
    main()
