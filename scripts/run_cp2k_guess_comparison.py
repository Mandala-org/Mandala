"""Run the five frozen Si initial-guess cases on an allocated CP2K node."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def inspect_log(log):
    steps = re.findall(r"SCF run converged in\s+(\d+)\s+steps", log)
    energies = re.findall(r"ENERGY\| Total FORCE_EVAL.*?([-+\d.]+)\s*$", log, re.M)
    guesses = re.findall(r"Density guess:\s+(\S+)", log)
    electrons = re.findall(r"Number of electrons:\s+(\d+)", log)
    return dict(
        converged=bool(steps),
        scf_steps=int(steps[-1]) if steps else None,
        energy_hartree=float(energies[-1]) if energies else None,
        guess=guesses[-1] if guesses else None,
        electrons=int(electrons[-1]) if electrons else None,
        program_ended="PROGRAM ENDED AT" in log,
        restart_fallback=bool(
            re.search(r"(restart.*not found|restart.*does not exist)", log, re.I)
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    root = args.prepared.resolve()
    if not (root / "READY.json").is_file():
        raise RuntimeError("Projection certification is incomplete")
    cases = json.loads((root / "comparison_cases.json").read_text())["cases"]
    binary = shutil.which("cp2k.psmp")
    if not binary:
        raise RuntimeError("CP2K is not on PATH")
    contract = dict(
        binary=binary,
        native_serialization_energy_tolerance_hartree=1e-10,
        final_energy_tolerance_hartree=1e-5,
        timeout_seconds=args.timeout,
        cases=cases,
        note="One geometry; no generalization or net speedup claim",
    )
    with (root / "comparison_contract.json").open("x") as f:
        json.dump(contract, f, indent=2)
    results = {}
    for case in cases:
        directory = Path(case["directory"])
        inp = directory / "input.inp"
        if hashlib.sha256(inp.read_bytes()).hexdigest() != case["input_sha256"]:
            raise RuntimeError("Frozen input changed")
        if (directory / "cp2k.out").exists():
            raise FileExistsError("Refusing to reuse a previous calculation directory")
        start = time.monotonic()
        with (
            (directory / "stdout").open("x") as out,
            (directory / "stderr").open("x") as err,
        ):
            run = subprocess.run(
                [binary, "-i", str(inp), "-o", str(directory / "cp2k.out")],
                cwd=directory,
                stdout=out,
                stderr=err,
                timeout=args.timeout,
            )
        result = inspect_log((directory / "cp2k.out").read_text())
        result.update(returncode=run.returncode, wall_seconds=time.monotonic() - start)
        expected = "ATOMIC" if case["name"] == "atomic" else "RESTART"
        result["execution_passed"] = bool(
            run.returncode == 0
            and result["converged"]
            and result["program_ended"]
            and result["electrons"] == 32
            and result["guess"] == expected
            and not result["restart_fallback"]
        )
        results[case["name"]] = result
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (root / "comparison_partial.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(case["name"], json.dumps(result), flush=True)
        if not result["execution_passed"]:
            raise RuntimeError(f"CP2K case failed: {case['name']}")
    a, b = results["native_original"], results["native_serialized"]
    control = (
        abs(a["energy_hartree"] - b["energy_hartree"]) <= 1e-10
        and a["scf_steps"] == b["scf_steps"]
    )
    baseline = results["atomic"]
    for result in results.values():
        result["energy_difference_from_atomic_hartree"] = (
            result["energy_hartree"] - baseline["energy_hartree"]
        )
        result["final_energy_agrees"] = (
            abs(result["energy_difference_from_atomic_hartree"]) <= 1e-5
        )
        result["scf_steps_saved"] = baseline["scf_steps"] - result["scf_steps"]
    summary = dict(
        cases=results,
        serialization_control_passed=control,
        passed=control and all(x["final_energy_agrees"] for x in results.values()),
    )
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise RuntimeError("Comparison scientific acceptance checks failed")


if __name__ == "__main__":
    main()
