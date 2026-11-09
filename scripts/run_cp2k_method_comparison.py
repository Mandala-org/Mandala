"""Execute one matched Gamma basis family on an allocated ROSI CPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parents[1]),
]
import numpy as np
from analysis.cp2k_gamma_compare import printed_gamma_overlap
from scripts.run_cp2k_guess_comparison import inspect_log


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--family", choices=["molopt", "fitted"], required=True)
    ap.add_argument("--timeout", type=int, default=2700)
    ap.add_argument(
        "--case", help="Execute one frozen case in its own execution record"
    )
    ap.add_argument("--mpi-ranks", type=int, default=1)
    args = ap.parse_args()
    directory = args.prepared.resolve() / args.family
    if not (directory / "READY.json").is_file():
        raise RuntimeError("Preparation/certification did not complete")
    specification = json.loads((directory / "cases.json").read_text())
    if args.mpi_ranks < 1 or args.timeout <= 0:
        ap.error("MPI rank count and timeout must be positive")
    selected = specification["cases"]
    if args.case:
        selected = [case for case in selected if case["name"] == args.case]
        if len(selected) != 1:
            ap.error("Case name must identify exactly one frozen case")
    execution = directory if args.case is None else directory / "executions" / args.case
    execution.mkdir(parents=True, exist_ok=True)
    if (execution / "execution_contract.json").exists():
        raise FileExistsError("Refusing to repeat or overwrite a frozen execution")
    if args.case and args.case != "atomic":
        calibration = json.loads(
            (directory / "native_overlap_calibration.json").read_text()
        )
        if not calibration["passed"]:
            raise RuntimeError("Atomic native-overlap calibration did not pass")
    binary = shutil.which("cp2k.psmp")
    if not binary:
        raise RuntimeError("CP2K executable missing")
    with (execution / "version.txt").open("x") as log:
        subprocess.run(
            [binary, "--version"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=30,
        )
    contract = dict(
        executable=binary,
        executable_sha256=sha(Path(binary)),
        cases=specification,
        timeout_seconds=args.timeout,
        version=(execution / "version.txt").read_text(),
        selected_case=args.case,
        mpi_ranks=args.mpi_ranks,
        omp_threads=os.environ.get("OMP_NUM_THREADS", "1"),
    )
    save(execution / "execution_contract.json", contract)
    results = {}
    for case in selected:
        run_dir = Path(case["directory"])
        inp = run_dir / "input.inp"
        if sha(inp) != case["input_sha256"]:
            raise ValueError("Frozen SCF input changed")
        if (
            case["guess"] is not None
            and sha(Path(case["guess"])) != case["guess_sha256"]
        ):
            raise ValueError("Frozen guess changed")
        if (run_dir / "cp2k.out").exists():
            raise FileExistsError("Calculation directory is not fresh")
        start = time.monotonic()
        timed_out = False
        command = [binary, "-i", str(inp), "-o", str(run_dir / "cp2k.out")]
        if args.mpi_ranks > 1:
            threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
            command = [
                "mpirun",
                "--np",
                str(args.mpi_ranks),
                "--bind-to",
                "core",
                "--map-by",
                f"slot:PE={threads}",
                *command,
            ]
        with (
            (run_dir / "stdout").open("x") as out,
            (run_dir / "stderr").open("x") as err,
        ):
            try:
                process = subprocess.run(
                    command, cwd=run_dir, stdout=out, stderr=err, timeout=args.timeout
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = None
                timed_out = True
        text = (
            (run_dir / "cp2k.out").read_text()
            if (run_dir / "cp2k.out").exists()
            else ""
        )
        result = inspect_log(text)
        result.update(
            returncode=code, timed_out=timed_out, wall_seconds=time.monotonic() - start
        )
        expected = "ATOMIC" if case["guess"] is None else "RESTART"
        result["execution_passed"] = bool(
            code == 0
            and result["converged"]
            and result["program_ended"]
            and result["electrons"] == specification["electrons"]
            and result["guess"] == expected
            and not result["restart_fallback"]
        )
        if case["name"] == "atomic" and code == 0:
            native = printed_gamma_overlap(run_dir / "cp2k.out", specification["nao"])
            analytic = np.load(directory / "target_overlap.npy")
            maximum = float(np.max(abs(native - analytic)))
            calibration = dict(
                max_native_analytic_overlap_error=maximum,
                relative_frobenius_error=float(
                    np.linalg.norm(native - analytic) / np.linalg.norm(native)
                ),
                passed=maximum <= specification["native_overlap_max_error_tolerance"],
            )
            save(directory / "native_overlap_calibration.json", calibration)
            result["basis_metric_calibration_passed"] = calibration["passed"]
            # Interface calibration is independent of learned-guess SCF results.
            if not calibration["passed"]:
                results[case["name"]] = result
                save(execution / "partial.json", results)
                raise RuntimeError(f"Native overlap convention failed: {maximum}")
        results[case["name"]] = result
        save(run_dir / "summary.json", result)
        save(execution / "partial.json", results)
        print(case["name"], json.dumps(result), flush=True)
        if code != 0:
            raise RuntimeError(
                "CP2K execution/runtime failed; preserve outputs before a new run"
            )
        # A scientific nonconvergence is an outcome; do not suppress independent
        # pre-registered guesses merely because an earlier guess needed more steps.
    if "atomic" in results:
        baseline = results["atomic"]
    else:
        atomic = next(
            case for case in specification["cases"] if case["name"] == "atomic"
        )
        baseline = json.loads((Path(atomic["directory"]) / "summary.json").read_text())
    for name, result in results.items():
        result["scf_steps_saved"] = (
            None
            if result["scf_steps"] is None or baseline["scf_steps"] is None
            else baseline["scf_steps"] - result["scf_steps"]
        )
        result["energy_difference_from_atomic_hartree"] = (
            None
            if result["energy_hartree"] is None or baseline["energy_hartree"] is None
            else result["energy_hartree"] - baseline["energy_hartree"]
        )
        result["final_energy_agrees"] = (
            result["energy_difference_from_atomic_hartree"] is not None
            and abs(result["energy_difference_from_atomic_hartree"])
            <= specification["final_energy_tolerance_hartree"]
        )
    summary = dict(
        material=specification["material"],
        family=args.family,
        cases=results,
        passed=all(
            r["execution_passed"] and r["final_energy_agrees"] for r in results.values()
        ),
        energy_tolerance_hartree=specification["final_energy_tolerance_hartree"],
        timing_scope="CP2K subprocess only; separately include preparation/model-export cost",
        cross_family_absolute_energy_comparison_valid=False,
    )
    save(execution / "summary.json", summary)
    if not summary["passed"]:
        # A valid atomic runtime/overlap gate can release independent guesses
        # even if its scientific convergence criterion was not met.
        if (
            args.case == "atomic"
            and baseline["returncode"] == 0
            and baseline.get("basis_metric_calibration_passed", False)
        ):
            print(
                "Atomic scientific checks failed; native interface passed; independent guesses remain eligible",
                flush=True,
            )
        else:
            raise RuntimeError("One or more scientific acceptance checks failed")


if __name__ == "__main__":
    main()
