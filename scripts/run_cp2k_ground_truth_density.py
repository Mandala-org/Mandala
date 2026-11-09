"""Registered Gamma finite-block OpenMX density controls on allocated nodes.

Uses native k-point AO-density restarts at a one-point Gamma mesh. No WFN
occupation reset, fixed occupation LIST, H-derived completion, or PSD repair.
The archived finite-block convention is explicit: unrecorded pairs are zero.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.cp2k_gamma_density import (
    finite_gamma_density,
    map_density,
    gamma_template_layout,
    pack_gamma,
)
from data.cp2k_restart import read_cp2k_density_restart, write_cp2k_density_restart
from scripts.run_cp2k_guess_comparison import inspect_log


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def target_metric(old):
    # The previous conventional-Gamma atomic output is an independent native
    # metric oracle at exactly this geometry and basis. No density/H values are
    # read from it. Its Gaussian overlap was already checked against PySCF.
    from analysis.cp2k_gamma_compare import printed_gamma_overlap

    analytic = np.load(old / "target_overlap.npy")
    native = printed_gamma_overlap(old / "cases/atomic/cp2k.out", len(analytic))
    if np.max(abs(native - analytic)) > 2e-6:
        raise ValueError("Independent/native Gaussian metric disagreement")
    return native


def input_text(text, *, steps=200, restart=None, write=False):
    text = re.sub(
        r"    &PRINT\n      &AO_MATRICES ON.*?    &END PRINT\n", "", text, flags=re.S
    )
    text = re.sub(r"MAX_SCF\s+\d+", f"MAX_SCF {steps}", text)
    if "&KPOINTS" not in text:
        text = text.replace(
            "    UKS F",
            "    UKS F\n    &KPOINTS\n      SCHEME MONKHORST-PACK 1 1 1\n      FULL_GRID T\n      SYMMETRY F\n      WAVEFUNCTIONS REAL\n    &END KPOINTS",
        )
    text = (
        text.replace("&RESTART OFF", "&RESTART ON\n          BACKUP_COPIES 0")
        if write
        else text.replace("&RESTART ON", "&RESTART OFF")
    )
    if restart is not None:
        text = text.replace("SCF_GUESS ATOMIC", "SCF_GUESS RESTART")
        text = text.replace(
            "    UKS F", "    UKS F\n    WFN_RESTART_FILE_NAME " + str(restart)
        )
    return text


def run(directory, text, *, electrons, timeout, expected):
    directory.mkdir(parents=True, exist_ok=False)
    inp = directory / "input.inp"
    inp.write_text(text)
    binary = shutil.which("cp2k.psmp")
    if binary is None:
        raise RuntimeError("CP2K executable missing")
    command = [binary, "-i", str(inp), "-o", str(directory / "cp2k.out")]
    version = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=True, timeout=30
    )
    save(
        directory / "execution_contract.json",
        dict(
            command=command,
            input_sha256=sha(inp),
            timeout_seconds=timeout,
            executable_sha256=sha(binary),
            version=version.stdout + version.stderr,
        ),
    )
    start = time.monotonic()
    timed_out = False
    with (
        (directory / "stdout").open("x") as out,
        (directory / "stderr").open("x") as err,
    ):
        try:
            code = subprocess.run(
                command, cwd=directory, stdout=out, stderr=err, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            code = None
            timed_out = True
    log = (
        (directory / "cp2k.out").read_text()
        if (directory / "cp2k.out").exists()
        else ""
    )
    result = inspect_log(log)
    iterations = re.findall(r"^\s+(\d+)\s+\S+/Diag\.", log, re.M)
    result.update(
        returncode=code,
        timed_out=timed_out,
        wall_seconds=time.monotonic() - start,
        iterations_executed=int(iterations[-1]) if iterations else None,
        runtime_interface_passed=bool(
            code == 0
            and result["program_ended"]
            and result["guess"] == expected
            and result["electrons"] == electrons
            and not result["restart_fallback"]
        ),
    )
    save(directory / "summary.json", result)
    print(json.dumps(result), flush=True)
    if not result["runtime_interface_passed"]:
        raise RuntimeError("Runtime/import/count failure; preserve outputs")
    return result


def certify(template, overlap, electrons):
    p, masks, report = gamma_template_layout(template)
    count = float(np.sum(p * overlap))
    report["native_electrons"] = count
    if abs(count - electrons) > 1e-5:
        raise ValueError(f"Native Gamma convention/charge mismatch: {count}")
    control = pack_gamma(template, p, masks)
    error = max(
        float(np.max(abs(a.matrix - b.matrix)))
        for a, b in zip(template.spins[0], control.spins[0])
    )
    report["native_repacking_error"] = error
    if error > 1e-9:
        raise ValueError("Native Gamma repacking failed")
    return p, masks, report, control


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=["atomic", "prepare", "guess", "probe", "collect"],
        required=True,
    )
    ap.add_argument("--material", choices=["silicon", "sio2"], default="silicon")
    ap.add_argument("--family", choices=["molopt", "fitted"], default="molopt")
    ap.add_argument(
        "--mapping", choices=["projection", "coefficient"], default="projection"
    )
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    root = Path(spec["root"])
    prior = Path(spec["prior"])
    directory = root / args.material / args.family
    cfg = spec["materials"][args.material]
    ne = cfg["electrons"]
    old = prior / "prepared" / args.material / args.family
    original = (old / "cases/atomic/input.inp").read_text()
    if args.mode == "atomic":
        if not (root / "probe/PASSED.json").exists():
            raise ValueError("Native restart probe not certified")
        directory.mkdir(parents=True, exist_ok=False)
        save(
            directory / "contract.json",
            dict(
                spec=spec,
                material=args.material,
                family=args.family,
                original_atomic_input_sha256=sha(old / "cases/atomic/input.inp"),
                guess_definition="Actual stored Gamma D; finite-support unrecorded pairs zero; scalar target charge normalization only",
                no_H_completion=True,
                no_occupation_rounding=True,
                no_PSD_repair=True,
                template_scope="Whole atom storage masks/metadata only; no template values in new guesses",
            ),
        )
        baseline = run(
            directory / "atomic",
            input_text(original, write=True),
            electrons=ne,
            timeout=spec["timeout"][args.material][args.family],
            expected="ATOMIC",
        )
        if not (directory / "atomic/benchmark-RESTART.kp").exists():
            raise ValueError("Native density restart missing")
        save(directory / "ATOMIC_FINISHED.json", baseline)
        return
    if args.mode == "probe":
        probe = Path(spec["native_probe"]) / "atomic"
        template = read_cp2k_density_restart(probe / "benchmark-RESTART.kp")
        st = target_metric(prior / "prepared/silicon/molopt")
        _, masks, report, control = certify(template, st, 32)
        out = root / "probe"
        out.mkdir(parents=True, exist_ok=False)
        path = out / "repacked.kp"
        write_cp2k_density_restart(path, control)
        reread = read_cp2k_density_restart(path)
        if (
            max(
                np.max(abs(a.matrix - b.matrix))
                for a, b in zip(control.spins[0], reread.spins[0])
            )
            > 0
        ):
            raise ValueError("Binary readback differs")
        run(
            out / "import",
            input_text(original, steps=2, restart=path),
            electrons=32,
            timeout=600,
            expected="RESTART",
        )
        save(out / "PASSED.json", report)
        print(json.dumps(report), flush=True)
        return
    if args.mode == "prepare":
        if not (root / "probe/PASSED.json").exists():
            raise ValueError("Native interface probe not certified")
        if not (directory / "ATOMIC_FINISHED.json").exists():
            raise ValueError("Atomic runtime failed")
        sys.path.insert(0, str(prior / "supervisor/tools/openmx"))
        from read_openmx_matrix import read_matrix
        from cp2k_wfn import BasisLayout

        source = Path(cfg["source"])
        inp = source / cfg["input"]
        info = inp.read_text()
        mesh = list(
            map(
                int,
                re.search(r"scf.Kgrid\s+(\d+)\s+(\d+)\s+(\d+)", info, re.I).groups(),
            )
        )
        if mesh != [1, 1, 1]:
            raise ValueError("Native-k density must not be collapsed into Gamma")
        if args.material == "silicon":
            log = (source / "openmx.stdout").read_text()
            de = re.findall(r"dUele\s*=\s*([-+0-9.Ee]+)", log)
            nr = re.findall(r"NormRD\s*=\s*([-+0-9.Ee]+)", log)
            if (
                not de
                or not nr
                or abs(float(de[-1])) >= 1e-8
                or float(nr[-1]) ** 2 >= 1e-7
            ):
                raise ValueError(
                    "Fresh OpenMX source did not satisfy its registered SCF criteria"
                )
        else:
            contract = json.loads(
                (prior / "prepared" / args.material / "contract.json").read_text()
            )
            for path in (source / "HS.out", inp):
                if sha(path) != contract["input_sha256"][str(path)]:
                    raise ValueError("Archived ground-truth source identity changed")
        dump = read_matrix(source / "HS.out")
        p, report = finite_gamma_density(dump)
        if dump.block_sizes != [13] * cfg["atoms"]:
            raise ValueError("Unexpected source AO sizes")
        offsets = np.cumsum([0, *dump.block_sizes])
        ss = np.zeros_like(p)
        for (i, j, _), value in dump.section("Overlap matrix").blocks.items():
            ss[offsets[i - 1] : offsets[i], offsets[j - 1] : offsets[j]] += value
        cached = np.load(prior / "prepared" / args.material / "source_overlap.npy")
        error = float(np.max(abs(ss - cached)))
        if error > 2e-6:
            raise ValueError("Source PAO/geometry differs from cached projection basis")
        count = float(np.sum(p * ss))
        if abs(count - ne) > 1e-5:
            raise ValueError(f"Actual source charge differs: {count}")
        report.update(
            source_sha256=sha(source / "HS.out"),
            source_input_sha256=sha(inp),
            source_overlap_max_error=error,
            source_electrons=count,
        )
        st = target_metric(old)
        b = np.load(old / "cross_overlap.npy")
        basis = (
            prior
            / "prepared"
            / args.material
            / "bases"
            / ("BASIS_FITTED" if args.family == "fitted" else "BASIS_MOLOPT_LABELLED")
        )
        expected = json.loads((old / "overlap_checks.json").read_text())[
            "target_basis_sha256"
        ]
        if sha(basis) != expected:
            raise ValueError("Frozen target basis changed")
        report.update(
            target_basis_sha256=expected,
            cross_overlap_sha256=sha(old / "cross_overlap.npy"),
        )
        template = read_cp2k_density_restart(directory / "atomic/benchmark-RESTART.kp")
        _, masks, native, control = certify(template, st, ne)
        control_file = directory / "native_control.kp"
        write_cp2k_density_restart(control_file, control)
        del control
        # Only native storage metadata are needed after exact repacking was
        # certified. Release dense image payloads before launching CP2K.
        for image in template.spins[0]:
            image.matrix = np.empty((0, 0))
        control_result = run(
            directory / "native_control",
            input_text(original, steps=2, restart=control_file),
            electrons=ne,
            timeout=spec["control_timeout"][args.material][args.family],
            expected="RESTART",
        )
        baseline = json.loads((directory / "atomic/summary.json").read_text())
        native["control_endpoint_difference_from_atomic_Ha"] = (
            control_result["energy_hartree"] - baseline["energy_hartree"]
        )
        native["endpoint_energy_check_required"] = baseline["converged"]
        # Extra iterations can legitimately change an unconverged endpoint.
        # Its interface gates remain exact payload repacking, charge, native
        # Gamma structure, and successful RESTART import without fallback.
        if baseline["converged"] and abs(
            native["control_endpoint_difference_from_atomic_Ha"]
        ) > max(1e-5, 1e-6 * cfg["atoms"]):
            raise ValueError("Native roundtrip endpoint energy differs")
        mappings = (
            ["projection"] if args.family == "molopt" else ["projection", "coefficient"]
        )
        paths = {}
        for mapping in mappings:
            with threadpool_limits(limits=1):
                if mapping == "projection":
                    mapped, diagnostic = map_density(p, st, cross=b, electrons=ne)
                else:
                    basis = prior / "prepared" / args.material / "bases/BASIS_FITTED"
                    block = re.search(
                        r"<Atoms.SpeciesAndCoordinates[^\n]*\n(.*?)Atoms.SpeciesAndCoordinates>",
                        info,
                        re.S,
                    )[1]
                    atoms = [
                        row.split()[1]
                        for row in block.splitlines()
                        if row.strip() and not row.lstrip().startswith("#")
                    ]
                    names = [
                        "OMX-FIT-" + cfg["paos"][atom]["selection"] for atom in atoms
                    ]
                    layout = BasisLayout.from_basis_file(basis, names)
                    mapped, diagnostic = map_density(
                        p, st, permutation=layout.permutation(), electrons=ne
                    )
            packed = pack_gamma(template, mapped, masks)
            destination = directory / (mapping + ".kp")
            write_cp2k_density_restart(destination, packed)
            union = np.logical_or.reduce(masks)
            union = union | union.T
            represented = (
                (
                    mapped.reshape(cfg["atoms"], 13, cfg["atoms"], 13).transpose(
                        0, 2, 1, 3
                    )
                    * union[:, :, None, None]
                )
                .transpose(0, 2, 1, 3)
                .reshape(mapped.shape)
            )
            actual = float(np.sum(represented * st))
            if abs(actual - ne) > 1e-5:
                raise ValueError("Native storage masks changed mapped electron count")
            paths[mapping] = dict(
                path=str(destination),
                sha256=sha(destination),
                diagnostics=diagnostic,
                stored_electrons=actual,
            )
        save(
            directory / "SOURCE_AND_MAPS.json",
            dict(
                source=report,
                native=native,
                mappings=paths,
                producer_script_sha256=sha(Path(__file__)),
                producer_source_manifest_sha256=sha(
                    ROOT.parent / "source_manifest.json"
                ),
            ),
        )
        save(directory / "READY.json", dict(prepared=True, mappings=mappings))
        return
    if args.mode == "guess":
        if not (directory / "READY.json").exists():
            raise ValueError("Density mapping/interface gate not passed")
        maps = json.loads((directory / "SOURCE_AND_MAPS.json").read_text())["mappings"]
        item = maps[args.mapping]
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError("Frozen density payload changed")
        result = run(
            directory / ("ground_truth_" + args.mapping),
            input_text(original, restart=path),
            electrons=ne,
            timeout=spec["timeout"][args.material][args.family],
            expected="RESTART",
        )
        baseline = json.loads((directory / "atomic/summary.json").read_text())
        result["baseline_converged"] = baseline["converged"]
        result["energy_difference_from_atomic_Ha"] = (
            result["energy_hartree"] - baseline["energy_hartree"]
        )
        result["scf_steps_saved"] = (
            baseline["scf_steps"] - result["scf_steps"]
            if baseline["converged"] and result["converged"]
            else None
        )
        result["energy_agrees"] = abs(
            result["energy_difference_from_atomic_Ha"]
        ) <= max(1e-5, 1e-6 * cfg["atoms"])
        save(directory / ("ground_truth_" + args.mapping) / "summary.json", result)
        if not result["converged"] or not result["energy_agrees"]:
            raise RuntimeError("Scientific checks failed; no speedup claim")
        return
    results = {}
    for material in spec["materials"]:
        for family in ("molopt", "fitted"):
            d = root / material / family
            cases = {}
            for name in ["atomic", "ground_truth_projection"] + (
                ["ground_truth_coefficient"] if family == "fitted" else []
            ):
                path = d / name / "summary.json"
                cases[name] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else dict(missing=True)
                )
            results[material + "_" + family] = cases
    save(root / "summary.json", dict(cases=results, definition=spec["definition"]))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
