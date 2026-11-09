"""Prepare paper Si/SiO2 Gamma projection versus coefficient-transfer jobs.

Scientific computation belongs on allocated ROSI nodes. No training occurs.
Raw publication exports are verified against their producer/checkpoint/input
manifests before the common source occupied space is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.linalg import eigh, qr
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from analysis.cp2k_gamma_compare import (
    BOHR_PER_ANGSTROM,
    compact_pao_quadrature,
    gamma_from_blocks,
    gamma_from_dump,
    gaussian_gamma_overlap,
    input_geometry,
    occupied_from_density,
    occupied_from_hamiltonian,
    periodic_cross_overlap,
    project_occupied,
)
from analysis.cp2k_projection import (
    gaussian_pao_cross_overlap_numpy,
    pao_quadrature,
    read_cp2k_shells,
)
from analysis.openmx_density_grid import parse_openmx_pao
from core.basis_converter import OpenMXE3NNConverter
from data.block_matrix import BlockMatrix


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def label_cp2k_basis(source, destination, selections):
    """Give element-specific aliases without changing any Gaussian numbers.

    The supplied WFN layout reader indexes names without their element, whereas
    CP2K's standard MOLOPT aliases are shared by many elements.
    """
    aliases = {element: "CMP-MOLOPT-" + element for element in selections}
    tokens = []
    for line in source.read_text().splitlines():
        fields = line.split("#", 1)[0].split("!", 1)[0].split()
        if fields:
            tokens.append(fields)
    lines = []
    for element, selection in selections.items():
        matches = [
            i
            for i, row in enumerate(tokens)
            if row[0] == element and selection in row[1:]
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous MOLOPT basis: {element} {selection}")
        start = matches[0] + 1
        end = start + 1
        for _ in range(int(tokens[start][0])):
            primitive_count = int(tokens[end][3])
            end += 1 + primitive_count
        lines.append(" " + element + " " + aliases[element])
        lines.extend(" ".join(row) for row in tokens[start:end])
    # Only selected entries: the installed full library also contains unrelated
    # formats unsupported by the supplied WFN metadata reader.
    destination.write_text("\n".join(lines) + "\n")
    return aliases


def cp2k_input(
    atoms, positions, cell, basis, names, potentials, *, guess=None, overlap=False
):
    project = "benchmark"
    kinds = "\n".join(
        f"""    &KIND {element}
      BASIS_SET {names[element]}
      POTENTIAL GTH-PBE-q{4 if element=='Si' else 6}
    &END KIND"""
        for element in sorted(set(atoms))
    )
    coordinates = "\n".join(
        "      " + a + " " + " ".join(f"{v:.14f}" for v in p)
        for a, p in zip(atoms, positions)
    )
    vectors = "\n".join(
        "      " + label + " " + " ".join(f"{v:.14f}" for v in row)
        for label, row in zip("ABC", cell)
    )
    restart = "" if guess is None else "    WFN_RESTART_FILE_NAME " + str(guess) + "\n"
    printing = (
        """    &PRINT
      &AO_MATRICES ON
        OVERLAP T
        NDIGITS 16
      &END AO_MATRICES
    &END PRINT
"""
        if overlap
        else ""
    )
    return f"""# Frozen Gamma initial-guess comparison; integer occupations, no smearing.
&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME {basis}
    POTENTIAL_FILE_NAME {potentials}
    CHARGE 0
    UKS F
{restart}    &QS
      EPS_DEFAULT 1.0E-12
      EPS_PGF_ORB 1.0E-12
      EPS_PPNL 1.0E-12
      CORE_PPL GRID
    &END QS
    &MGRID
      CUTOFF [Ry] 600
      REL_CUTOFF [Ry] 80
    &END MGRID
    &POISSON
      PERIODIC XYZ
    &END POISSON
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
{printing}    &SCF
      SCF_GUESS {'ATOMIC' if guess is None else 'RESTART'}
      EPS_SCF 1.0E-7
      EPS_EIGVAL 1.0E-5
      MAX_SCF 200
      ADDED_MOS -1
      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION
      &MIXING
        METHOD BROYDEN_MIXING
        ALPHA 0.15
        NBUFFER 16
      &END MIXING
      &PRINT
        &RESTART OFF
        &END RESTART
      &END PRINT
    &END SCF
  &END DFT
  &SUBSYS
    &CELL
{vectors}
      PERIODIC XYZ
    &END CELL
    &COORD
{coordinates}
    &END COORD
{kinds}
  &END SUBSYS
&END FORCE_EVAL
"""


def full_target_orbitals(c, overlap, threshold=1e-5):
    values, vectors = eigh((overlap + overlap.T) / 2)
    keep = values > threshold
    whitening = vectors[:, keep] / np.sqrt(values[keep])
    orthogonal = np.sqrt(values[keep])[:, None] * (vectors[:, keep].T @ c)
    complete, _ = qr(orthogonal, mode="full")
    # Completion can rotate/sign-flip occupied columns, preserving their span/P.
    return whitening @ complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--linalg-threads", type=int, default=4)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    case = spec["materials"][args.material]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(spec["supervisor_tools"])))
    from read_openmx_matrix import read_matrix
    from cp2k_wfn import BasisLayout, Orbitals, write_wfn, read_wfn
    from openmx_to_cp2k import main as supervisor_converter

    start = time.monotonic()
    source = Path(case["source"])
    input_path = source / case["input"]
    atoms, positions, cell = input_geometry(input_path)
    electrons = sum(4 if a == "Si" else 6 for a in atoms)
    if len(atoms) != case["atoms"] or electrons != case["electrons"]:
        raise ValueError(
            "Source composition/electron count differs from frozen specification"
        )
    if electrons % 2:
        raise ValueError("This controlled test requires a closed-shell system")
    publication = json.loads(Path(case["publication_contract"]).read_text())
    producer = json.loads(Path(case["prediction_cache"], "export.json").read_text())
    if (
        not producer["strict_load"]
        or publication["checkpoint_sha256"] != case["checkpoint_sha256"]
    ):
        raise ValueError("Paper inference producer provenance mismatch")
    if publication["split"]["validation"][case["prediction_index"]][0] != str(
        source / "HS.out"
    ):
        raise ValueError(
            "Prediction cache index does not identify the requested structure"
        )
    identities = {
        str(p): digest(p)
        for p in [
            args.spec,
            input_path,
            source / "HS.out",
            source / case["info"],
            source / case["cif"],
            Path(case["checkpoint"]),
            Path(case["publication_contract"]),
            Path(case["prediction_cache"], "export.json"),
            Path(__file__),
        ]
    }
    if identities[case["checkpoint"]] != case["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes changed")
    publication_inputs = {}
    for record in publication["input_sha256"]:
        publication_inputs.update(record)
    for name in ("HS.out", case["info"], case["cif"]):
        path = str(source / name)
        if publication_inputs.get(path) != identities[path]:
            raise ValueError(f"Publication input identity mismatch {path}")
    hpath = Path(case["prediction_cache"], "hamiltonian.pt")
    if digest(hpath) != producer["operator_sha256"]["hamiltonian"]:
        raise ValueError("Cached raw paper Hamiltonian changed")
    identities[str(hpath)] = digest(hpath)
    for element, paodata in case["paos"].items():
        path = Path(spec["pao_dir"], paodata["file"])
        if digest(path) != paodata["sha256"]:
            raise ValueError("PAO identity mismatch")
        identities[str(path)] = digest(path)
    write_json(
        output / "contract.json",
        dict(
            spec=spec,
            material=args.material,
            input_sha256=identities,
            torch=torch.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            source_density="Zero-temperature occupied generalized eigenstates of H with reference OpenMX S; no learned S",
            spin="restricted total; source H-derived density has occupation 2 already",
            periodicity="Gamma; native Gamma for SiO2, explicitly changed sampling for controlled Si interface comparison",
            model_semantics="Strict frozen historical paper exports; historical FiLM does not guarantee O(3); no weight migration",
            no_scf_results_used_to_select_checkpoint_or_fit=True,
        ),
    )
    print("Verifying and folding source operators at Gamma", flush=True)
    dump = read_matrix(source / "HS.out")
    if dump.block_sizes != [13] * len(atoms):
        raise ValueError("Expected native s2p2d1 atom blocks")
    ss = gamma_from_dump(dump.section("Overlap matrix"), dump.block_sizes)
    href = gamma_from_dump(
        dump.section("Kohn-Sham Hamiltonian spin=0"), dump.block_sizes
    )
    ss = (ss + ss.T) / 2
    prediction = BlockMatrix.load(hpath)
    if tuple(prediction.atoms) != atoms:
        raise ValueError("Model output atom ordering mismatch")
    if prediction.basis == "e3nn":
        prediction = OpenMXE3NNConverter(prediction.orbital_cfg).matrix_to_openmx(
            prediction
        )
    hmodel = gamma_from_blocks(prediction)
    with threadpool_limits(limits=args.linalg_threads):
        occupied = {}
        source_reports = {}
        for label, h in [("reference_H", href), ("model_H", hmodel)]:
            occupied[label], source_reports[label] = occupied_from_hamiltonian(
                h, ss, electrons
            )
        dpath = Path(case["prediction_cache"], "density.pt")
        if args.material == "silicon" and dpath.exists():
            if digest(dpath) != producer["operator_sha256"]["density"]:
                raise ValueError("Cached paper density changed")
            dm = BlockMatrix.load(dpath)
            if dm.basis == "e3nn":
                dm = OpenMXE3NNConverter(dm.orbital_cfg).matrix_to_openmx(dm)
            occupied["model_D_projector"], source_reports["model_D_projector"] = (
                occupied_from_density(gamma_from_blocks(dm), ss, electrons)
            )
    write_json(output / "source_density.json", source_reports)
    np.save(output / "source_overlap.npy", ss)
    for label, c in occupied.items():
        np.save(output / (label + "_density.npy"), 2 * c @ c.T)
    (output / "bases").mkdir()
    molopt = output / "bases/BASIS_MOLOPT_LABELLED"
    molopt_names = label_cp2k_basis(
        Path(spec["cp2k_basis"]),
        molopt,
        {e: "DZVP-MOLOPT-SR-GTH" for e in case["paos"]},
    )
    fitted = output / "bases/BASIS_FITTED"
    selections = [v["selection"] for _, v in sorted(case["paos"].items())]
    command = [
        sys.executable,
        str(Path(spec["supervisor_tools"], "convert_basis.py")),
        *selections,
        "--pao-dir",
        spec["pao_dir"],
        "--output",
        str(fitted),
        "--nprimitive",
        "24",
        "--fit-method",
        "shared",
        "--allow-inaccurate",
    ]
    with (output / "basis_fit.log").open("x") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    fitted_report = json.loads(fitted.with_name(fitted.name + ".json").read_text())
    quality = {}
    for item in fitted_report["basis_sets"]:
        radial = max(max(v["normalized_radial_l2_errors"]) for v in item["radial_fits"])
        kinetic = max(
            max(v["kinetic_absolute_errors_hartree"]) for v in item["radial_fits"]
        )
        quality[item["element"]] = dict(
            max_radial_l2=radial,
            max_kinetic_error_hartree=kinetic,
            accepted_under_original_radial_kinetic_gates=radial <= 0.01
            and kinetic <= 0.001,
        )
    write_json(
        output / "fit_quality.json",
        dict(
            fixed_recipe="Supervisor shared fit, 24 primitives/channel; explicit diagnostic allowance",
            channels=quality,
        ),
    )
    tables = {
        e: parse_openmx_pao(Path(spec["pao_dir"], v["file"]), dtype=torch.float64)
        for e, v in case["paos"].items()
    }
    positions_b, cell_b = positions * BOHR_PER_ANGSTROM, cell * BOHR_PER_ANGSTROM
    density_count = {
        label: float(np.sum(np.load(output / (label + "_density.npy")) * ss))
        for label in occupied
    }
    write_json(output / "source_counts.json", density_count)
    for family in ("molopt", "fitted"):
        family_dir = output / family
        family_dir.mkdir()
        basis = molopt if family == "molopt" else fitted
        names = (
            molopt_names
            if family == "molopt"
            else {e: "OMX-FIT-" + v["selection"] for e, v in case["paos"].items()}
        )
        shells = {e: read_cp2k_shells(basis, e, name) for e, name in names.items()}
        if any([l for l, _, _ in v] != [0, 0, 1, 1, 2] for v in shells.values()):
            raise ValueError("Target radial/angular shell order differs from s2p2d1")
        print(f"{family}: independent Gaussian Gamma overlap", flush=True)
        with threadpool_limits(limits=args.linalg_threads):
            st = gaussian_gamma_overlap(atoms, positions_b, cell_b, shells)
        np.save(family_dir / "target_overlap.npy", st)
        quadratures = {}
        quad_reports = {}
        probes = np.array(
            [[0.0, 0.0, 0.0], [1.0, -0.8, 2.3], [6.0, -3.0, 1.0], [12.0, 0.0, 0.0]]
        )
        for element, table in tables.items():
            fine = pao_quadrature(
                table.radii.numpy(),
                {l: v.numpy() for l, v in table.values_by_l.items()},
                [2, 2, 1],
                order=6,
            )
            for order in (512, 1024, 2048, 4096):
                q = compact_pao_quadrature(table, order=order)
                error = 0.0
                for target in shells:
                    aa = gaussian_pao_cross_overlap_numpy(probes, shells[target], *q)
                    bb = gaussian_pao_cross_overlap_numpy(probes, shells[target], *fine)
                    error = max(error, float(np.max(abs(aa - bb))))
                if error <= 5e-7:
                    quadratures[element] = q
                    quad_reports[element] = dict(
                        order=order, max_cross_probe_error=error
                    )
                    break
            else:
                raise ValueError("Compact radial quadrature could not be certified")
        write_json(family_dir / "quadrature.json", quad_reports)
        cross, tail = periodic_cross_overlap(
            atoms,
            positions_b,
            cell_b,
            shells,
            quadratures,
            {e: float(v.radii[-1]) for e, v in tables.items()},
            workers=args.workers,
            progress=lambda s: print(s, flush=True),
        )
        np.save(family_dir / "cross_overlap.npy", cross)
        with threadpool_limits(limits=args.linalg_threads):
            sv, su = eigh((st + st.T) / 2)
            keep = sv > 1e-5
            invroot = su[:, keep] / np.sqrt(sv[keep])
            residual = ss - (invroot.T @ cross).T @ (invroot.T @ cross)
            minimum = float(
                eigh(
                    (residual + residual.T) / 2,
                    subset_by_index=[0, 0],
                    eigvals_only=True,
                )[0]
            )
        write_json(
            family_dir / "overlap_checks.json",
            dict(
                min_joint_gram_residual=minimum,
                tolerance=-1e-4,
                periodic_tail=tail,
                target_basis_sha256=digest(basis),
            ),
        )
        if minimum < -1e-4:
            raise ValueError(
                f"{family} cross/source/target Gram inconsistency: {minimum}"
            )
        layout = BasisLayout.from_basis_file(basis, [names[e] for e in atoms])
        cases = [dict(name="atomic", guess=None)]
        projections = {}
        for label, c in occupied.items():
            print(f"{family}: {label} occupied projection", flush=True)
            with threadpool_limits(limits=args.linalg_threads):
                projected, diagnostic = project_occupied(c, st, cross)
                full = full_target_orbitals(projected, st)
            nocc = electrons // 2
            occupations = np.r_[np.full(nocc, 2.0), np.zeros(full.shape[1] - nocc)]
            guess = family_dir / (label + "_projection.wfn")
            write_wfn(
                guess,
                layout,
                [
                    Orbitals(
                        full,
                        occupations,
                        np.zeros(len(occupations)),
                        electrons,
                        nocc,
                        nocc + 1,
                    )
                ],
                "little",
                4,
            )
            restored_layout, restored = read_wfn(guess, "little", 4)
            if restored_layout != layout or not np.array_equal(
                restored[0].coefficients, full
            ):
                raise ValueError("Projected WFN binary readback mismatch")
            cases.append(
                dict(
                    name=label + "_projection",
                    guess=str(guess),
                    guess_sha256=digest(guess),
                )
            )
            projections[label] = diagnostic
            if family == "fitted":
                direct = family_dir / (label + "_coefficient.wfn")
                arguments = [
                    str(output / (label + "_density.npy")),
                    "--input-format",
                    "dense",
                    "--source-overlap",
                    str(output / "source_overlap.npy"),
                    "--target-overlap",
                    str(family_dir / "target_overlap.npy"),
                    "--basis-file",
                    str(basis),
                    "--basis-names",
                    *[case["paos"][e]["selection"] for e in atoms],
                    "--spin-mode",
                    "restricted-total",
                    "--electrons",
                    str(electrons),
                    "--format",
                    "wfn",
                    "--output",
                    str(direct),
                    "--allow-approximate-basis",
                ]
                # The supervisor WFN preserve route factors in the source metric.
                # Target overlap is measured separately, not passed to that
                # factorization (which would demand exact target-metric charge).
                index = arguments.index("--target-overlap")
                del arguments[index : index + 2]
                if supervisor_converter(arguments) != 0:
                    raise ValueError("Supervisor coefficient WFN conversion failed")
                p = np.load(output / (label + "_density.npy"))
                perm = layout.permutation()
                source_target_count = float(np.sum(p[np.ix_(perm, perm)] * st))
                projections[label][
                    "coefficient_target_metric_electrons"
                ] = source_target_count
                cases.append(
                    dict(
                        name=label + "_coefficient",
                        guess=str(direct),
                        guess_sha256=digest(direct),
                    )
                )
        write_json(family_dir / "projection.json", projections)
        for item in cases:
            directory = family_dir / "cases" / item["name"]
            directory.mkdir(parents=True)
            text = cp2k_input(
                atoms,
                positions,
                cell,
                basis,
                names,
                spec["cp2k_potentials"],
                guess=item["guess"],
                overlap=item["name"] == "atomic",
            )
            (directory / "input.inp").write_text(text)
            item.update(
                directory=str(directory), input_sha256=digest(directory / "input.inp")
            )
        write_json(
            family_dir / "cases.json",
            dict(
                material=args.material,
                family=family,
                atoms=len(atoms),
                nao=len(ss),
                electrons=electrons,
                cases=cases,
                cp2k="2023.1",
                final_energy_tolerance_hartree=max(1e-5, 1e-6 * len(atoms)),
                native_overlap_max_error_tolerance=2e-6,
                periodicity="Gamma",
                cross_family_energy_comparison_valid=False,
            ),
        )
        write_json(family_dir / "READY.json", dict(prepared=True, cases=len(cases)))
    write_json(
        output / "READY.json",
        dict(prepared=True, preparation_seconds=time.monotonic() - start),
    )
    print("Both basis families prepared", flush=True)


if __name__ == "__main__":
    main()
