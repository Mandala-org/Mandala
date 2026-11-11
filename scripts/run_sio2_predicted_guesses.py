"""Frozen four-arm SiO2 paper-H initialization experiment, on compute nodes."""

import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.run_cp2k_ground_truth_density import (
    sha,
    save,
    target_metric,
    input_text,
    run,
    certify,
)
from analysis.cp2k_gamma_density import map_density, pack_gamma
from analysis.cp2k_gamma_compare import gamma_from_blocks, occupied_from_hamiltonian
from core.basis_converter import OpenMXE3NNConverter
from data.block_matrix import BlockMatrix
from data.cp2k_restart import read_cp2k_density_restart, write_cp2k_density_restart


def source_density(prior):
    contract = json.loads((prior / "prepared/sio2/contract.json").read_text())
    case = contract["spec"]["materials"]["sio2"]
    publication = json.loads(Path(case["publication_contract"]).read_text())
    producer = json.loads((Path(case["prediction_cache"]) / "export.json").read_text())
    hp = Path(case["prediction_cache"]) / "hamiltonian.pt"
    assert producer["strict_load"]
    assert (
        sha(case["checkpoint"])
        == case["checkpoint_sha256"]
        == publication["checkpoint_sha256"]
    )
    assert sha(hp) == producer["operator_sha256"]["hamiltonian"]
    assert publication["split"]["validation"][case["prediction_index"]][0] == str(
        Path(case["source"]) / "HS.out"
    )
    for path, digest in contract["input_sha256"].items():
        if Path(path).name in ("HS.out", case["input"], case["info"], case["cif"]):
            assert sha(path) == digest
    prediction = BlockMatrix.load(hp)
    info = (Path(case["source"]) / case["input"]).read_text()
    block = re.search(
        r"<Atoms.SpeciesAndCoordinates[^\n]*\n(.*?)Atoms.SpeciesAndCoordinates>",
        info,
        re.S,
    )[1]
    atoms = [
        line.split()[1]
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert tuple(prediction.atoms) == tuple(atoms) and len(atoms) == 150
    if prediction.basis == "e3nn":
        prediction = OpenMXE3NNConverter(prediction.orbital_cfg).matrix_to_openmx(
            prediction
        )
    s = np.load(prior / "prepared/sio2/source_overlap.npy")
    with threadpool_limits(limits=1):
        c, report = occupied_from_hamiltonian(gamma_from_blocks(prediction), s, 800)
        p = 2 * c @ c.T
    assert abs(np.sum(p * s) - 800) < 1e-6
    # Independent reproduction check against the frozen earlier H-derived density.
    error = float(np.max(abs(p - np.load(prior / "prepared/sio2/model_H_density.npy"))))
    assert error < 1e-7, error
    return (
        p,
        atoms,
        case,
        dict(
            report,
            checkpoint_sha256=sha(case["checkpoint"]),
            hamiltonian_sha256=sha(hp),
            previous_density_max_error=error,
            source="Paper SiO2 Hamiltonian-only checkpoint; reference source overlap; zero-T occupation2; not a learned D head",
        ),
    )


def prepare(spec):
    root = Path(spec["root"])
    prior = Path(spec["prior"])
    baseline = Path(spec["baseline"])
    root.joinpath("prepared").mkdir(exist_ok=False)
    p, atoms, case, provenance = source_density(prior)
    save(root / "prepared/source.json", provenance)
    np.save(root / "prepared/model_H_density.npy", p)
    sys.path.insert(0, str(prior / "supervisor/tools/openmx"))
    from cp2k_wfn import BasisLayout

    for family in ("molopt", "fitted"):
        old = prior / "prepared/sio2" / family
        st = target_metric(old)
        b = np.load(old / "cross_overlap.npy")
        basis = (
            prior
            / "prepared/sio2/bases"
            / ("BASIS_FITTED" if family == "fitted" else "BASIS_MOLOPT_LABELLED")
        )
        assert (
            sha(basis)
            == json.loads((old / "overlap_checks.json").read_text())[
                "target_basis_sha256"
            ]
        )
        names = [
            (
                ("OMX-FIT-" + case["paos"][a]["selection"])
                if family == "fitted"
                else "CMP-MOLOPT-" + a
            )
            for a in atoms
        ]
        layout = BasisLayout.from_basis_file(basis, names)
        # Same multiplicity alone is insufficient: enforce corresponding s,p,d slots.
        fitted_layout = BasisLayout.from_basis_file(
            prior / "prepared/sio2/bases/BASIS_FITTED",
            ["OMX-FIT-" + case["paos"][a]["selection"] for a in atoms],
        )
        assert np.array_equal(layout.permutation(), fitted_layout.permutation())
        assert len(layout.permutation()) == 1950
        template_path = baseline / "sio2" / family / "atomic/benchmark-RESTART.kp"
        template = read_cp2k_density_restart(template_path)
        _, masks, native, control = certify(template, st, 800)
        del control
        for image in template.spins[0]:
            image.matrix = np.empty((0, 0))
        original = (baseline / "sio2" / family / "atomic/input.inp").read_text()
        for mapping in ("projection", "coefficient"):
            directory = root / "prepared" / (family + "_" + mapping)
            directory.mkdir()
            with threadpool_limits(limits=1):
                mapped, report = map_density(
                    p,
                    st,
                    electrons=800,
                    **(
                        dict(cross=b)
                        if mapping == "projection"
                        else dict(permutation=layout.permutation())
                    ),
                )
            union = np.logical_or.reduce(masks)
            union |= union.T
            represented = (
                (
                    mapped.reshape(150, 13, 150, 13).transpose(0, 2, 1, 3)
                    * union[:, :, None, None]
                )
                .transpose(0, 2, 1, 3)
                .reshape(mapped.shape)
            )
            assert abs(np.sum(represented * st) - 800) < 1e-5
            payload = directory / "guess.kp"
            packed = pack_gamma(template, mapped, masks)
            write_cp2k_density_restart(payload, packed)
            del packed
            identity = dict(
                mapping=report,
                native=native,
                basis_sha256=sha(basis),
                cross_overlap_sha256=sha(old / "cross_overlap.npy"),
                payload_sha256=sha(payload),
                template_sha256=sha(template_path),
                original_input_sha256=sha(
                    baseline / "sio2" / family / "atomic/input.inp"
                ),
                mapping_interpretation=(
                    "Exploratory coefficient transplantation: radial ordinal slots identified, not a fitted OpenMX basis"
                    if family == "molopt" and mapping == "coefficient"
                    else mapping
                ),
            )
            save(directory / "identity.json", identity)
            # Mandatory exact-path, one-step CP2K preflight before long runs.
            result = run(
                directory / "preflight",
                input_text(original, steps=1, restart=payload),
                electrons=800,
                timeout=3600,
                expected="RESTART",
            )
            save(
                directory / "READY.json",
                dict(payload_sha256=identity["payload_sha256"], preflight=result),
            )
        del template
    save(root / "PREPARED.json", dict(arms=spec["arms"], preflights_passed=4))


def calculate(spec, family, mapping):
    root = Path(spec["root"])
    assert (root / "PREPARED.json").exists()
    directory = root / "prepared" / (family + "_" + mapping)
    ready = json.loads((directory / "READY.json").read_text())
    payload = directory / "guess.kp"
    assert sha(payload) == ready["payload_sha256"]
    base = Path(spec["baseline"]) / "sio2" / family / "atomic"
    original = (base / "input.inp").read_text()
    result = run(
        root / "results" / (family + "_" + mapping),
        input_text(original, restart=payload),
        electrons=800,
        timeout=spec["timeout"][family],
        expected="RESTART",
    )
    atomic = json.loads((base / "summary.json").read_text())
    result.update(
        atomic_scf_steps=atomic["scf_steps"],
        atomic_wall_seconds=atomic["wall_seconds"],
        energy_difference_from_atomic_Ha=result["energy_hartree"]
        - atomic["energy_hartree"],
        steps_saved=(
            atomic["scf_steps"] - result["scf_steps"] if result["converged"] else None
        ),
        includes_transfer_time=False,
    )
    save(root / "results" / (family + "_" + mapping) / "summary.json", result)
    if (
        not result["converged"]
        or abs(result["energy_difference_from_atomic_Ha"]) > 1e-5
    ):
        raise RuntimeError(
            "Scientific convergence/endpoint gate failed; outputs retained"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--mode", choices=("prepare", "calculate"), required=True)
    ap.add_argument("--family", choices=("molopt", "fitted"), default="molopt")
    ap.add_argument(
        "--mapping", choices=("projection", "coefficient"), default="projection"
    )
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.mode == "prepare":
        prepare(spec)
    else:
        calculate(spec, args.family, args.mapping)


if __name__ == "__main__":
    main()
