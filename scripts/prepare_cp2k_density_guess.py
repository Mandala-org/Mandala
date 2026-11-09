"""Certify and prepare restricted Si s2p2d1 AO guesses for CP2K 2023.1.

Run on an allocated compute node. The native CP2K overlap is an initialization
layout/metric oracle; its converged density is used only for an import control.
The learned guess uses the frozen paper checkpoint's density head.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.cp2k_projection import (
    gaussian_pao_cross_overlap,
    infer_image_layout,
    metric_project_density,
    pack_images,
    pao_quadrature,
    read_cp2k_shells,
    unpack_images,
)
from analysis.openmx_density_grid import load_openmx_snapshot, parse_openmx_pao
from core.basis_converter import OpenMXE3NNConverter
from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.cp2k_restart import read_cp2k_density_restart, write_cp2k_density_restart
from data.structure_inference import build_model_input_from_structure
from net.e3gnn import E3GNN
from scripts.evaluate_checkpoint_materials import (
    _load_checkpoint,
    _restore_config,
    _sanitize_eval_config,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def emit(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2) + "\n")


def geometry(conf):
    text = conf.read_text()
    block = re.search(
        r"<Atoms.SpeciesAndCoordinates[^\n]*\n(.*?)Atoms.SpeciesAndCoordinates>",
        text,
        re.S,
    )[1]
    rows = [line.split() for line in block.splitlines() if line.strip()]
    if len(rows) != 8 or any(row[1] != "Si" for row in rows):
        raise ValueError("Expected eight Si atoms")
    positions = np.array([[float(x) for x in row[2:5]] for row in rows])
    block = re.search(r"<Atoms.UnitVectors[^\n]*\n(.*?)Atoms.UnitVectors>", text, re.S)[
        1
    ]
    box = np.array(
        [list(map(float, line.split())) for line in block.splitlines() if line.strip()]
    )
    return positions, box


def overlap_prints(path, nao, nimage):
    text = path.read_text()
    pieces = text.split("\n OVERLAP MATRIX\n")[1:]
    if len(pieces) != nimage:
        raise ValueError(f"Expected {nimage} native overlaps, found {len(pieces)}")
    output = []
    for piece in pieces:
        a = np.full((nao, nao), np.nan)
        columns = None
        for line in piece.splitlines():
            fields = line.split()
            if fields and all(x.isdigit() for x in fields):
                columns = np.array(fields, dtype=int) - 1
            elif (
                columns is not None
                and len(fields) >= 5
                and fields[0].isdigit()
                and fields[1].isdigit()
                and fields[2] == "Si"
            ):
                if len(fields) - 4 != len(columns):
                    continue
                row = int(fields[0]) - 1
                a[row, columns] = list(map(float, fields[4:]))
                if not np.isnan(a).any():
                    break
        if not np.isfinite(a).all():
            raise ValueError("Incomplete native overlap")
        output.append(a)
    return np.stack(output)


def block_images(matrix):
    if matrix.basis != "openmx":
        raise ValueError("Expected native OpenMX basis")
    result = {}
    for key, edges in matrix.pair_edges.items():
        for index, (sx, sy, sz, i, j) in enumerate(edges.T.tolist()):
            shift = (sx, sy, sz)
            result.setdefault(shift, np.zeros((104, 104)))
            result[shift][13 * i : 13 * (i + 1), 13 * j : 13 * (j + 1)] = (
                matrix.pair_blocks[key][index].double().numpy()
            )
    return result


def forward_transform(images, k):
    cells = np.array(list(images))
    phase = np.exp(2j * np.pi * (k @ cells.T))
    return np.einsum(
        "kr,rij->kij", phase, np.stack(list(images.values())), optimize=True
    )


def inverse_transform(matrices, k, cells):
    cells = np.asarray(list(cells), dtype=int)
    values = np.einsum(
        "rk,kij->rij", np.exp(-2j * np.pi * (cells @ k.T)), matrices, optimize=True
    ) / len(k)
    if np.max(abs(values.imag)) > 2e-9:
        raise ValueError("Time-reversal / image-phase failure")
    return {tuple(cell): value.real for cell, value in zip(cells, values)}


def evaluate_blocks(
    cells,
    positions,
    box,
    shells,
    quadrature,
    *,
    source_cp2k=False,
    source_positions=None,
):
    if source_positions is None:
        source_positions = positions
    d = (
        source_positions[None, None, :, :]
        + (np.asarray(cells) @ box)[:, None, None, :]
        - positions[None, :, None, :]
    )
    array = gaussian_pao_cross_overlap(d.reshape(-1, 3), shells, *quadrature)
    if source_cp2k:
        array = array[:, :, [0, 1, 3, 4, 2, 6, 7, 5, 10, 12, 8, 11, 9]]
    return (
        array.reshape(len(cells), 8, 8, 13, 13)
        .transpose(0, 1, 3, 2, 4)
        .reshape(len(cells), 104, 104)
    )


def gaussian_quadrature(shells):
    x, w = np.polynomial.legendre.leggauss(256)
    end = math.sqrt(60 / min(np.min(a) for _, a, _ in shells))
    r, weights = (x + 1) * end / 2, w * end / 2
    radials = []
    for l in range(3):
        values = []
        for angular, a, c in shells:
            if angular != l:
                continue
            norm = np.sqrt(2 * (2 * a) ** (l + 1.5) / math.gamma(l + 1.5))
            values.append((np.exp(-r[:, None] ** 2 * a) * r[:, None] ** l * norm) @ c)
        radials.append((l, np.stack(values, axis=1)))
    return r, weights, radials


@torch.no_grad()
def predict(checkpoint, positions, box, output, *, density_convention=None):
    if (
        sha(checkpoint)
        == "e2b2f922b10c092e068a67ef879b615b267ba3fcc41e7431043069691c05ef39"
    ):
        raise ValueError(
            "The paper silicon checkpoint is not numerically faithful under the current "
            "model compatibility path (historical replay: 1.88% density error; current: 57%). "
            "Use the frozen historical source e78a7c04e18d005d47def9eb7c9bd0f68b168d99, "
            "then scripts/project_cp2k_density_guess.py with its strict-load, rotation, "
            "inversion, and restricted-total density provenance. See "
            "docs/cp2k_initial_guess_feasibility.md."
        )
    payload = _load_checkpoint(checkpoint)
    cfg = _sanitize_eval_config(_restore_config(payload))
    cfg.device = "cpu"
    cfg.gpus = 0
    # Explicit geometry graph: reference H/S/P values never enter model input.
    cfg.data = dict(
        getattr(cfg, "data", {}),
        graph_source="radius",
        graph_radius=float(cfg.cutoff_radius),
    )
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"Si": "2s2p1d"}))
    model = E3GNN(cfg=cfg, mapper=mapper)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    dtype = next(model.parameters()).dtype
    q = torch.eye(3, dtype=dtype)[[2, 0, 1]]
    pos = torch.as_tensor(positions, dtype=dtype) @ q
    cell = torch.as_tensor(box, dtype=dtype) @ q

    def infer(p, b):
        x = build_model_input_from_structure(
            atoms=("Si",) * 8, positions=p, box=b, cfg=cfg, mapper=mapper
        )
        matrix = model.predict_matrices(x, physical=True)["density"]
        return matrix

    raw = infer(pos, cell)
    raw.save(output / "raw_density_e3nn.pt")
    adjoint = raw.adjoint()
    herm = max(
        float((raw.pair_blocks[key] - adjoint.pair_blocks[key]).abs().max())
        for key in raw.pair_blocks
    )
    rawscale = max(float(value.abs().max()) for value in raw.pair_blocks.values())
    rotations = []
    rotation = torch.tensor(
        [
            [math.cos(0.37), -math.sin(0.37), 0.0],
            [math.sin(0.37), math.cos(0.37), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    for r in (rotation, -torch.eye(3, dtype=dtype)):
        actual = infer(pos @ r.T, cell @ r.T)
        expected = raw.rotate(r)
        error = 0.0
        for edge, (key, index) in actual.lookup.items():
            ekey, eindex = expected.lookup[edge]
            error = max(
                error,
                float(
                    (
                        actual.pair_blocks[key][index]
                        - expected.pair_blocks[ekey][eindex]
                    )
                    .abs()
                    .max()
                ),
            )
        rotations.append(
            {
                "determinant": float(torch.linalg.det(r)),
                "max_error": error,
                "relative_to_max_density": error / max(rawscale, 1e-12),
            }
        )
    if max(x["relative_to_max_density"] for x in rotations) > 1e-3:
        emit(
            output, "model_equivariance.json", {"rotations": rotations, "passed": False}
        )
        raise ValueError("Paper checkpoint failed decoded density equivariance gate")
    projected = (raw + adjoint) * 0.5
    native = OpenMXE3NNConverter(mapper.orbital_cfg).matrix_to_openmx(projected)
    native.save(output / "predicted_density_openmx.pt")
    emit(
        output,
        "model_equivariance.json",
        {
            "rotations": rotations,
            "passed": True,
            "raw_hermiticity_max": herm,
            "graph_radius_angstrom": cfg.cutoff_radius,
            "density_convention": density_convention,
            "density_is_spin_summed": (
                None
                if density_convention is None
                else density_convention == "restricted-total"
            ),
            "paper_run": "92sqde98",
        },
    )
    return native


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pao", type=Path, required=True)
    ap.add_argument("--basis", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--native-log", type=Path, required=True)
    ap.add_argument("--input-template", type=Path, required=True)
    ap.add_argument("--native-probe", type=Path)
    ap.add_argument(
        "--checkpoint-density-convention",
        required=True,
        choices=["openmx-spin0", "restricted-total"],
        help="Training convention, established from checkpoint provenance; do not infer from the current parser",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    sources = [
        args.checkpoint,
        args.pao,
        args.basis,
        args.native,
        args.native_log,
        args.input_template,
        args.source / "HS.out",
        args.source / "Si.out",
        args.source / "conf.dat",
        Path(__file__),
    ]
    if args.native_probe is not None:
        sources.append(args.native_probe)
    contract = {
        "purpose": "Pre-registered AO overlap projection and comparison preparation",
        "source_sha256": {str(p): sha(p) for p in sources},
        "torch": torch.__version__,
        "kmesh": [8, 8, 8],
        "reference_spin_sum_factor": 2,
        "nelectrons": 32,
        "checkpoint_density_convention": args.checkpoint_density_convention,
        "checkpoint_spin_sum_factor": (
            1 if args.checkpoint_density_convention == "restricted-total" else 2
        ),
        "native_overlap_tolerance": 2e-8,
        "native_repacking_tolerance": 1e-10,
        "source_onsite_overlap_tolerance": 2e-6,
        "joint_gram_tolerance": -2e-5,
        "native_electron_count_tolerance": 1e-5,
        "radial_quadrature_order": 4,
        "radial_check_order": 6,
        "paper_run": "92sqde98",
        "checkpoint_selection_uses_this_snapshot": False,
        "metric_repair": "global shared water-filling shift, equal k weights, occupations in [0,2]",
    }
    emit(output, "contract.json", contract)
    start = time.monotonic()
    positions, box = geometry(args.source / "conf.dat")
    # CP2K centers each atom in the central periodic cell before constructing
    # its neighbor lists. Source OpenMX image labels refer to input positions.
    fractional = positions @ np.linalg.inv(box)
    target_positions = (fractional - np.floor(fractional + 0.5)) @ box
    bohr = 1.8897261254578281
    shells = read_cp2k_shells(args.basis, "Si", "DZVP-MOLOPT-SR-GTH")
    if [l for l, _, _ in shells] != [0, 0, 1, 1, 2]:
        raise ValueError("Target shell layout changed")
    restart = read_cp2k_density_restart(args.native)
    if restart.nao != 104 or restart.natom != 8 or len(restart.spins) != 1:
        raise ValueError("Native restart dimensions differ")
    cells = [image.cell for image in restart.spins[0]]
    unique_cells = sorted(set(cells) | {tuple(-np.array(c)) for c in cells})
    print("Certifying native Gaussian overlaps and image layout", flush=True)
    if args.native_probe is None:
        stored_s = overlap_prints(args.native_log, 104, len(cells))
        expected = evaluate_blocks(
            unique_cells,
            target_positions * bohr,
            box * bohr,
            shells,
            gaussian_quadrature(shells),
            source_cp2k=True,
        )
        np.savez(
            output / "gaussian_layout_probe.npz",
            cells=np.asarray(unique_cells),
            expected=expected,
            stored=stored_s,
            native_cells=np.asarray(cells),
        )
    else:
        cache = np.load(args.native_probe)
        np.testing.assert_array_equal(cache["cells"], np.asarray(unique_cells))
        np.testing.assert_array_equal(cache["native_cells"], np.asarray(cells))
        expected, stored_s = cache["expected"], cache["stored"]
    expected = dict(zip(unique_cells, expected))
    layout, overlap_error = infer_image_layout(
        stored_s, expected, cells, block_size=13, printed_diagonal_triangle=True
    )
    # Restore the unprinted lower triangles from the independently integrated
    # Gaussian overlaps, after certifying all printed upper entries.
    for image, cell in enumerate(cells):
        for atom in range(8):
            sign = int(layout[image, atom, atom])
            if sign:
                block = slice(13 * atom, 13 * (atom + 1))
                stored_s[image, block, block] = expected[
                    tuple(sign * np.asarray(cell))
                ][block, block]
    s_images = unpack_images(stored_s, cells, layout, block_size=13)
    native_p = np.stack([image.matrix for image in restart.spins[0]])
    unpacked_p = unpack_images(native_p, cells, layout, block_size=13)
    k1 = (np.arange(8) + 0.5) / 8 - 0.5
    k = np.array(np.meshgrid(k1, k1, k1, indexing="ij")).reshape(3, -1).T
    p_round = pack_images(
        inverse_transform(forward_transform(unpacked_p, k), k, unpacked_p.keys()),
        cells,
        layout,
        block_size=13,
    )
    repack_error = float(np.max(abs(p_round - native_p)))
    if repack_error > 1e-10:
        raise ValueError(f"Image/Fourier roundtrip failed: {repack_error}")
    target_s = forward_transform(s_images, k)
    native_pk = forward_transform(unpacked_p, k)
    native_electrons = float(np.einsum("kij,kji->k", native_pk, target_s).real.mean())
    if abs(native_electrons - 32) > 1e-5:
        raise ValueError(
            f"Native density image/metric electron-count failure: {native_electrons}"
        )
    emit(
        output,
        "native_layout.json",
        {
            "passed": True,
            "overlap_max_error": overlap_error,
            "image_fourier_repacking_error": repack_error,
            "nimages": len(cells),
            "physical_images": len(unique_cells),
            "target_positions_angstrom": target_positions.tolist(),
            "source_to_target_integer_shifts": (
                -np.floor(fractional + 0.5).astype(int)
            ).tolist(),
            "native_electrons": native_electrons,
            "native_diagonal_block_asymmetry": float(
                max(np.max(abs(a[:13, :13] - a[:13, :13].T)) for a in native_p)
            ),
        },
    )
    np.savez(
        output / "native_layout.npz",
        cells=np.array(cells),
        layout=layout,
        kpoints_fractional=k,
        target_overlap=target_s,
    )

    snapshot = load_openmx_snapshot(
        args.source / "HS.out", args.source / "Si.out", dtype=torch.float64
    )
    if (
        snapshot.density.orbital_cfg.to_dict()
        != OrbitalIrrepConfig.from_dict({"Si": "2s2p1d"}).to_dict()
    ):
        raise ValueError("Source orbital inventory changed")
    pao = parse_openmx_pao(args.pao, dtype=torch.float64)
    quadrature = pao_quadrature(
        pao.radii.numpy(),
        {l: v.numpy() for l, v in pao.values_by_l.items()},
        [2, 2, 1],
        order=4,
    )
    finer = pao_quadrature(
        pao.radii.numpy(),
        {l: v.numpy() for l, v in pao.values_by_l.items()},
        [2, 2, 1],
        order=6,
    )
    source_s = block_images(snapshot.overlap)
    gram = np.zeros((13, 13))
    offset = 0
    for l, radial in quadrature[2]:
        g = radial.T @ ((quadrature[1] * quadrature[0] ** 2)[:, None] * radial)
        block = np.kron(g, np.eye(2 * l + 1))
        n = block.shape[0]
        gram[offset : offset + n, offset : offset + n] = block
        offset += n
    source_error = max(
        float(
            np.max(
                abs(
                    source_s[(0, 0, 0)][13 * i : 13 * (i + 1), 13 * i : 13 * (i + 1)]
                    - gram
                )
            )
        )
        for i in range(8)
    )
    if source_error > 2e-6:
        raise ValueError(
            f"Official PAO disagrees with source onsite overlap: {source_error}"
        )
    # Training-independent quadrature certification at preselected displacements.
    probes = np.array([[0.0, 0.0, 0.0], [1.0, -0.8, 2.3], [6.0, -3.0, 1.0]])
    check_a = gaussian_pao_cross_overlap(probes, shells, *quadrature)
    check_b = gaussian_pao_cross_overlap(probes, shells, *finer)
    quadrature_error = float(np.max(abs(check_a - check_b)))
    if quadrature_error > 2e-8:
        raise ValueError(f"Radial quadrature not certified: {quadrature_error}")
    emit(
        output,
        "source_basis_checks.json",
        {
            "onsite_overlap_error": source_error,
            "cross_quadrature_error": quadrature_error,
            "official_pao_sha256": sha(args.pao),
            "private_source_copy_hash_known": False,
        },
    )
    print("Integrating numerical PAO/Gaussian cross overlaps", flush=True)
    mixed = evaluate_blocks(
        unique_cells,
        target_positions * bohr,
        box * bohr,
        shells,
        quadrature,
        source_positions=positions * bohr,
    )
    np.savez(output / "cross_overlap.npz", cells=np.array(unique_cells), overlap=mixed)
    cross_k = forward_transform(dict(zip(unique_cells, mixed)), k)
    source_sk = forward_transform(source_s, k)
    residual = source_sk - cross_k.swapaxes(-1, -2).conj() @ np.linalg.solve(
        target_s, cross_k
    )
    residual = 0.5 * (residual + residual.swapaxes(-1, -2).conj())
    loss_eigen = np.linalg.eigvalsh(residual)
    minimum = float(loss_eigen.min())
    emit(
        output,
        "projection_residual.json",
        {
            "minimum_gram_residual_eigenvalue": minimum,
            "mean_basis_norm_loss": float(
                np.trace(residual, axis1=1, axis2=2).real.mean() / 104
            ),
            "source_overlap_min_eigenvalue": float(np.linalg.eigvalsh(source_sk).min()),
        },
    )
    if minimum < -2e-5:
        raise ValueError("Cross/source/target overlaps violate joint Gram positivity")
    predicted = predict(
        args.checkpoint,
        positions,
        box,
        output,
        density_convention=args.checkpoint_density_convention,
    )
    reports = {}
    for name, matrix in [
        ("reference_openmx", snapshot.density),
        ("mandala", predicted),
    ]:
        source_p = block_images((matrix + matrix.adjoint()) * 0.5)
        factor = (
            2
            if name == "reference_openmx"
            or args.checkpoint_density_convention == "openmx-spin0"
            else 1
        )
        source_pk = factor * forward_transform(source_p, k)
        projected, diagnostics = metric_project_density(
            source_pk, target_s, cross_k, 32
        )
        physical = inverse_transform(projected, k, unique_cells)
        packed = pack_images(physical, cells, layout, block_size=13)
        guess = copy.deepcopy(restart)
        for image, density in zip(guess.spins[0], packed):
            image.matrix = density
        write_cp2k_density_restart(output / (name + ".kp"), guess)
        raw_count = np.einsum("kij,kji->k", source_pk, source_sk).real.mean()
        diagnostics["raw_source_electrons"] = float(raw_count)
        diagnostics["spin_sum_factor"] = factor
        reports[name] = diagnostics
    emit(output, "density_projection.json", reports)
    # Only output paths/guess differ from the frozen 14-step atomic calculation.
    text = args.input_template.read_text()
    text = re.sub(r"\s*&PRINT\s*\n\s*&AO_MATRICES\b.*?&END PRINT", "", text, flags=re.S)
    text = text.replace("&RESTART ON", "&RESTART OFF")
    cases = [
        ("atomic", None),
        ("native_original", args.native.resolve()),
        ("native_serialized", (output / "native_serialized.kp")),
        ("reference_openmx", (output / "reference_openmx.kp")),
        ("mandala", (output / "mandala.kp")),
    ]
    write_cp2k_density_restart(output / "native_serialized.kp", restart)
    case_paths = []
    for name, guess in cases:
        directory = output / "comparison" / name
        directory.mkdir(parents=True)
        inp = text
        if guess is not None:
            inp = inp.replace("SCF_GUESS ATOMIC", "SCF_GUESS RESTART")
            inp = inp.replace(
                "&DFT", "&DFT\n    WFN_RESTART_FILE_NAME " + str(guess), 1
            )
        (directory / "input.inp").write_text(inp)
        case_paths.append(
            {
                "name": name,
                "directory": str(directory),
                "input_sha256": sha(directory / "input.inp"),
            }
        )
    emit(
        output,
        "comparison_cases.json",
        {
            "cases": case_paths,
            "preparation_seconds": time.monotonic() - start,
            "scientific_settings": "same as 14-step ATOMIC baseline; native controls diagnose import",
            "no_scf_result_used_to_choose_checkpoint_or_projection": True,
        },
    )
    emit(
        output,
        "READY.json",
        {"prepared": True, "cases": [c["name"] for c in case_paths]},
    )
    print("Comparison cases prepared", flush=True)


if __name__ == "__main__":
    main()
