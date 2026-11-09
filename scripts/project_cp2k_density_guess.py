"""Project a certified prediction using frozen CP2K/PAO overlap integrals.

This deliberately does not restore a model. Historical checkpoints must run
under an implementation that reproduces their published predictions. Run on
an allocated node or an explicitly authorized standalone compute host.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.cp2k_projection import metric_project_density, pack_images, unpack_images
from core.basis_converter import OpenMXE3NNConverter
from data.block_matrix import BlockMatrix
from data.cp2k_restart import read_cp2k_density_restart, write_cp2k_density_restart


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fourier(images, k):
    cells = np.asarray(list(images))
    return np.einsum(
        "kr,rij->kij",
        np.exp(2j * np.pi * (k @ cells.T)),
        np.stack(list(images.values())),
        optimize=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "native-layout",
        "cross-overlap",
        "native-template",
        "density-matrix",
        "prediction-report",
        "prediction-contract",
        "output",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument(
        "--density-convention",
        required=True,
        choices=["openmx-spin0", "restricted-total"],
    )
    args = ap.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite a density guess or its evidence")
    report = json.loads(args.prediction_report.read_text())
    if report.get("strict_load") is not True:
        raise ValueError("A strictly loaded prediction is required")
    if report.get("density_convention") != args.density_convention:
        raise ValueError("Prediction/export spin conventions disagree")
    rotations = report.get("rotations", [])
    if (
        len(rotations) != 2
        or min(r["determinant"] for r in rotations) >= 0
        or max(r["determinant"] for r in rotations) <= 0
        or max(r["max_error"] for r in rotations) > 1e-4
    ):
        raise ValueError(
            "Decoded rotation/inversion certification is missing or failed"
        )
    identities = {
        name: digest(getattr(args, name))
        for name in (
            "native_layout",
            "cross_overlap",
            "native_template",
            "density_matrix",
            "prediction_report",
            "prediction_contract",
        )
    }
    cache = np.load(args.native_layout)
    cross = np.load(args.cross_overlap)
    k, s = cache["kpoints_fractional"], cache["target_overlap"]
    cells, layout = cache["cells"], cache["layout"]
    if k.shape != (512, 3) or s.shape != (512, 104, 104):
        raise ValueError("Expected the certified eight-Si full 8x8x8 integration case")
    cross_k = fourier(dict(zip(map(tuple, cross["cells"]), cross["overlap"])), k)
    p = BlockMatrix.load(args.density_matrix)
    if p.basis == "e3nn":
        p = OpenMXE3NNConverter(p.orbital_cfg).matrix_to_openmx(p)
    if p.basis != "openmx" or tuple(p.atoms) != ("Si",) * 8:
        raise ValueError("Expected the eight-Si native OpenMX density")
    p = (p + p.adjoint()) * 0.5
    images = {}
    for (sx, sy, sz, i, j), (key, index) in p.lookup.items():
        image = images.setdefault((sx, sy, sz), np.zeros((104, 104)))
        image[13 * i : 13 * (i + 1), 13 * j : 13 * (j + 1)] = (
            p.pair_blocks[key][index].double().numpy()
        )
    factor = 2 if args.density_convention == "openmx-spin0" else 1
    projected, diagnostics = metric_project_density(
        factor * fourier(images, k), s, cross_k, 32
    )
    phase = np.exp(-2j * np.pi * (cross["cells"] @ k.T))
    values = np.einsum("rk,kij->rij", phase, projected, optimize=True) / len(k)
    imaginary = float(np.max(abs(values.imag)))
    if imaginary > 2e-9:
        raise ValueError(f"Time-reversal/image-phase failure: {imaginary}")
    packed = pack_images(
        dict(zip(map(tuple, cross["cells"]), values.real)), cells, layout, block_size=13
    )
    restart = copy.deepcopy(read_cp2k_density_restart(args.native_template))
    if (
        restart.nao != 104
        or len(restart.spins) != 1
        or [i.cell for i in restart.spins[0]] != list(map(tuple, cells))
    ):
        raise ValueError(
            "Native template metadata/image labels differ from certified layout"
        )
    for image, value in zip(restart.spins[0], packed):
        image.matrix = value
    # Verify charge after finite native image packing, not only in dense k space.
    reloaded_k = fourier(unpack_images(packed, cells, layout, block_size=13), k)
    count = float(np.einsum("kij,kji->k", reloaded_k, s).real.mean())
    if abs(count - 32) > 1e-7:
        raise ValueError(f"Finite native packing changed the electron count: {count}")
    write_cp2k_density_restart(args.output, restart)
    diagnostics.update(
        input_sha256=identities,
        density_convention=args.density_convention,
        spin_sum_factor=factor,
        native_packed_electrons=count,
        output_sha256=digest(args.output),
        prediction_contract=json.loads(args.prediction_contract.read_text()),
    )
    with args.output.with_suffix(".json").open("x") as f:
        json.dump(diagnostics, f, indent=2)
        f.write("\n")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
