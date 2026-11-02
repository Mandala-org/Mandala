"""Evaluate a trained checkpoint on a reference calculation.

Prints matrix errors and the band energy, and plots predicted against
reference bands. Run from the repository root, for example::

    PYTHONPATH=src:. python examples/quickstart/evaluate.py \
        --checkpoint runs/silicon/best_model.pt \
        --structure data/small/Si_perturbed_scale1_078 --plot bands.png

``--plugin`` imports a module that registers custom transformations before
the checkpoint is restored (needed for models built from custom layers).
"""

import argparse
import importlib.util
from pathlib import Path

import torch

from core.sparse_math import trace_matmul_sparse_block_matrix
from data.snapshot import Snapshot
from data.sources import openmx_info_path
from net.common import Config
from net.evaluation import (
    predict_snapshot,
    predictions_to_snapshot,
    restore_model_for_snapshot,
)
from utils.units import HARTREE_TO_EV

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--structure", type=Path, required=True)
parser.add_argument("--plugin", type=Path)
parser.add_argument("--plot", type=Path)
args = parser.parse_args()

if args.plugin:
    spec = importlib.util.spec_from_file_location(args.plugin.stem, args.plugin)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

reference = Snapshot.from_openmx(
    args.structure / "HS.out",
    openmx_info_path(args.structure),
    convention="e3nn",
    cfg=Config(verbosity=0),
)
model = restore_model_for_snapshot(args.checkpoint, reference)
predicted = predict_snapshot(model, reference)
H = predicted["hamiltonian"]

# Compare on the directed edges (Lx, Ly, Lz, i, j) shared by the prediction
# and the reference, in the orbital convention of the prediction.
if H.basis == "openmx":
    reference = reference.to_openmx()


def blocks_by_edge(matrix):
    return {
        tuple(edge): block
        for key in matrix.keys()
        for edge, block in zip(matrix.edges(key).T.tolist(), matrix[key])
    }


reference_blocks = {name: blocks_by_edge(reference[name]) for name in predicted}
predicted_blocks = {name: blocks_by_edge(matrix) for name, matrix in predicted.items()}
shared = [
    e for e in predicted_blocks["hamiltonian"] if e in reference_blocks["hamiltonian"]
]
print(f"compared blocks: {len(shared)}")

for name in predicted:
    error = torch.stack(
        [(predicted_blocks[name][e] - reference_blocks[name][e]).abs() for e in shared]
    )
    scale = HARTREE_TO_EV * 1000 if name == "hamiltonian" else 1.0
    unit = "meV" if name == "hamiltonian" else ""
    print(
        f"{name:12s} MAE {float(error.mean()) * scale:9.4f} {unit}"
        f"  max {float(error.max()) * scale:9.4f} {unit}"
    )


def band_energy(hamiltonian, density):
    """Tr(DH) = sum over edges of tr(H_ij^L D_ji^-L), restricted to the shared edges."""
    total = sum(
        torch.trace(hamiltonian[(sx, sy, sz, i, j)] @ density[(-sx, -sy, -sz, j, i)])
        for sx, sy, sz, i, j in shared
    )
    return float(total) * HARTREE_TO_EV / len(H.atoms)


density = (
    reference_blocks["density"]
    if "density" in reference_blocks
    else blocks_by_edge(reference.density)
)
print(
    f"band energy Tr(D_ref H) per atom: reference "
    f"{band_energy(reference_blocks['hamiltonian'], density):.4f} eV, "
    f"predicted {band_energy(predicted_blocks['hamiltonian'], density):.4f} eV"
)

if args.plot:
    import matplotlib.pyplot as plt

    # Predicted Hamiltonian with the reference overlap on its full support.
    combined = predictions_to_snapshot(
        {"hamiltonian": H}, reference, reference_matrices=("overlap", "density")
    )
    bands = {
        "reference": reference.get_band_structure(path="GXMGR", npoints=80),
        "MANDALA": combined.get_band_structure(path="GXMGR", npoints=80),
    }
    # Tr(DS) per spin channel is the number of doubly occupied bands.
    occupied = round(
        float(trace_matmul_sparse_block_matrix(reference.density, reference.overlap))
    )
    vbm = float(bands["reference"].eigenvalues[:, occupied - 1].max())
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    styles = (
        {"color": "black", "lw": 1.6},
        {"color": "#d62728", "lw": 1.0, "ls": "--"},
    )
    for (label, result), style in zip(bands.items(), styles):
        energies = (result.eigenvalues - vbm) * HARTREE_TO_EV
        for band in range(energies.shape[1]):
            ax.plot(
                result.linear_k,
                energies[:, band],
                label=label if band == 0 else None,
                **style,
            )
    ax.set_xticks(bands["reference"].tick_positions, bands["reference"].tick_labels)
    ax.set_xlim(
        float(bands["reference"].linear_k[0]), float(bands["reference"].linear_k[-1])
    )
    ax.set_ylim(-13, 6)
    ax.set_ylabel("E - E$_\\mathrm{VBM}$ (eV)")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=160)
    print(f"band structure written to {args.plot}")
