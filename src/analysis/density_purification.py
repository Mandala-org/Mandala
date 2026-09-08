"""Evaluation-only McWeeny comparisons on periodic, shift-resolved blocks."""

from __future__ import annotations

import math
import time

import torch

from core.density_purification import McWeenyPurifier
from core.sparse_math import (
    build_trace_alignment_from_pair_edges,
    trace_matmul_sparse_block_matrix_aligned,
)
from data.block_matrix import BlockMatrix
from utils.units import HARTREE_TO_EV


def on_support(matrix: BlockMatrix, support: BlockMatrix) -> BlockMatrix:
    """Gather exact periodic edges, zero-filling absent blocks; no image folding."""
    if (
        matrix.atoms != support.atoms
        or matrix.basis != support.basis
        or matrix.orbital_cfg.to_dict() != support.orbital_cfg.to_dict()
    ):
        raise ValueError(
            "Reference and prediction must share atoms, AO basis and orbitals."
        )
    blocks = {}
    for key, template in support.pair_blocks.items():
        rows = []
        for edge in support.pair_edges[key].T.cpu().tolist():
            match = matrix.lookup.get(tuple(edge))
            rows.append(
                template.new_zeros(template.shape[1:])
                if match is None
                else matrix.pair_blocks[match[0]][match[1]]
            )
        blocks[key] = torch.stack(rows) if rows else torch.zeros_like(template)
    return support._replace_pair_blocks(blocks, basis=support.basis)


def density_errors(prediction: BlockMatrix, reference: BlockMatrix) -> dict:
    aligned = on_support(reference, prediction)
    errors = [
        prediction.pair_blocks[key] - aligned.pair_blocks[key]
        for key in prediction.pair_blocks
    ]
    # Report on the union: missing predicted edges are zeros, not excluded errors.
    missing_count = 0
    for edge, (key, index) in reference.lookup.items():
        if edge not in prediction.lookup:
            errors.append(-reference.pair_blocks[key][index])
            missing_count += 1
    count = sum(value.numel() for value in errors)
    absolute = sum(value.abs().sum().item() for value in errors)
    squared = sum(value.square().sum().item() for value in errors)
    norm_ref = sum(
        value.square().sum().item() for value in reference.pair_blocks.values()
    )
    return dict(
        mae=absolute / count,
        rmse=math.sqrt(squared / count),
        relative_frobenius=math.sqrt(squared / max(norm_ref, 1e-300)),
        scalar_count=count,
        missing_prediction_edges=missing_count,
    )


def evaluate_purification(
    density: BlockMatrix,
    overlap: BlockMatrix,
    reference_density: BlockMatrix,
    reference_overlap: BlockMatrix,
    reference_hamiltonian: BlockMatrix,
    *,
    predicted_hamiltonian: BlockMatrix | None = None,
    chunk_size: int = 4096,
    spin_degeneracy: int = 1,
) -> dict:
    """Return all four pre-registered iterates plus a reference-density control.

    Iteration uses the raw fixed-support polynomial. Each row additionally
    reports its global reverse-pair Hermitian projection for physical metrics;
    that projection is NOT fed into the next iterate. Reference H is held fixed
    to isolate density-induced band-energy error. Predicted H, when supplied,
    gives the full model Tr(D H_pred) error against Tr(D_ref H_ref).
    spin_degeneracy explicitly converts per-spin traces to total energies and
    electron counts; it never rescales the density used in purification.
    """
    if isinstance(spin_degeneracy, bool) or spin_degeneracy not in (1, 2):
        raise ValueError("spin_degeneracy must be 1 or 2.")

    def double(matrix):
        return matrix._replace_pair_blocks(
            {
                key: value.to(dtype=torch.float64)
                for key, value in matrix.pair_blocks.items()
            },
            basis=matrix.basis,
        )

    density, overlap, reference_density, reference_overlap, reference_hamiltonian = map(
        double,
        (density, overlap, reference_density, reference_overlap, reference_hamiltonian),
    )
    if predicted_hamiltonian is not None:
        predicted_hamiltonian = double(predicted_hamiltonian)
    # Geometry/orbital identity is checked before comparing matrices.
    ref_on_support = on_support(reference_density, density)
    s_true = on_support(reference_overlap, density)
    h_true = on_support(reference_hamiltonian, density)
    h_pred = (
        None
        if predicted_hamiltonian is None
        else on_support(predicted_hamiltonian, density)
    )
    reverse = build_trace_alignment_from_pair_edges(density.pair_edges)
    ref_reverse = build_trace_alignment_from_pair_edges(reference_density.pair_edges)
    ref_h = on_support(reference_hamiltonian, reference_density)
    ref_s = on_support(reference_overlap, reference_density)

    def trace(d, a, alignment=reverse):
        return trace_matmul_sparse_block_matrix_aligned(d, a, alignment).item()

    energy_ref = spin_degeneracy * trace(reference_density, ref_h, ref_reverse)
    occupation_ref = trace(reference_density, ref_s, ref_reverse)
    electrons_ref = spin_degeneracy * occupation_ref
    start = time.perf_counter()
    purifier = McWeenyPurifier(density, overlap, chunk_size=chunk_size)
    plan_seconds = time.perf_counter() - start
    control_purifier = McWeenyPurifier(ref_on_support, s_true, chunk_size=chunk_size)
    rows = []
    current, control = density, ref_on_support
    for iteration in range(4):
        start = time.perf_counter()
        if iteration:
            current = purifier.step(current)
        if current.pair_blocks and next(iter(current.pair_blocks.values())).is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not all(torch.isfinite(v).all() for v in current.pair_blocks.values()):
            raise FloatingPointError(
                f"Non-finite density at McWeeny iteration {iteration}."
            )
        if iteration:
            control = control_purifier.step(control)
        physical = current.symmetrize_aligned(reverse)
        control_physical = control.symmetrize_aligned(reverse)
        energy = spin_degeneracy * trace(physical, h_true)
        occupation = trace(physical, s_true)
        electrons = spin_degeneracy * occupation
        transposed = current.transpose_aligned(reverse)
        herm_errors = [
            v - transposed.pair_blocks[k] for k, v in current.pair_blocks.items()
        ]
        row = dict(
            iteration=iteration,
            raw_density=density_errors(current, reference_density),
            physical_density=density_errors(physical, reference_density),
            raw_hermiticity_max=max(v.abs().max().item() for v in herm_errors),
            band_energy_ref_h_ha=energy,
            band_energy_ref_h_abs_error_ev=abs(energy - energy_ref) * HARTREE_TO_EV,
            band_energy_ref_h_abs_error_ev_per_atom=abs(energy - energy_ref)
            * HARTREE_TO_EV
            / len(density.atoms),
            electron_count=electrons,
            occupation_trace=occupation,
            electron_count_abs_error=abs(electrons - electrons_ref),
            reference_control_density=density_errors(
                control_physical, reference_density
            ),
            step_seconds=elapsed,
        )
        if h_pred is not None:
            energy_pred = spin_degeneracy * trace(physical, h_pred)
            row.update(
                band_energy_pred_h_ha=energy_pred,
                band_energy_pred_h_abs_error_ev=abs(energy_pred - energy_ref)
                * HARTREE_TO_EV,
            )
        rows.append(row)
    baseline = rows[0]["physical_density"]["mae"]
    for row in rows:
        row["density_mae_improvement_percent"] = (
            100 * (1 - row["physical_density"]["mae"] / baseline) if baseline else None
        )
    return dict(
        rows=rows,
        atoms=len(density.atoms),
        edges=len(density.lookup),
        reference_band_energy_ha=energy_ref,
        reference_electrons=electrons_ref,
        reference_occupation_trace=occupation_ref,
        spin_degeneracy=spin_degeneracy,
        plan_seconds=plan_seconds,
        sd_paths=purifier.sd_alignment.num_paths,
        dd_paths=purifier.dd_alignment.num_paths,
        chunk_size=chunk_size,
    )
