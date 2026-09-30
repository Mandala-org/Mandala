"""Template-free real OpenMX text-matrix export (not binary SCFOUT).

Generation_ATV order: origin first, then i/j/k ascending with k innermost.
The inverse ratv table, not an arithmetic index formula, supplies Rn.
Source: https://raw.githubusercontent.com/rigarash/openmx/master/source/openmx_common.c
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from core.basis_converter import OpenMXE3NNConverter
from core.sparse_math import build_trace_alignment_from_pair_edges
from data.block_matrix import BlockMatrix


@dataclass(frozen=True)
class OpenMXWriteStats:
    """Counts for the predicted matrix sections written to the output."""

    blocks_written: Mapping[str, int]
    density_sections: int


@dataclass(frozen=True)
class OpenMXTranslationTable:
    cpy_cell: int
    atv_ijk: np.ndarray
    ratv: np.ndarray

    @property
    def tcpy_cell(self):
        return len(self.atv_ijk) - 1

    def rn(self, shift):
        shift = tuple(shift)
        if len(shift) != 3 or any(
            not isinstance(n, (int, np.integer)) or abs(n) > self.cpy_cell
            for n in shift
        ):
            raise ValueError("Translation must contain three integers within CpyCell.")
        return int(self.ratv[tuple(n + self.cpy_cell for n in shift)])


def build_openmx_translation_table(cpy_cell: int) -> OpenMXTranslationTable:
    """Reproduce OpenMX Generation_ATV, and invert atv_ijk into ratv."""
    if isinstance(cpy_cell, bool) or not isinstance(cpy_cell, int) or cpy_cell < 0:
        raise ValueError("CpyCell must be a nonnegative integer.")
    shifts = [(0, 0, 0)]
    for i in range(-cpy_cell, cpy_cell + 1):
        for j in range(-cpy_cell, cpy_cell + 1):
            for k in range(-cpy_cell, cpy_cell + 1):
                if (i, j, k) != (0, 0, 0):
                    shifts.append((i, j, k))
    atv_ijk = np.zeros((len(shifts), 4), dtype=np.int64)
    atv_ijk[:, 1:4] = shifts
    ratv = np.empty((2 * cpy_cell + 1,) * 3, dtype=np.int64)
    for rn, row in enumerate(atv_ijk):
        ratv[tuple(row[1:4] + cpy_cell)] = rn
    atv_ijk.setflags(write=False)
    ratv.setflags(write=False)
    return OpenMXTranslationTable(cpy_cell, atv_ijk, ratv)


def openmx_cutoff_layout(atoms, positions, lattice, cutoff_radius):
    """Return a complete cutoff graph and its minimal finite translation table.

    Positions/lattice are native Cartesian Angstrom coordinates, NOT wrapped.
    ASE's general-cell neighbor search handles skewed cells and unwrapped atoms.
    This is an OpenMX-compatible table, not OpenMX's solver-dependent choice of
    potentially larger CpyCell. Cutoff equality is excluded, as in model graphs.
    """
    if not math.isfinite(cutoff_radius) or cutoff_radius <= 0:
        raise ValueError("Matrix cutoff must be finite and positive.")
    positions = torch.as_tensor(positions).detach().cpu().double().numpy()
    if (
        not atoms
        or positions.shape != (len(atoms), 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError(
            "Expected finite (N,3) positions and nonempty atom identities."
        )
    if lattice is not None:
        lattice = torch.as_tensor(lattice).detach().cpu().double().numpy()
        if (
            lattice.shape != (3, 3)
            or not np.isfinite(lattice).all()
            or abs(np.linalg.det(lattice)) < 1e-12
        ):
            raise ValueError("Expected a finite nonsingular (3,3) lattice.")
    structure = Atoms(
        symbols=atoms, positions=positions, cell=lattice, pbc=lattice is not None
    )
    src, dst, shifts = neighbor_list(
        "ijS", structure, cutoff_radius, self_interaction=False
    )
    edges = {(0, 0, 0, i, i) for i in range(len(atoms))}
    edges.update(
        (*map(int, shift), int(i), int(j)) for i, j, shift in zip(src, dst, shifts)
    )
    cpy_cell = max(abs(n) for edge in edges for n in edge[:3])
    table = build_openmx_translation_table(cpy_cell)
    # OpenMX Trn_System loops over atom j, then Rn; onsite is local index 0.
    ordered = sorted(
        edges, key=lambda e: (e[3], e != (0, 0, 0, e[3], e[3]), e[4], table.rn(e[:3]))
    )
    return table, ordered


def write_openmx_from_structure(
    output_path: str | Path,
    predictions: Mapping[str, BlockMatrix],
    *,
    atoms,
    positions,
    lattice,
    cutoff_radius: float,
) -> OpenMXWriteStats:
    """Write predicted sections without reading any reference/template file.

    Real H/S/D blocks use OpenMX headers and 20-place decimal rows. Only supplied
    targets are emitted; unpredicted physical quantities are NEVER fabricated.
    A translation-table preamble carries shifts instead of fake position overlaps.
    All predicted matrices must have exactly the structure's cutoff support.
    """
    if not predictions or set(predictions) - {"hamiltonian", "overlap", "density"}:
        raise ValueError("Supply a nonempty subset of H/S/D predictions.")
    atoms = tuple(atoms)
    table, edges = openmx_cutoff_layout(atoms, positions, lattice, cutoff_radius)
    expected = set(edges)
    first = next(iter(predictions.values()))
    converter = OpenMXE3NNConverter(first.orbital_cfg)
    matrices = {}
    for name, matrix in predictions.items():
        if (
            matrix.atoms != atoms
            or matrix.orbital_cfg.to_dict() != first.orbital_cfg.to_dict()
        ):
            raise ValueError(
                "Predictions must match atom order and orbital configuration."
            )
        if matrix.basis not in {"openmx", "e3nn"}:
            raise ValueError("Only OpenMX/e3nn predictions are supported.")
        matrix = matrix.to("cpu")
        lookup = {}
        for key, block in matrix.pair_blocks.items():
            edge_tensor = matrix.pair_edges[key]
            if (
                block.ndim != 3
                or block.shape[1:] != matrix.orbital_cfg.block_dims(key)
                or edge_tensor.shape != (5, len(block))
                or edge_tensor.dtype != torch.long
            ):
                raise ValueError("Invalid matrix block/edge shape or dtype.")
            if not torch.is_floating_point(block) or not torch.isfinite(block).all():
                raise ValueError(
                    "Predictions must be finite real floating-point blocks."
                )
            for index, edge in enumerate(edge_tensor.T.tolist()):
                edge = tuple(edge)
                if (
                    edge not in expected
                    or edge in lookup
                    or key != f"{atoms[edge[3]]}-{atoms[edge[4]]}"
                ):
                    raise ValueError(
                        "Invalid, duplicate, or outside-cutoff prediction edge."
                    )
                lookup[edge] = (key, index)
        if lookup != matrix.lookup or set(lookup) != expected:
            raise ValueError(
                "Prediction support must exactly match the structure cutoff graph."
            )
        matrix = matrix.symmetrize_aligned(
            build_trace_alignment_from_pair_edges(matrix.pair_edges)
        )
        matrices[name] = converter.matrix_to_openmx(matrix)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=output.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(f"atomnum={len(atoms)}\nCatomnum=0\nLatomnum=0\nRatomnum=0\n")
            handle.write(f"CpyCell={table.cpy_cell}\nTCpyCell={table.tcpy_cell}\n")
            for rn, row in enumerate(table.atv_ijk):
                handle.write(f"atv_ijk Rn={rn} {row[1]} {row[2]} {row[3]}\n")
            titles = {
                "hamiltonian": "Kohn-Sham Hamiltonian spin=0",
                "overlap": "Overlap matrix",
                "density": "Density matrix spin=0",
            }
            for name, title in titles.items():
                if name not in matrices:
                    continue
                matrix = matrices[name]
                handle.write(f"\n\n{title}\n")
                previous, local = None, 0
                for edge in edges:
                    sx, sy, sz, i, j = edge
                    if previous != i:
                        previous, local = i, 0
                    handle.write(
                        f"global index={i + 1}  local index={local} (global={j + 1}, Rn={table.rn((sx, sy, sz))})\n"
                    )
                    key, index = matrix.lookup[edge]
                    for row in matrix.pair_blocks[key][index].detach().tolist():
                        handle.write(
                            " ".join(f"{value:23.20f}" for value in row) + "\n"
                        )
                    local += 1
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return OpenMXWriteStats(
        {name: len(edges) for name in matrices},
        int("density" in matrices),
    )


def export_model_openmx(model, *, atoms, positions, lattice, output_path):
    """Infer/export using ONLY a loaded model and native Cartesian geometry."""
    from data.structure_inference import build_model_input_from_structure
    from net.common import get_torch_dtype

    device = next(model.parameters()).device
    dtype = get_torch_dtype(model.cfg.dtype)
    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    lattice = (
        None
        if lattice is None
        else torch.as_tensor(lattice, dtype=dtype, device=device)
    )
    # Match Snapshot.to_e3nn's Cartesian convention, without any reference data.
    cob = torch.eye(3, dtype=dtype, device=device)[[2, 0, 1]]
    x = build_model_input_from_structure(
        atoms=tuple(atoms),
        positions=positions @ cob,
        box=None if lattice is None else lattice @ cob,
        cfg=model.cfg,
        mapper=model.mapper,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            predictions = model.predict_matrices(x, physical=True)
    finally:
        model.train(was_training)
    return write_openmx_from_structure(
        output_path,
        predictions,
        atoms=atoms,
        positions=positions,
        lattice=lattice,
        cutoff_radius=model.cfg.cutoff_radius,
    )
