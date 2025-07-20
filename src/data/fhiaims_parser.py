"""
fhiaims_parser.py
=================
Parses FHI-AIMS output files into a **Snapshot** object.

- `geometry.in`: Contains atomic positions and lattice vectors.
- `basis-indices.out`: Defines the orbital basis set.
- `*.csc`: Sparse matrix files for Hamiltonian, Overlap, and Density.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot


def read_geometry(
    file_path: str | Path,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Parses a `geometry.in` file."""
    box = []
    atoms = []
    pos_frac = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                box.append([float(x) for x in parts[1:]])
            if parts[0] == "atom_frac":
                atoms.append(parts[-1])
                pos_frac.append([float(x) for x in parts[1:4]])
    box = torch.tensor(box, dtype=torch.float32)
    pos_frac = torch.tensor(pos_frac, dtype=torch.float32)
    pos = torch.matmul(pos_frac, box)
    return atoms, pos, box


def read_elsi_to_csc(filename: str | Path) -> sp.csc_matrix:
    """Reads an ELSI CSC format file."""
    with open(filename, "rb") as mat:
        data = mat.read()
    i8 = "l"
    i4 = "i"

    start = 0
    end = 128
    header = struct.unpack(i8 * 16, data[start:end])

    n_basis = header[3]
    nnz = header[5]

    start = end
    end = start + n_basis * 8
    col_ptr = struct.unpack(i8 * n_basis, data[start:end])
    col_ptr += (nnz + 1,)
    col_ptr = np.array(col_ptr)

    start = end
    end = start + nnz * 4
    row_idx = struct.unpack(i4 * nnz, data[start:end])
    row_idx = np.array(row_idx)

    start = end
    if header[2] == 0:
        end = start + nnz * 8
        nnz_val = struct.unpack("d" * nnz, data[start:end])
    else:
        end = start + nnz * 16
        nnz_val = struct.unpack("d" * nnz * 2, data[start:end])
        nnz_val_real = np.array(nnz_val[0::2])
        nnz_val_imag = np.array(nnz_val[1::2])
        nnz_val = nnz_val_real + 1j * nnz_val_imag

    nnz_val = np.array(nnz_val)

    for i_val in range(nnz):
        row_idx[i_val] -= 1
    for i_col in range(n_basis + 1):
        col_ptr[i_col] -= 1

    return sp.csc_matrix((nnz_val, row_idx, col_ptr), shape=(n_basis, n_basis))


def parse_basis_indices(file_path: str | Path, atoms: List[str]) -> OrbitalIrrepConfig:
    """Parses a `basis-indices.out` file."""
    orbital_counts = {}
    atom_to_atom_id = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#") or line.startswith("  fn."):
                continue
            parts = line.split()
            if not parts:
                continue
            _, _, atom_id, _, l, _ = parts
            atom_id = int(atom_id) - 1
            l = int(l)
            atom = atoms[atom_id]
            if atom not in atom_to_atom_id:
                atom_to_atom_id[atom] = atom_id
            if atom_to_atom_id[atom] == atom_id:
                orbital_counts[(atom, l)] = orbital_counts.get((atom, l), 0) + 1

    orbital_set_counts = {}
    for (atom, l), count in orbital_counts.items():
        orbital_set_counts[(atom, l)] = count // (2 * l + 1)

    orbital_dict = {}
    for (atom, l), count in orbital_set_counts.items():
        if atom not in orbital_dict:
            orbital_dict[atom] = []
        orbital_dict[atom].append(f"{count}x{l}{'e' if l % 2 == 0 else 'o'}")

    return OrbitalIrrepConfig.from_dict(orbital_dict)


def parse_fhiaims_output(
    geometry_path: str | Path,
    basis_path: str | Path,
    hamiltonian_path: str | Path,
    overlap_path: str | Path,
    density_path: str | Path,
) -> Snapshot:
    """Orchestrates the parsing of FHI-AIMS output files."""
    atoms, pos, box = read_geometry(geometry_path)
    orbital_cfg = parse_basis_indices(basis_path, atoms)

    ham_csc = read_elsi_to_csc(hamiltonian_path)
    ovl_csc = read_elsi_to_csc(overlap_path)
    den_csc = read_elsi_to_csc(density_path)

    ham = BlockMatrix.from_dense(
        torch.tensor(ham_csc.todense(), dtype=torch.float32),
        orbital_cfg=orbital_cfg,
        atoms=atoms,
        basis="fhi-aims",
    )
    ovl = BlockMatrix.from_dense(
        torch.tensor(ovl_csc.todense(), dtype=torch.float32),
        orbital_cfg=orbital_cfg,
        atoms=atoms,
        basis="fhi-aims",
    )
    den = BlockMatrix.from_dense(
        torch.tensor(den_csc.todense(), dtype=torch.float32),
        orbital_cfg=orbital_cfg,
        atoms=atoms,
        basis="fhi-aims",
    )

    return Snapshot(
        hamiltonian=ham,
        overlap=ovl,
        density=den,
        positions=pos,
        box=box,
    )
