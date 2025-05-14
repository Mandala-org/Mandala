"""
openmx_parser.py
================

Single-file parser that converts one OpenMX ``*.scfout`` file into
``MatrixBlockData`` objects for **Hamiltonian**, **Overlap**, and
**Density** matrices.

Highlights
----------
* Handles junk header lines automatically - scans until the first recognised
  section header.
* Parses *every* block header of the form::

      global index=I  local index=L (global=J, Rn=R)

  interpreting **I** as *row atom*, **J** as *column atom* (1-based).
* Optional `pbc_sum=True` collapses all duplicate blocks with different Rn
  (periodic images) by simple addition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from core.basis_converter import OpenMXE3NNConverter
from data.snapshot_block import MatrixBlockData

__all__ = ["OpenMXParseError", "parse_openmx_scfout"]

_HEADER_RE = re.compile(
    r"global index=(\d+)\s+local index=\d+\s+\(global=(\d+),\s*Rn=([-]?\d+)\)"
)


class OpenMXParseError(RuntimeError):
    """Raised when SCFOUT format is not as expected."""


# ─────────────────────────────────────────────────────────────────────────────
def parse_openmx_scfout(
    path: str | Path,
    atoms: List[str] | Tuple[str, ...],
    orbital_cfg: OrbitalIrrepConfig,
    convention: str = "e3nn",
    symmetrize_density: bool = True,
) -> Dict[str, MatrixBlockData]:
    """
    Parse a single OpenMX ``*.scfout`` file and return snapshots for the three
    matrices of interest.  Blocks with identical global (i,j) indices are
    always **summed**, regardless of Rn or local index.
    """
    atoms = list(atoms)
    mapper = BlockIrrepMapper(orbital_cfg, diagonal=False, device="cpu")

    # ---------------------------------------------------------------- section headers
    wanted_headers = {
        "Kohn-Sham Hamiltonian spin=0": "hamiltonian",
        "Overlap matrix": "overlap",
        "Density matrix spin=0": "density",  # only FIRST one kept
    }
    density_taken = False

    # storage:  matrix_type -> key -> (i,j) -> accumulated tensor
    blocks_sum: Dict[str, Dict[str, Dict[Tuple[int, int], torch.Tensor]]] = {
        "hamiltonian": {},
        "overlap": {},
        "density": {},
    }

    def _add_block(mat: str, key: str, i: int, j: int, blk: torch.Tensor):
        mat_dict = blocks_sum[mat].setdefault(key, {})
        if (i, j) in mat_dict:
            mat_dict[(i, j)] += blk
        else:
            mat_dict[(i, j)] = blk.clone()

    # ---------------------------------------------------------------- parse file
    path = Path(path)
    with path.open() as fh:
        line_iter = iter(fh)

        current_mat: str | None = None
        for raw in line_iter:
            line = raw.strip()

            # --- section header ------------------------------------------------
            if line in wanted_headers:
                # density spin=0 only first time
                if line == "Density matrix spin=0":
                    if density_taken:
                        current_mat = None
                        continue
                    density_taken = True
                current_mat = wanted_headers[line]
                continue

            # ignore any other header beginning with Overlap matrix with...
            if line.startswith("Overlap matrix with"):
                current_mat = None
                continue

            # --- block header within a wanted section -------------------------
            if current_mat is None:
                continue
            head_match = _HEADER_RE.match(line)
            if not head_match:
                continue

            i_glob = int(head_match.group(1)) - 1  # 0‑based
            j_glob = int(head_match.group(2)) - 1
            el_i, el_j = atoms[i_glob], atoms[j_glob]
            key = f"{el_i}-{el_j}"
            d_i, d_j = mapper.block_dims(key)

            # read the next d_i lines of numbers
            rows: List[List[float]] = []
            while len(rows) < d_i:
                try:
                    num_line = next(line_iter).strip()
                except StopIteration:
                    raise OpenMXParseError("Unexpected EOF inside block")
                if not num_line:
                    continue  # skip blank lines inside strange outputs
                nums = [float(x) for x in num_line.split()]
                if len(nums) != d_j:
                    raise OpenMXParseError(
                        f"Row length {len(nums)} != expected {d_j} for {key}"
                    )
                rows.append(nums)

            block_tensor = torch.tensor(rows, dtype=torch.float32)
            _add_block(current_mat, key, i_glob, j_glob, block_tensor)

    # ---------------------------------------------------------------- build MatrixBlockData
    out: Dict[str, MatrixBlockData] = {}
    for mat, per_key in blocks_sum.items():
        if not per_key:  # matrix absent
            continue
        pair_blocks: Dict[str, List[torch.Tensor]] = {}
        pair_edges: Dict[str, List[List[int]]] = {}
        lookup: Dict[Tuple[int, int], Tuple[str, int]] = {}

        for key, pair_dict in per_key.items():
            # deterministic ordering of edges
            items = sorted(pair_dict.items(), key=lambda t: t[0])  # sort by (i,j)
            pair_blocks[key] = [blk for (_ij, blk) in items]
            pair_edges[key] = [[i, j] for (i, j), _blk in items]
            for idx, ((i, j), _blk) in enumerate(items):
                lookup[(i, j)] = (key, idx)

        # stack tensors
        pair_blocks_t = {k: torch.stack(v) for k, v in pair_blocks.items()}
        pair_edges_t = {
            k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
        }

        matrix = MatrixBlockData(
            atoms=tuple(atoms),
            pair_blocks=pair_blocks_t,
            pair_edges=pair_edges_t,
            lookup=lookup,
            mapper=mapper,
        )
        out[mat] = matrix
    if symmetrize_density:
        out["density"] = out["density"] + out["density"].transpose()

    if convention == "e3nn":
        # convert to e3nn basis
        converter = OpenMXE3NNConverter(orbital_cfg)
        for key, matrix in out.items():
            out[key] = converter.snapshot_to_e3nn(matrix)
    elif convention == "openmx":
        pass
    else:
        raise ValueError(f"Unknown convention '{convention}'")

    return out
