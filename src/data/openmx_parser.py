"""
openmx_parser.py
================
Parses a single OpenMX ``*.scfout`` into a **Snapshot** object that bundles
Hamiltonian H, Overlap S and Density D block-matrices.

*  Sums duplicate periodic-image blocks (different Rn)
*  Returns the matrices either in **OpenMX** or **E3NN** convention
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from core.basis_converter import OpenMXE3NNConverter
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot  # <── new aggregate container

__all__ = ["OpenMXParseError", "parse_openmx_scfout"]

# ---------------------------------------------------------------- regexes
_SECTION_RE = re.compile(
    r"^(Kohn-Sham Hamiltonian spin=0|Overlap matrix|Density matrix spin=0|Overlap matrix with position operator x)$"
)
_HEADER_RE = re.compile(
    r"global index=(\d+)\s+local index=\d+\s+\(global=(\d+),\s*Rn=([-]?\d+)\)"
)
_HEADER_OVERLAP_POS_X_RE = re.compile(
    r"global index=(\d+)\s+local index=\d+\s+\(global=(\d+),\s*Rn=([-]?\d+) ([-]?\d+) ([-]?\d+) ([-]?\d+)\)"
)

_SKIP_OVERLAP_RE = re.compile(r"^Overlap matrix with (position|momentum) operator")


# ---------------------------------------------------------------- errors
class OpenMXParseError(RuntimeError):
    """Raised when file format deviates from what the parser expects."""


# ---------------------------------------------------------------- main entry
def parse_openmx_scfout(
    path: str | Path,
    atoms: List[str] | Tuple[str, ...],
    orbital_cfg: OrbitalIrrepConfig,
    *,
    convention: str = "e3nn",  # "openmx" | "e3nn"
    symmetrize_density: bool = True,  # D ← D + Dᵀ
) -> Snapshot:
    """
    Parameters
    ----------
    path
        Path to ``*.out`` or ``*.scfout`` produced by OpenMX (single-k-point file).
    atoms
        Global atom order (list like ``["H","H","H","H","O","O"]``).
    orbital_cfg
        Same spec that is later passed to `BlockIrrepMapper`.
    convention
        "openmx"  - keep native basis;
        "e3nn"    - convert real-SH ordering to the Wikipedia / e3nn convention.
    symmetrize_density
        If *True* (default) replaces ``D`` with ``D + Dᵀ`` **after** parsing.
    """
    atoms = list(atoms)
    mapper = BlockIrrepMapper(
        orbital_cfg, diagonal=False, device="cpu", dtype=torch.float32
    )

    # ─────────────────────────────────────── storage: mat→key→(i,j)→tensor
    accum: Dict[str, Dict[str, Dict[Tuple[int, int], torch.Tensor]]] = {
        "hamiltonian": {},
        "overlap": {},
        "density": {},
    }
    rn_shift_map: Dict[int, Tuple[int, int, int]] = {}

    def _add(mat: str, key: str, i: int, j: int, rn: int, blk: torch.Tensor) -> None:
        dct = accum[mat].setdefault(key, {})
        if (i, j, rn) in dct:
            raise OpenMXParseError(
                f"Duplicate block found for {mat} {key} ({i},{j}) Rn={rn}"
            )
        dct[(i, j, rn)] = blk

    # ─────────────────────────────────────── parse loop
    current: str | None = None  # "hamiltonian" | "overlap" | "density"
    density_seen = False

    path = Path(path)
    with path.open() as fh:
        line_iter = iter(fh)
        for raw in line_iter:
            line = raw.strip()

            # ---------- section headers ------------------------------------
            if _SECTION_RE.match(line):
                if line.startswith("Density"):
                    if density_seen:  # ignore further spin-resolved density blocks
                        current = None
                        continue
                    density_seen = True
                    current = "density"
                elif line.startswith("Kohn-Sham"):
                    current = "hamiltonian"
                elif line.startswith("Overlap matrix with position operator x"):
                    current = "overlap position x"
                else:  # bare “Overlap matrix”
                    current = "overlap"
                continue
            if _SKIP_OVERLAP_RE.match(line) and current != "overlap position x":
                current = None  # discard position / momentum overlap blocks
                continue

            # ---------- inside section: individual block -------------------
            if current is None:
                continue

            if current == "overlap position x":
                m = _HEADER_OVERLAP_POS_X_RE.match(line)
                if m:
                    rn, sx, sy, sz = (
                        int(m.group(3)),
                        int(m.group(4)),
                        int(m.group(5)),
                        int(m.group(6)),
                    )
                    rn_shift_map[rn] = (sx, sy, sz)
                continue

            m = _HEADER_RE.match(line)
            if not m:
                continue

            i_glob = int(m.group(1)) - 1
            j_glob = int(m.group(2)) - 1
            rn = int(m.group(3))
            el_i, el_j = atoms[i_glob], atoms[j_glob]
            key = f"{el_i}-{el_j}"
            d_i, d_j = mapper.block_dims(key)

            rows: List[List[float]] = []
            # read d_i numerical lines
            while len(rows) < d_i:
                try:
                    data_line = next(line_iter).strip()
                except StopIteration:
                    raise OpenMXParseError("Unexpected EOF within a block")
                if not data_line:
                    continue
                nums = [float(x) for x in data_line.split()]
                if len(nums) != d_j:
                    raise OpenMXParseError(
                        f"Row len {len(nums)} does not match expected {d_j} for {key}"
                    )
                rows.append(nums)

            block = torch.tensor(rows, dtype=torch.float32)
            _add(current, key, i_glob, j_glob, rn, block)

    # ─────────────────────────────────────── build BlockMatrix objects
    def _to_block_matrix(
        sub: Dict[str, Dict[Tuple[int, int, int], torch.Tensor]],
        rn_shift_map: Dict[int, Tuple[int, int, int]],
    ) -> BlockMatrix:
        from collections import Counter

        pair_blocks: Dict[str, List[torch.Tensor]] = {}
        pair_edges: Dict[str, List[List[int]]] = {}
        lookup: Dict[Tuple[int, int], Tuple[str, int]] = {}
        for key, block_dict in sub.items():
            # order edges deterministically by (i, j, rn)
            items = sorted(block_dict.items(), key=lambda t: t[0])
            pair_blocks[key] = [b for (_ijr, b) in items]
            # pair_edges[key] = [[i, j, rn] for (i, j, rn), _ in items]
            pair_edges[key] = []
            for (i, j, rn), _ in items:
                sx, sy, sz = rn_shift_map[rn]
                pair_edges[key].append([i, j, sx, sy, sz])
            for idx, ((i, j, rn), _) in enumerate(items):
                sx, sy, sz = rn_shift_map[rn]
                lookup[(i, j, sx, sy, sz)] = (key, idx)

        pair_blocks_t = {k: torch.stack(v) for k, v in pair_blocks.items()}
        pair_edges_t = {
            k: torch.tensor(v, dtype=torch.long).t() for k, v in pair_edges.items()
        }

        # Sanity checks to make sure edges make sense
        # 1. Test whether all edges are unique
        all_edges = set()
        for edges in pair_edges_t.values():
            for edge in edges.t().tolist():
                all_edges.add(tuple(edge))
        if len(all_edges) != sum(len(edges.t()) for edges in pair_edges_t.values()):
            raise OpenMXParseError("Duplicate edges found in pair edges")
        # 2. Test whether lookup contains all edges
        if len(lookup) != len(all_edges):
            raise OpenMXParseError("Lookup size does not match edge count")
        for i, j, sx, sy, sz in all_edges:
            if (i, j, sx, sy, sz) not in lookup:
                raise OpenMXParseError(f"Edge {(i, j, sx, sy, sz)} not found in lookup")
            key, idx = lookup[(i, j, sx, sy, sz)]
            if key not in pair_blocks_t or idx >= len(pair_blocks_t[key]):
                raise OpenMXParseError(
                    f"Edge {(i, j, sx, sy, sz)} lookup points to invalid block"
                )
        # 3. Test whether all self-edges are present
        for i, atom in enumerate(atoms):
            key = f"{atom}-{atom}"
            if key not in pair_blocks_t:
                raise OpenMXParseError(f"Self-edge {key} not found in pair blocks")
            if (i, i, 0, 0, 0) not in lookup:
                raise OpenMXParseError(f"Self-edge lookup for {key} missing")
        # 4. Test whether graph is symmetric
        for i, j, sx, sy, sz in lookup:
            key, idx = lookup[(i, j, sx, sy, sz)]
            if (j, i, -sx, -sy, -sz) not in lookup:
                raise OpenMXParseError(
                    f"Edge {(i, j, sx, sy, sz)} is not symmetric with {(j, i, -sx, -sy, -sz)}"
                )

        return BlockMatrix(
            atoms=tuple(atoms),
            atom_counts=Counter(atoms),
            pair_blocks=pair_blocks_t,
            pair_edges=pair_edges_t,
            lookup=lookup,
            orbital_cfg=orbital_cfg,
            basis="openmx",
        )

    ham = _to_block_matrix(accum["hamiltonian"], rn_shift_map)
    ovl = _to_block_matrix(accum["overlap"], rn_shift_map)
    den = _to_block_matrix(accum["density"], rn_shift_map)

    if symmetrize_density:
        den = den + den.transpose()

    # ─────────────────────────────────────── optional basis conversion
    if convention == "e3nn":
        conv = OpenMXE3NNConverter(orbital_cfg)
        ham = conv.matrix_to_e3nn(ham)
        ovl = conv.matrix_to_e3nn(ovl)
        den = conv.matrix_to_e3nn(den)
        print(
            "Warning: Parsed matrices converted to e3nn convention, positions may need adjustment."
        )
    elif convention != "openmx":
        raise ValueError(f"convention must be 'openmx' or 'e3nn', not '{convention}'")

    # ─────────────────────────────────────── Snapshot aggregation
    return Snapshot(hamiltonian=ham, overlap=ovl, density=den)
