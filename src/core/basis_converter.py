"""
basis_converter.py
==================

Tools to convert matrix blocks between **OpenMX real‑SH ordering** and the
ordering expected by **E3NN** (Wikipedia real SH ordering, l≤3).

The conversion is *purely a permutation plus possible sign flip*, so it is
implemented with constant orthogonal matrices Uₗ (one per ℓ).

Usage
-----
>>> conv = OpenMXE3NNConverter(orbital_cfg)   # knows each element's orbitals
>>> snap_e3 = conv.snapshot_to_e3nn(snap_openmx)
>>> snap_back = conv.snapshot_to_openmx(snap_e3)
"""

from __future__ import annotations

from typing import Dict, List

import torch
from e3nn.o3 import Irreps

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.snapshot_block import MatrixBlockData

__all__ = ["OpenMXE3NNConverter"]


# --------------------------------------------------------------------- U matrices
_U_OPENMX_TO_WIKI: Dict[int, torch.Tensor] = {
    0: torch.eye(1, dtype=torch.float32),
    1: torch.eye(3, dtype=torch.float32)[[1, 2, 0]],
    2: torch.eye(5, dtype=torch.float32)[[2, 4, 0, 3, 1]],
    3: torch.eye(7, dtype=torch.float32)[[6, 4, 2, 0, 1, 3, 5]],
}
_U_WIKI_TO_OPENMX = {l: U.T for l, U in _U_OPENMX_TO_WIKI.items()}


def _orbital_types_from_irreps(irreps: Irreps) -> List[int]:
    """Expand [(mul,l)] → [l,l,...] multiplicity times."""
    types: List[int] = []
    for mul, (l, _p) in irreps:
        types.extend([l] * mul)
    return types


# --------------------------------------------------------------------- converter
class OpenMXE3NNConverter:
    """
    Converts *all blocks in a snapshot* between ``basis="openmx"`` and
    ``basis="e3nn"``.  Requires the *same* `OrbitalIrrepConfig` used to build
    the snapshot's BlockIrrepMapper (so it knows orbital ordering per element).
    """

    def __init__(self, orbital_cfg: OrbitalIrrepConfig, device="cpu"):
        self.cfg = orbital_cfg
        self.device = torch.device(device)

        # cache block‑diag matrices per element symbol
        self._U_openmx2wiki: Dict[str, torch.Tensor] = {}
        self._U_wiki2openmx: Dict[str, torch.Tensor] = {}
        for el in self.cfg.elements():
            typs = _orbital_types_from_irreps(self.cfg.element_to_irreps[el])
            mats_U = [_U_OPENMX_TO_WIKI[l] for l in typs]
            mats_V = [_U_WIKI_TO_OPENMX[l] for l in typs]
            self._U_openmx2wiki[el] = torch.block_diag(*mats_U).to(self.device)
            self._U_wiki2openmx[el] = torch.block_diag(*mats_V).to(self.device)

    # ---------------- block‑level helpers -----------------------------------
    def block_openmx_to_e3nn(self, key: str, block: torch.Tensor) -> torch.Tensor:
        el_i, el_j = key.split("-")
        U_i = self._U_openmx2wiki[el_i]
        U_j = self._U_openmx2wiki[el_j]
        return U_i @ block @ U_j.T

    def block_e3nn_to_openmx(self, key: str, block: torch.Tensor) -> torch.Tensor:
        el_i, el_j = key.split("-")
        V_i = self._U_wiki2openmx[el_i]
        V_j = self._U_wiki2openmx[el_j]
        return V_i @ block @ V_j.T

    # ---------------- snapshot‑level helpers --------------------------------
    def snapshot_to_e3nn(self, matrix: MatrixBlockData) -> MatrixBlockData:
        if matrix.basis == "e3nn":
            return matrix
        new_blocks = {
            k: self.block_openmx_to_e3nn(k, blk)
            for k, blk in matrix.pair_blocks.items()
        }
        return matrix._replace_pair_blocks(new_blocks, basis="e3nn")

    def snapshot_to_openmx(self, matrix: MatrixBlockData) -> MatrixBlockData:
        if matrix.basis == "openmx":
            return matrix
        new_blocks = {
            k: self.block_e3nn_to_openmx(k, blk)
            for k, blk in matrix.pair_blocks.items()
        }
        return matrix._replace_pair_blocks(new_blocks, basis="openmx")
