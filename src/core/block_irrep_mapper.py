"""
block_irrep_mapper.py
=====================

Utility that *owns all* change-of-basis matrices (Q) required to map
**sparse matrix blocks  ↔  irrep vectors** for every atom-pair that
appears in a training dataset.

It is a thin, dependency-free layer on top of
``e3nn.o3.ReducedTensorProducts`` and is reused by the data-loader,
the model heads and the evaluation code.

Example
-------
>>> from core.orbital_irrep_config import OrbitalIrrepConfig
>>> cfg = OrbitalIrrepConfig.from_dict({"H": ["1x0e"], "Si": ["2x0e","2x1o","1x2e"]})
>>> mapper = BlockIrrepMapper(cfg)
>>> vec = mapper.blocks_to_vectors(("Si","Si"), torch.randn(4,13,13))
>>> rec = mapper.vectors_to_blocks(("Si","Si"), vec)
>>> torch.allclose(rec, rec.transpose(-1,-2))   # symmetry not enforced here
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This import is only for mypy/flake8/etc., not at runtime
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from data.block_matrix import BlockMatrix, IrrepsBlockData


from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from e3nn.o3 import Irreps, ReducedTensorProducts


class MappingKeyError(KeyError):
    """Raised when an unknown (element_A, element_B) key is requested."""


@dataclass(slots=True)
class _IrrepToMatrix:
    rtp: ReducedTensorProducts
    dim_i: int
    dim_j: int
    n_vec: int

    @classmethod
    def from_irreps(
        cls,
        irreps_i: Irreps,
        irreps_j: Irreps,
        diagonal: bool,
    ) -> Tuple["_IrrepToMatrix", torch.Tensor]:
        assert diagonal is False, "diagonal=True not supported"
        formula = "ij=ji" if diagonal else "ij"
        rtp = ReducedTensorProducts(formula, i=irreps_i, j=irreps_j)
        q = rtp.change_of_basis.flatten(-2)

        itm = cls(
            rtp=rtp,
            dim_i=irreps_i.dim,
            dim_j=irreps_j.dim,
            n_vec=q.shape[0],
        )
        return itm, q

    # ------------------------- mapping helpers ------------------------------ #
    def blocks_to_vectors(self, blocks: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Map ``(..., d_i, d_j)`` blocks → ``(..., n_vec)`` irrep vectors.
        """
        flat = blocks.flatten(-2)  # (..., d_i*d_j)
        return flat @ q.T

    def vectors_to_blocks(self, vectors: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Inverse of :meth:`blocks_to_vectors`.
        """
        flat = vectors @ q  # (..., d_i*d_j)
        return flat.view(*vectors.shape[:-1], self.dim_i, self.dim_j)


# --------------------------------------------------------------------------- #
class BlockIrrepMapper(nn.Module):
    """
    Manages *all* per-pair `_IrrepToMatrix` instances.

    Parameters
    ----------
    orbital_cfg
        Instance built from user YAML / dict.
    diagonal
        If **True** and *only* for *same* element pairs, use ``"ij=ji"``
        (symmetric) reduced tensor product.
    device
        Where the Q matrices live (CPU by default - they are tiny).
    """

    def __init__(
        self,
        orbital_cfg: "OrbitalIrrepConfig",
        *,
        diagonal: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert diagonal is False, "diagonal=True not supported"

        from core.orbital_irrep_config import (
            OrbitalIrrepConfig,
        )  # local import to avoid cycle

        if not isinstance(orbital_cfg, OrbitalIrrepConfig):
            raise TypeError("orbital_cfg must be an OrbitalIrrepConfig")

        self.orbital_cfg = orbital_cfg
        self.diagonal = diagonal

        self._maps: Dict[Tuple[str, str], _IrrepToMatrix] = {}

        # build for **ordered** pairs (A,B) appearing in Cartesian product
        els = orbital_cfg.elements()
        for el_a in els:
            for el_b in els:
                irreps_a = orbital_cfg.element_to_irreps[el_a]
                irreps_b = orbital_cfg.element_to_irreps[el_b]
                itm, q = _IrrepToMatrix.from_irreps(
                    irreps_a, irreps_b, diagonal and el_a == el_b
                )
                self._maps[(el_a, el_b)] = itm
                # Sanitize key for buffer name
                buffer_name = f"q_{el_a}_{el_b}"
                self.register_buffer(buffer_name, q.to(device=device, dtype=dtype))

    def _get_q(self, pair: Tuple[str, str]) -> torch.Tensor:
        """Retrieve the q tensor buffer for a given pair."""
        buffer_name = f"q_{pair[0]}_{pair[1]}"
        return getattr(self, buffer_name)

    # ------------------------- public API ----------------------------------- #
    def blocks_to_vectors(
        self,
        pair: Tuple[str, str] | str,
        blocks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert matrix blocks **for a given atom pair** to irrep vectors.

        ``pair`` can be ``("Si","H")`` or `"Si-H"` (string with hyphen).
        """
        key = self._canonical_pair(pair)
        itm = self._maps[key]
        q = self._get_q(key)
        return itm.blocks_to_vectors(blocks, q)

    def vectors_to_blocks(
        self,
        pair: Tuple[str, str] | str,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse mapping vector → block.
        """
        key = self._canonical_pair(pair)
        itm = self._maps[key]
        q = self._get_q(key)
        return itm.vectors_to_blocks(vectors, q)

    # ------------------------- meta-info ------------------------------------ #
    def vector_dim(self, pair: Tuple[str, str] | str) -> int:
        """Number of irrep coefficients for *one* block of `pair`."""
        return self._lookup(pair).n_vec

    def block_dims(self, pair: Tuple[str, str] | str) -> Tuple[int, int]:
        """Underlying dimensions (d_i, d_j) of the orbital block."""
        itm = self._lookup(pair)
        return (itm.dim_i, itm.dim_j)

    def get_pair_irreps(self, pair: Tuple[str, str] | str) -> Irreps:
        """Return the output Irreps for a given pair."""
        itm = self._lookup(pair)
        return itm.rtp.irreps_out

    # ------------------------- internals ------------------------------------ #
    def _canonical_pair(self, pair: Tuple[str, str] | str) -> Tuple[str, str]:
        if isinstance(pair, str):
            if "-" not in pair:
                raise MappingKeyError("String key must look like 'Si-H'")
            a, b = pair.split("-", 1)
            return (a.strip(), b.strip())
        if isinstance(pair, tuple) and len(pair) == 2:
            return pair  # type: ignore[return-value]
        raise MappingKeyError("pair must be tuple(str,str) or 'A-B' string")

    def _lookup(self, pair: Tuple[str, str] | str) -> _IrrepToMatrix:
        key = self._canonical_pair(pair)
        try:
            return self._maps[key]
        except KeyError as exc:
            raise MappingKeyError(f"Unknown pair key {key}") from exc

    # ------------------ snapshot helpers ------------------------------------ #
    def snapshot_blocks_to_vectors(self, snap: BlockMatrix) -> IrrepsBlockData:
        return snap.to_vectors()  # delegates to dataclass

    def snapshot_vectors_to_blocks(self, snap: IrrepsBlockData) -> BlockMatrix:
        return snap.to_blocks()
