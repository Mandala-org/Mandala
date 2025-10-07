# ────────────────────────────────────────────────────────────────────────────
# data/factory.py
# ────────────────────────────────────────────────────────────────────────────
"""
Factory helpers that build **one global BlockIrrepMapper** together with
train / validation :class:`E3GNNDataset` instances.

Typical usage
-------------
>>> from data.factory import DatasetFactory
>>>
>>> fac = DatasetFactory(cutoff_gnn=4.5, cutoff_matrix=7.0, device="cpu")
>>> fac.add_snapshot("run1-H2O.scfout", "run1-H2O.info.out", purpose="train")
>>> fac.add_snapshot("run2-CO2.scfout", "run2-CO2.info.out", purpose="val")
>>> train_ds, val_ds, mapper = fac.create()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Literal, Optional


from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.gnn_dataset import E3GNNDataset
from data.openmx_info_parser import parse_info_out, InfoOutData
from net.common import Config

Purpose = Literal["train", "val"]


# ════════════════════════════════════════════════════════════════════════
class DatasetFactory:
    """Collects snapshot paths → builds datasets + **shared** mapper."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        cfg: Config,
        convention: str = "e3nn",
    ):
        self.cfg = cfg
        self.convention = convention
        # cache root for processed snapshots; if None, caching is disabled
        if self.cfg.cache_root is not None:
            self.cfg.cache_root = Path(self.cfg.cache_root).expanduser()

        # paths grouped by purpose --------------------------------------
        self._pairs: Dict[Purpose, List[Tuple[Path, Path]]] = {"train": [], "val": []}

    # ------------------------------------------------------------------ public API
    def add_snapshot(
        self,
        matrix_path: str | os.PathLike,
        info_path: str | os.PathLike,
        purpose: Purpose = "train",
    ) -> None:
        """
        Register one **(matrix, info)** file pair.

        Parameters
        ----------
        matrix_path
            Path to the ``*.scfout`` file with H/S/D blocks.
        info_path
            Path to the corresponding ``*.info`` / ``*.out`` file.
        purpose
            Either ``"train"`` (default) or ``"val"``.
        """
        if purpose not in ("train", "val"):
            raise ValueError("purpose must be 'train' or 'val'")
        pair = (Path(matrix_path), Path(info_path))
        self._pairs[purpose].append(pair)

    # ------------------------------------------------------------------ helpers
    def _load_info(self, path: Path) -> InfoOutData:
        return parse_info_out(path)

    # ------------------------------------------------------------------ create
    def create(
        self,
    ) -> Tuple[E3GNNDataset, Optional[E3GNNDataset], BlockIrrepMapper]:
        """
        Build datasets **and** a shared mapper.

        Returns
        -------
        train_ds
            :class:`E3GNNDataset` for training (batch size is handled elsewhere).
        val_ds
            Validation dataset, or *None* if no validation snapshots were added.
        mapper
            The single :class:`BlockIrrepMapper` used by both datasets.
        """
        # ①  Merge all orbital configs ---------------------------------
        info_all = []
        for _mat, info_p in self._pairs["train"] + self._pairs["val"]:
            info_all.append(self._load_info(info_p))

        # OrbitalIrrepConfig utility: union of all elements / orbitals
        orb_cfg = OrbitalIrrepConfig.from_info_list(info_all)  # type: ignore[attr-defined]

        # ②  Global mapper ---------------------------------------------
        mapper = BlockIrrepMapper(
            orb_cfg,
            diagonal=False,
            device="cpu",
            dtype=self.cfg.dtype,
        )

        # ④  Build datasets --------------------------------------------
        train_ds = E3GNNDataset(self._pairs["train"], mapper, self.cfg, self.convention)
        val_ds = (
            E3GNNDataset(self._pairs["val"], mapper, self.cfg, self.convention)
            if self._pairs["val"]
            else None
        )

        return train_ds, val_ds, mapper


# ════════════════════════════════════════════════════════════════════════
# Convenience *function* building the train/val datasets
# ════════════════════════════════════════════════════════════════════════
def build_datasets(
    train_pairs: Sequence[Tuple[str | os.PathLike, str | os.PathLike]],
    val_pairs: Sequence[Tuple[str | os.PathLike, str | os.PathLike]] | None = None,
    cfg: Config = None,
    convention: str = "e3nn",
) -> Tuple[E3GNNDataset, Optional[E3GNNDataset], BlockIrrepMapper]:
    """Build datasets from snapshot pairs.
    Example
    -------
    >>> train_ds, val_ds, mapper = build_datasets(
    ...     train_pairs=[("run1.scfout", "run1.info")],
    ...     cfg=Config(cutoff_gnn=4.5),
    ... )
    """
    fac = DatasetFactory(cfg, convention)
    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    if val_pairs:
        for m, i in val_pairs:
            fac.add_snapshot(m, i, purpose="val")
    return fac.create()
