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
>>> fac = DatasetFactory(cutoff_radius=7.0, device="cpu")
>>> fac.add_snapshot("run1-H2O.scfout", "run1-H2O.info.out", purpose="train")
>>> fac.add_snapshot("run2-CO2.scfout", "run2-CO2.info.out", purpose="val")
>>> train_ds, val_ds, mapper = fac.create()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Literal, Optional
from types import SimpleNamespace


from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.gnn_dataset import E3GNNDataset
from data.openmx_info_parser import parse_info_out, InfoOutData
from data.pyscf_baseline_parser import load_pyscf_metadata
from data.deeph_e3_parser import is_deeph_e3_snapshot, load_deeph_e3_metadata
from net.common import Config
from utils.summary import print_dataset_summary

Purpose = Literal["train", "val", "test"]


# ════════════════════════════════════════════════════════════════════════
class DatasetFactory:
    """Collects snapshot paths -> builds datasets + **shared** mapper."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        cfg: Config,
        convention: str = "e3nn",
    ):
        self.cfg = cfg
        self.convention = convention

        # paths grouped by purpose --------------------------------------
        self._pairs: Dict[Purpose, List[Tuple[Path, Path]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

    # ------------------------------------------------------------------ public API
    def add_snapshot(
        self,
        matrix_path: str | os.PathLike,
        info_path: str | os.PathLike,
        purpose: Purpose = "train",
    ) -> None:
        """
        Register one OpenMX **(matrix, info)** file pair.

        Parameters
        ----------
        matrix_path
            Path to the ``*.scfout`` file with H/S/D blocks.
        info_path
            Path to the corresponding ``*.info`` / ``*.out`` file.
        purpose
            One of ``"train"`` (default), ``"val"``, or ``"test"``.
        """
        if purpose not in ("train", "val", "test"):
            raise ValueError("purpose must be 'train', 'val', or 'test'")
        pair = (Path(matrix_path), Path(info_path))
        self._pairs[purpose].append(pair)

    # ------------------------------------------------------------------ helpers
    def _load_info(self, matrix_path: Path, path: Path) -> InfoOutData:
        if is_deeph_e3_snapshot(matrix_path, path):
            meta = load_deeph_e3_metadata(path.parent, dtype=self.cfg.dtype)
            return SimpleNamespace(orbital_set=meta.orbital_set)
        if path.suffix == ".json":
            meta = load_pyscf_metadata(path)
            return SimpleNamespace(orbital_set=meta.orbital_set)
        return parse_info_out(path)

    # ------------------------------------------------------------------ create
    def create(self, *, include_test: bool = False):
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
        for matrix_path, info_p in (
            self._pairs["train"] + self._pairs["val"] + self._pairs["test"]
        ):
            info_all.append(self._load_info(matrix_path, info_p))

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
        test_ds = (
            E3GNNDataset(self._pairs["test"], mapper, self.cfg, self.convention)
            if self._pairs["test"]
            else None
        )

        # Print dataset summaries if verbosity >= 1
        print_dataset_summary(
            train_ds, name="Training Dataset", verbosity=self.cfg.verbosity
        )
        if val_ds:
            print_dataset_summary(
                val_ds, name="Validation Dataset", verbosity=self.cfg.verbosity
            )
        if test_ds:
            print_dataset_summary(
                test_ds, name="Test Dataset", verbosity=self.cfg.verbosity
            )

        if include_test:
            return train_ds, val_ds, test_ds, mapper
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
    ...     cfg=Config(cutoff_radius=7.0),
    ... )
    """
    fac = DatasetFactory(cfg, convention)
    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    if val_pairs:
        for m, i in val_pairs:
            fac.add_snapshot(m, i, purpose="val")
    return fac.create()
