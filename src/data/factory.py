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

import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.gnn_dataset import E3GNNDataset
from data.snapshot import Snapshot
from data.openmx_info_parser import parse_info_out, InfoOutData

Purpose = Literal["train", "val"]


# ════════════════════════════════════════════════════════════════════════
class DatasetFactory:
    """Collects snapshot paths → builds datasets + **shared** mapper."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        cutoff_gnn: float = 5.0,
        cutoff_matrix: float = 7.5,
        l_max_sh: int = 3,
        n_radial: int = 64,
        keep_snapshots: bool = False,
        device: torch.device | str = "cpu",
    ):
        self.cutoff_gnn = float(cutoff_gnn)
        self.cutoff_matrix = float(cutoff_matrix)
        self.l_max_sh = int(l_max_sh)
        self.n_radial = int(n_radial)
        self.keep_snapshots = bool(keep_snapshots)
        self.device = torch.device(device)

        # paths grouped by purpose --------------------------------------
        self._pairs: Dict[Purpose, List[Tuple[Path, Path]]] = {"train": [], "val": []}

        # shallow caches -------------------------------------------------
        self._info_cache: Dict[Path, InfoOutData] = {}
        self._snap_cache: Dict[Tuple[Path, Path], Snapshot] = {}

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
        if path not in self._info_cache:
            self._info_cache[path] = parse_info_out(path)
        return self._info_cache[path]

    def _load_snapshot(self, matrix_p: Path, info_p: Path) -> Snapshot:
        key = (matrix_p, info_p)
        if key not in self._snap_cache:
            snap = Snapshot.from_openmx(
                matrix_path=matrix_p,
                info_path=info_p,
                convention="e3nn",
                symmetrize_density=True,
            )
            self._snap_cache[key] = snap
        return self._snap_cache[key]

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
        info_all = [self._load_info(p) for p in self._info_cache]  # cache may be empty
        # Ensure we also parse info for pairs not cached yet
        for _mat, info_p in self._pairs["train"] + self._pairs["val"]:
            info_all.append(self._load_info(info_p))

        # OrbitalIrrepConfig utility: union of all elements / orbitals
        orb_cfg = OrbitalIrrepConfig.from_info_list(info_all)  # type: ignore[attr-defined]

        # ②  Global mapper ---------------------------------------------
        mapper = BlockIrrepMapper(orb_cfg, diagonal=False, device=self.device)

        # ③  Load snapshots (they stay on CPU for now) ------------------
        def _collect(pairs: Sequence[Tuple[Path, Path]]) -> List[Snapshot]:
            snaps: List[Snapshot] = []
            for mat_p, info_p in pairs:
                snaps.append(self._load_snapshot(mat_p, info_p))
            return snaps

        snaps_train = _collect(self._pairs["train"])
        snaps_val = _collect(self._pairs["val"]) if self._pairs["val"] else []

        # ④  Build datasets --------------------------------------------
        ds_kwargs = dict(
            mapper=mapper,
            cutoff_gnn=self.cutoff_gnn,
            cutoff_matrix=self.cutoff_matrix,
            l_max_sh=self.l_max_sh,
            n_radial=self.n_radial,
            device=self.device,
        )
        train_ds = E3GNNDataset(snaps_train, **ds_kwargs)
        val_ds = (
            E3GNNDataset(snaps_val, **ds_kwargs) if snaps_val else None  # type: ignore[arg-type]
        )

        return train_ds, val_ds, mapper


# ════════════════════════════════════════════════════════════════════════
# Convenience *function* mirroring the old train.py helper
# ════════════════════════════════════════════════════════════════════════
def build_datasets(
    train_pairs: Sequence[Tuple[str | os.PathLike, str | os.PathLike]],
    val_pairs: Sequence[Tuple[str | os.PathLike, str | os.PathLike]] | None = None,
    **factory_kwargs,
) -> Tuple[E3GNNDataset, Optional[E3GNNDataset], BlockIrrepMapper]:
    """
    One-shot helper – mirrors the OO factory but keeps the old functional look
    used in *train.py*.

    Example
    -------
    >>> train_ds, val_ds, mapper = build_datasets(
    ...     train_pairs=[("run1.scfout", "run1.info")],
    ...     cutoff_gnn=4.5,
    ... )
    """
    fac = DatasetFactory(**factory_kwargs)
    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    if val_pairs:
        for m, i in val_pairs:
            fac.add_snapshot(m, i, purpose="val")
    return fac.create()
