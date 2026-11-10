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

import torch

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Literal, Optional


from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.gnn_dataset import E3GNNDataset
from data.sources import resolve_snapshot_source, load_source_metadata
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
        convention: str | None = None,
        preprocessing=None,
    ):
        self.cfg = cfg
        representation = cfg.data.get("representation", "orbital_blocks")
        if representation not in {"orbital_blocks", "packed_shells"}:
            raise ValueError(
                "data.representation must be orbital_blocks or packed_shells"
            )
        self.convention = convention or cfg.data.get(
            "basis", "openmx" if representation == "packed_shells" else "e3nn"
        )
        cfg.data = dict(cfg.data, basis=self.convention)
        if preprocessing is None and representation == "packed_shells":
            from data.preprocessing import PackedOperatorPreprocessing

            preprocessing = PackedOperatorPreprocessing(cfg.to_mapping())
        self.preprocessing = preprocessing

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
        info_path: str | os.PathLike | None = None,
        purpose: Purpose = "train",
    ) -> None:
        """
        Register a snapshot directory or an explicit **(matrix, info)** pair.

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
        source = resolve_snapshot_source(
            matrix_path, info_path, matrix_files=self.cfg.data.get("matrix_files")
        )
        pair = (source.matrix_path, source.info_path)
        self._pairs[purpose].append(pair)

    # ------------------------------------------------------------------ helpers
    def _load_info(self, matrix_path: Path, path: Path):
        return load_source_metadata(
            resolve_snapshot_source(matrix_path, path),
            dtype=self.cfg.dtype,
            options=self.cfg.data,
        )

    # ------------------------------------------------------------------ create
    def create(
        self, *, include_test: bool = False, load_splits: Sequence[Purpose] | None = None
    ):
        """
        Build datasets **and** a shared mapper.

        ``load_splits`` optionally limits matrix loading; omitted splits return
        None. All metadata still contributes to the original mapper checks.

        Returns
        -------
        train_ds
            :class:`E3GNNDataset` for training (batch size is handled elsewhere).
        val_ds
            Validation dataset, or *None* if no validation snapshots were added.
        mapper
            The single :class:`BlockIrrepMapper` used by both datasets.
        """
        # Metadata from every split is still checked, even when evaluation
        # requests only validation matrices. In particular the mapper and resolved
        # configuration must retain the complete training orbital vocabulary.
        load_splits = {"train", "val", "test"} if load_splits is None else set(load_splits)
        if not load_splits <= {"train", "val", "test"}:
            raise ValueError("load_splits must contain only train, val, or test")

        # ①  Merge all orbital configs ---------------------------------
        info_all = []
        for matrix_path, info_p in self._pairs["train"]:
            info_all.append(self._load_info(matrix_path, info_p))
        if not info_all:
            raise ValueError("A dataset factory requires training metadata")

        declared = self.cfg.species.get("shells")
        orb_cfg = (
            OrbitalIrrepConfig.from_dict(
                {
                    species: [
                        f"1x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in shells
                    ]
                    for species, shells in declared.items()
                }
            )
            if declared
            else OrbitalIrrepConfig.from_info_list(info_all)
        )
        spinful = bool(getattr(info_all[0], "is_spinful", False))
        for split, pairs in self._pairs.items():
            for matrix_path, info_path in pairs:
                metadata = self._load_info(matrix_path, info_path)
                if bool(getattr(metadata, "is_spinful", False)) != spinful:
                    raise ValueError(f"{split} spin layout differs from training")
                vocabulary = OrbitalIrrepConfig.from_info_list([metadata])
                for species, irreps in vocabulary.element_to_irreps.items():
                    if (
                        species not in orb_cfg.element_to_irreps
                        or orb_cfg.element_to_irreps[species] != irreps
                    ):
                        raise ValueError(
                            f"{split} orbital vocabulary differs from declared/training vocabulary for {species}"
                        )

        # ②  Global mapper ---------------------------------------------
        mapper = BlockIrrepMapper(
            orb_cfg,
            diagonal=False,
            device="cpu",
            dtype=self.cfg.dtype,
        )

        # Resolve one common configuration against declared/training metadata.
        shells = {
            species: tuple(ir.l for mul, ir in irreps for _ in range(mul))
            for species, irreps in mapper.orbital_cfg.element_to_irreps.items()
        }
        with torch.random.fork_rng(devices=[]):
            self.configuration, self.operator_layout = Config.resolve(
                self.cfg.to_mapping(), species_shells=shells, spinful=spinful
            )
        resolved = Config.from_mapping(self.configuration)
        for name in Config.__dataclass_fields__:
            setattr(self.cfg, name, getattr(resolved, name))
        if self.preprocessing is not None:
            self.preprocessing.prepare(
                mapper,
                info_all,
                configuration=self.configuration,
                layout=self.operator_layout,
            )

        def build(split):
            return E3GNNDataset(
                self._pairs[split],
                mapper,
                self.cfg,
                self.convention,
                preprocessing=self.preprocessing,
                split=split,
            )

        train_ds = build("train") if "train" in load_splits else None
        val_ds = build("val") if "val" in load_splits and self._pairs["val"] else None
        test_ds = build("test") if "test" in load_splits and self._pairs["test"] else None

        # Print dataset summaries if verbosity >= 1
        if train_ds is not None:
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
