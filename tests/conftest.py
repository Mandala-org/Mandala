import pytest
from pathlib import Path
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout
from data.snapshot import Snapshot
from data.factory import DatasetFactory
from net.common import Config


@pytest.fixture(scope="session")
def h2o_orbital_cfg():
    return OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})


@pytest.fixture(scope="session")
def h2o_snapshot(h2o_orbital_cfg):
    sample = Path("./data/small/H2O/original/H2O.matrix")
    atoms = list("HHHHOO")
    cfg = Config(cutoff_matrix=5.0)
    return parse_openmx_scfout(
        sample, atoms, h2o_orbital_cfg, convention="openmx", cfg=cfg
    )


@pytest.fixture(scope="session")
def h2o_snapshot_e3nn(h2o_orbital_cfg):
    sample = Path("./data/small/H2O/original/H2O.matrix")
    atoms = list("HHHHOO")
    cfg = Config(cutoff_matrix=5.0)
    return parse_openmx_scfout(
        sample, atoms, h2o_orbital_cfg, convention="e3nn", cfg=cfg
    )


@pytest.fixture(scope="session")
def si_snapshot():
    base = Path("./data/big/silicon/2700K")
    matrix_path = base / "Si_DM"
    info_path = base / "info.txt"
    cfg = Config(cutoff_matrix=15.0)
    return Snapshot.from_openmx(
        str(matrix_path),
        str(info_path),
        convention="openmx",
        cfg=cfg,
    )


@pytest.fixture(scope="session")
def silicon_pair():
    # Use the silicon data in the repository
    base = Path("data") / "big" / "silicon" / "900K"
    mat = base / "Si_DM"
    info = base / "info.txt"
    return mat, info


@pytest.fixture(scope="session")
def factory_results():
    # Build shared train/val datasets and mapper (expensive IO)
    # Paths to actual data files
    PAIR_TRAIN_1 = (
        Path("data/small/H2O/original/H2O.matrix"),
        Path("data/small/H2O/original/H2O.info.out"),
    )
    PAIR_TRAIN_2 = (
        Path("data/big/silicon/900K/Si_DM"),
        Path("data/big/silicon/900K/info.txt"),
    )
    # PAIR_VAL_1 = (
    #     Path("data/big/silicon/2700K/Si_DM"),
    #     Path("data/big/silicon/2700K/info.txt"),
    # )
    PAIR_VAL_1 = (
        Path("data/small/H2O/original/H2O.matrix"),
        Path("data/small/H2O/original/H2O.info.out"),
    )
    cfg = Config(
        cutoff_gnn=7.0,
        cutoff_matrix=7.0,
        device="cpu",
        train_target="matrix",
        dtype=torch.float32,
        safety_checks=True,
    )
    fac = DatasetFactory(cfg)
    fac.add_snapshot(*PAIR_TRAIN_1, purpose="train")
    fac.add_snapshot(*PAIR_TRAIN_2, purpose="train")
    fac.add_snapshot(*PAIR_VAL_1, purpose="val")
    train_ds, val_ds, mapper = fac.create()
    return train_ds, val_ds, mapper
