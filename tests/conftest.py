import os
import sys
import random
import numpy as np
import torch
import pytest
from pathlib import Path
from data.factory import DatasetFactory

# Ensure the project src/ directory is on PYTHONPATH for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
# Add venv site-packages to PYTHONPATH to allow importing e3nn, torch, etc.
venv_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "e3gnn4matrix-venv")
)
venv_sp = os.path.join(
    venv_dir,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
if os.path.isdir(venv_sp):
    sys.path.insert(0, venv_sp)


def pytest_configure(config):
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=5)


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
    PAIR_VAL_1 = (
        Path("data/big/silicon/2700K/Si_DM"),
        Path("data/big/silicon/2700K/info.txt"),
    )
    fac = DatasetFactory(
        cutoff_gnn=4.5,
        cutoff_matrix=7.5,
        l_max_sh=3,
        n_radial=64,
        keep_snapshots=False,
        device="cpu",
    )
    fac.add_snapshot(*PAIR_TRAIN_1, purpose="train")
    fac.add_snapshot(*PAIR_TRAIN_2, purpose="train")
    fac.add_snapshot(*PAIR_VAL_1, purpose="val")
    train_ds, val_ds, mapper = fac.create()
    return train_ds, val_ds, mapper
