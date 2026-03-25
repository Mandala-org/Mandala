import pytest
from pathlib import Path
import torch

from data.factory import DatasetFactory
from net.common import Config


@pytest.fixture
def silicon_pair():
    # Use the silicon data in the repository
    base = Path("data") / "big" / "silicon" / "900K"
    mat = base / "Si_DM"
    info = base / "info.txt"
    return mat, info


def load_dataset(pair, cache_dir, cutoff_radius):
    mat, info = pair
    cfg = Config(
        cutoff_radius=cutoff_radius,
        n_radial=64,
        device="cpu",
        snapshot_cache_dir=str(cache_dir),
        safety_checks=True,
    )
    fac = DatasetFactory(cfg)
    fac.add_snapshot(mat, info)
    train_ds, _, _ = fac.create()
    return train_ds


@pytest.mark.integration
def test_dataset_cache_with_silicon_data(tmp_path, silicon_pair):
    cache_dir = tmp_path / "cache"

    # 1. Load with cutoff 5.0
    ds1 = load_dataset(silicon_pair, cache_dir, cutoff_radius=5.0)
    x1, y1 = ds1[0]
    hamiltonian1 = y1["hamiltonian"]

    # 2. Load with cutoff 3.0
    ds2 = load_dataset(silicon_pair, cache_dir, cutoff_radius=3.0)
    x2, _ = ds2[0]

    # 3. Check that the edge cutoff index is now smaller
    assert x2["edge_index"].shape[1] < x1["edge_index"].shape[1]

    # 4. Load with cutoff 5.0 again (cache hit) #! Test whether it is an actual cache hit
    ds3 = load_dataset(silicon_pair, cache_dir, cutoff_radius=5.0)
    _, y3 = ds3[0]
    hamiltonian3 = y3["hamiltonian"]

    # 5. Check that some random matrix block is the same as in the first dataset
    block_key = list(hamiltonian1.pair_blocks.keys())[0]
    assert torch.equal(
        hamiltonian1.pair_blocks[block_key], hamiltonian3.pair_blocks[block_key]
    )
