import pytest
from pathlib import Path

import torch

from data.factory import DatasetFactory


@pytest.fixture
def silicon_pair():
    # Use the silicon data in the repository
    base = Path("data") / "big" / "silicon" / "900K"
    mat = base / "Si_DM"
    info = base / "info.txt"
    return mat, info


def load_dataset(pair, cache_dir, cutoff_gnn):
    mat, info = pair
    fac = DatasetFactory(
        cutoff_gnn=cutoff_gnn,
        cutoff_matrix=7.5,
        l_max_sh=3,
        n_radial=64,
        device="cpu",
        cache_root=str(cache_dir),
    )
    fac.add_snapshot(mat, info, purpose="train")
    ds, val_ds, _ = fac.create()
    # Expect only train split
    assert val_ds is None
    return ds


@pytest.mark.integration
def test_dataset_cache_with_silicon_data(tmp_path, silicon_pair):
    cache_dir = tmp_path / "cache"

    # 1. Load with cutoff 5.0
    ds1 = load_dataset(silicon_pair, cache_dir, cutoff_gnn=5.0)
    x1, y1 = ds1[0]
    hamiltonian1 = y1["hamiltonian"]

    # 2. Load with cutoff 3.0
    ds2 = load_dataset(silicon_pair, cache_dir, cutoff_gnn=3.0)
    x2, _ = ds2[0]

    # 3. Check that the edge cutoff index is now smaller
    assert x2["index_gnn_cutoff"] < x1["index_gnn_cutoff"]

    # 4. Load with cutoff 5.0 again (cache hit) #! Test whether it is an actual cache hit
    ds3 = load_dataset(silicon_pair, cache_dir, cutoff_gnn=5.0)
    _, y3 = ds3[0]
    hamiltonian3 = y3["hamiltonian"]

    # 5. Check that some random matrix block is the same as in the first dataset
    block_key = list(hamiltonian1.pair_vectors.keys())[0]
    assert torch.equal(
        hamiltonian1.pair_vectors[block_key], hamiltonian3.pair_vectors[block_key]
    )
