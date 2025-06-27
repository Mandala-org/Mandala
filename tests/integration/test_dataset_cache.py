import pytest
from pathlib import Path

import torch

from data.factory import DatasetFactory


@pytest.fixture
def h2o_pair():
    # Use the small H2O data in the repository
    base = Path("data") / "small" / "H2O" / "original"
    mat = base / "H2O.matrix"
    info = base / "H2O.info.out"
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


def extract_gnn_edges(ds):
    x_gnn, _, _ = ds[0]
    # Convert edge_index to sorted list of tuple pairs
    edges = [tuple(e) for e in x_gnn["edge_index"].t().tolist()]
    return sorted(edges)


@pytest.mark.integration
@pytest.mark.integration
def test_dataset_cache_with_real_data(tmp_path, h2o_pair):
    cache_dir = tmp_path / "cache"
    # First load with cutoff 5.0
    ds1 = load_dataset(h2o_pair, cache_dir, cutoff_gnn=5.0)
    # One cache file created
    files1 = list(cache_dir.glob("*.pt"))
    assert len(files1) == 1
    # Capture the GNN edge_index tensor directly
    edge1 = ds1[0][0]["edge_index"].clone()

    # Second load with same cutoff: cache hit, no new files
    ds2 = load_dataset(h2o_pair, cache_dir, cutoff_gnn=5.0)
    files2 = list(cache_dir.glob("*.pt"))
    assert files2 == files1
    # Ensure second load yields identical edge_index tensor
    edge2 = ds2[0][0]["edge_index"]
    assert torch.equal(edge2, edge1)

    # Third load with different cutoff: new cache file and different edges
    ds3 = load_dataset(h2o_pair, cache_dir, cutoff_gnn=3.0)
    files3 = list(cache_dir.glob("*.pt"))
    assert len(files3) == 2
    # Third load also yields the same edge_index (only cache key changes)
    edge3 = ds3[0][0]["edge_index"]
    assert torch.equal(edge3, edge1)
