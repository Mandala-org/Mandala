import torch
import pytest
from pathlib import Path

import data.gnn_dataset as gnn_mod
from data.gnn_dataset import E3GNNDataset
from data.factory import DatasetFactory
import dataclasses


def deep_compare(a, b):
    """Recursively compare samples of dicts, tensors, and lists."""
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(b.keys())
        for k in a:
            deep_compare(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b):
            deep_compare(x, y)
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert torch.allclose(a, b)
    elif dataclasses.is_dataclass(a):
        # compare dataclass fields
        d1 = dataclasses.asdict(a)
        d2 = dataclasses.asdict(b)
        deep_compare(d1, d2)
    else:
        assert a == b


@pytest.mark.parametrize(
    "cut_gnn,cut_mat,lmax,radial",
    [
        (5.0, 7.5, 3, 16),
    ],
)
def test_caching(tmp_path, monkeypatch, cut_gnn, cut_mat, lmax, radial):
    # Redirect cache directory to temporary path
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(gnn_mod, "CACHE_ROOT", cache_dir)
    # Prepare a snapshot via factory
    fac = DatasetFactory(
        cutoff_gnn=cut_gnn,
        cutoff_matrix=cut_mat,
        l_max_sh=lmax,
        n_radial=radial,
        device="cpu",
    )
    # Use small H2O test data
    mat = Path("data/small/H2O/original/H2O.matrix")
    info = Path("data/small/H2O/original/H2O.info.out")
    fac.add_snapshot(mat, info, purpose="train")
    # Load one Snapshot instance
    snap = fac._load_snapshot(mat, info)
    # Create shared mapper
    _, _, mapper = fac.create()
    # First instantiation should process and write cache
    ds1 = E3GNNDataset(
        [snap],
        mapper,
        cutoff_gnn=cut_gnn,
        cutoff_matrix=cut_mat,
        l_max_sh=lmax,
        n_radial=radial,
        device="cpu",
    )
    cache_files = list(cache_dir.glob("*.pt"))
    assert len(cache_files) == 1
    # Second instantiation should reuse cache; no new cache file created
    ds2 = E3GNNDataset(
        [snap],
        mapper,
        cutoff_gnn=cut_gnn,
        cutoff_matrix=cut_mat,
        l_max_sh=lmax,
        n_radial=radial,
        device="cpu",
    )
    cache_files2 = list(cache_dir.glob("*.pt"))
    assert len(cache_files2) == 1, "Expected caching to reuse existing cache file"

    # Compare a sample from both datasets
    sample1 = ds1[0]
    sample2 = ds2[0]
    x_gnn_1, x_matrix_1, y_1 = sample1
    x_gnn_2, x_matrix_2, y_2 = sample2
    deep_compare(x_gnn_1, x_gnn_2)
    deep_compare(x_matrix_1, x_matrix_2)
    deep_compare(y_1["density"].pair_vectors, y_2["density"].pair_vectors)
