import pytest
from pathlib import Path
from types import SimpleNamespace
import torch

from data.factory import DatasetFactory
from data.snapshot import Snapshot
from net.common import Config


@pytest.fixture
def small_snapshot_pair():
    base = Path("data") / "small" / "H2O" / "original"
    mat = base / "H2O.matrix"
    info = base / "H2O.info.out"
    return mat, info


def load_dataset(pair, cache_dir, cutoff_radius):
    mat, info = pair
    cfg = Config(
        cutoff_radius=cutoff_radius,
        n_radial=8,
        device="cpu",
        snapshot_cache_dir=str(cache_dir),
        safety_checks=True,
    )
    fac = DatasetFactory(cfg)
    fac.add_snapshot(mat, info)
    train_ds, _, _ = fac.create()
    return train_ds


@pytest.mark.integration
def test_dataset_cache_reuses_preprocessed_snapshot(
    tmp_path, small_snapshot_pair, small_angular_snapshot_e3nn, monkeypatch
):
    cache_dir = tmp_path / "cache"
    load_calls = 0

    def load_small_snapshot(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return small_angular_snapshot_e3nn

    monkeypatch.setattr(Snapshot, "from_openmx", load_small_snapshot)
    monkeypatch.setattr(
        DatasetFactory,
        "_load_info",
        lambda self, *paths: SimpleNamespace(orbital_set={"H": "1s1p"}),
    )

    # 1. Load with cutoff 5.0
    ds1 = load_dataset(small_snapshot_pair, cache_dir, cutoff_radius=5.0)
    x1, y1 = ds1[0]
    hamiltonian1 = y1["hamiltonian"]

    # 2. Load with cutoff 3.0
    ds2 = load_dataset(small_snapshot_pair, cache_dir, cutoff_radius=3.0)
    x2, _ = ds2[0]

    # 3. Check that the edge cutoff index is now smaller
    assert x2["edge_index"].shape[1] < x1["edge_index"].shape[1]

    # 4. Load with cutoff 5.0 again (cache hit) #! Test whether it is an actual cache hit
    cache_files_before = {
        path: path.stat().st_mtime_ns for path in cache_dir.rglob("*") if path.is_file()
    }
    ds3 = load_dataset(small_snapshot_pair, cache_dir, cutoff_radius=5.0)
    _, y3 = ds3[0]
    hamiltonian3 = y3["hamiltonian"]

    # 5. Check that some random matrix block is the same as in the first dataset
    block_key = list(hamiltonian1.pair_blocks.keys())[0]
    assert torch.equal(
        hamiltonian1.pair_blocks[block_key], hamiltonian3.pair_blocks[block_key]
    )
    assert cache_files_before
    assert cache_files_before == {
        path: path.stat().st_mtime_ns for path in cache_dir.rglob("*") if path.is_file()
    }
    # The raw snapshot is cutoff-independent and reused by both preprocessing
    # configurations; only the preprocessed sample cache is cutoff-specific.
    assert load_calls == 1
