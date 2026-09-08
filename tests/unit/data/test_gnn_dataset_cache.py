import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from data.gnn_dataset import E3GNNDataset


def test_corrected_density_normalization_invalidates_both_cache_layers(
    tmp_path, monkeypatch
):
    import data.gnn_dataset as module
    from net.common import Config

    ds = object.__new__(E3GNNDataset)
    ds.cfg = Config(snapshot_cache_dir=str(tmp_path / "cache"))
    ds.convention, ds.dtype = "e3nn", torch.float32
    ds.mapper = SimpleNamespace(
        orbital_cfg=SimpleNamespace(to_dict=lambda: {"Si": "1s"})
    )
    ds.hamiltonian_envelope_mode = "off"
    ds.pair_distance_normalization = "off"
    ds.loss_weighting_mode = "off"
    matrix_path, info_path = tmp_path / "HS.out", tmp_path / "Si.out"
    matrix_path.write_text("matrix")
    info_path.write_text("info")
    new_snapshot = ds._snapshot_cache_file(matrix_path, info_path)
    new_sample = ds._preprocessed_sample_cache_file(matrix_path, info_path)
    monkeypatch.setattr(module, "SNAPSHOT_CACHE_VERSION", "v3")
    monkeypatch.setattr(module, "PREPROCESSED_SAMPLE_CACHE_VERSION", "v7")
    assert new_snapshot != ds._snapshot_cache_file(matrix_path, info_path)
    assert new_sample != ds._preprocessed_sample_cache_file(matrix_path, info_path)


@pytest.mark.unit
def test_load_preprocessed_sample_uses_weights_only_false(monkeypatch, tmp_path):
    ds = object.__new__(E3GNNDataset)
    cache_file = tmp_path / "sample.pt"
    cache_file.write_bytes(b"placeholder")
    captured = {}

    def fake_torch_load(path, map_location=None, weights_only=None, **kwargs):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return ({"x": 1}, {"y": 2})

    monkeypatch.setattr(torch, "load", fake_torch_load)

    loaded = E3GNNDataset._load_preprocessed_sample(ds, cache_file)

    assert loaded == ({"x": 1}, {"y": 2})
    assert captured["path"] == cache_file
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is False


@pytest.mark.unit
def test_load_preprocessed_sample_reports_and_deletes_bad_cache(
    monkeypatch, tmp_path, capsys
):
    ds = object.__new__(E3GNNDataset)
    cache_file = tmp_path / "sample.pt"
    cache_file.write_bytes(b"placeholder")

    def fake_torch_load(path, map_location=None, weights_only=None, **kwargs):
        raise RuntimeError("bad cache payload")

    monkeypatch.setattr(torch, "load", fake_torch_load)

    loaded = E3GNNDataset._load_preprocessed_sample(ds, cache_file)

    assert loaded is None
    assert not cache_file.exists()
    captured = capsys.readouterr()
    assert "Failed to load preprocessed sample cache" in captured.out
    assert "bad cache payload" in captured.out


@pytest.mark.unit
def test_preprocessed_cache_hit_bypasses_raw_snapshot_loading(monkeypatch, tmp_path):
    ds = object.__new__(E3GNNDataset)
    ds.preprocessed_cache_hits = 0
    matrix_path = tmp_path / "HS.out"
    info_path = tmp_path / "sample.out"
    cache_path = tmp_path / "sample.pt"
    cache_path.write_bytes(b"cached")
    expected = ({"cached": True}, {"target": True})

    monkeypatch.setattr(
        E3GNNDataset,
        "_preprocessed_sample_cache_file",
        lambda self, matrix, info: cache_path,
    )
    monkeypatch.setattr(
        E3GNNDataset,
        "_load_preprocessed_sample",
        lambda self, path: expected,
    )

    result = ds._load_preprocessed_sample_if_available(matrix_path, info_path)

    assert result is expected
    assert ds.preprocessed_cache_hits == 1


@pytest.mark.unit
def test_gnn_dataset_shuffles_warmup_order_but_preserves_final_order(
    monkeypatch,
):
    ds = object.__new__(E3GNNDataset)
    cfg = SimpleNamespace(
        dtype=torch.float32,
        train_on_forces=False,
        train_on_stress=False,
        enable_forces=False,
        enable_stress=False,
        l_max=4,
        verbosity=0,
        snapshot_cache_dir=None,
        cutoff_radius=7.0,
        apply_cutoff_to_targets=True,
        matrix_targets=["hamiltonian", "overlap", "density"],
        train_target="matrix",
        symmetrize_hamiltonian_targets=True,
        require_exact_edge_match=True,
        precompute_edge_features=True,
        separate_shifted_self=True,
        shuffle_snapshot_load_order=True,
    )
    mapper = SimpleNamespace(
        orbital_cfg=SimpleNamespace(to_dict=lambda: {}),
    )
    ds.cfg = cfg
    ds.mapper = mapper
    ds.convention = "e3nn"
    ds.dtype = torch.float32
    ds.sh_irreps = object()
    ds.device = torch.device("cpu")
    ds.snapshot_paths = [
        (Path(f"/tmp/mat{i}"), Path(f"/tmp/info{i}")) for i in range(5)
    ]
    ds.snapshots = []
    ds.snapshot_cache_hits = 0
    ds.snapshot_cache_misses = 0
    ds.preprocessed_cache_hits = 0
    ds.preprocessed_cache_misses = 0
    call_order = []

    monkeypatch.setattr("data.gnn_dataset.secrets.randbits", lambda _: 1)

    def fake_load_snapshot(self, matrix_path, info_path):
        call_order.append(matrix_path.name)
        return object()

    def fake_load_cached(self, matrix_path, info_path):
        return None

    def fake_build_and_cache(self, matrix_path, info_path, snapshot):
        return ({"matrix": matrix_path.name}, {"info": info_path.name})

    monkeypatch.setattr(E3GNNDataset, "_load_snapshot", fake_load_snapshot)
    monkeypatch.setattr(
        E3GNNDataset, "_load_preprocessed_sample_if_available", fake_load_cached
    )
    monkeypatch.setattr(
        E3GNNDataset, "_build_and_cache_preprocessed_sample", fake_build_and_cache
    )

    E3GNNDataset.__init__(ds, ds.snapshot_paths, mapper, cfg, "e3nn")

    assert [sample[0]["matrix"] for sample in ds.snapshots] == [
        "mat0",
        "mat1",
        "mat2",
        "mat3",
        "mat4",
    ]
    assert sorted(call_order) == ["mat0", "mat1", "mat2", "mat3", "mat4"]
    assert call_order != ["mat0", "mat1", "mat2", "mat3", "mat4"]


@pytest.mark.unit
def test_snapshot_cache_file_changes_when_cif_metadata_changes(tmp_path):
    ds = object.__new__(E3GNNDataset)
    cache_root = tmp_path / "cache"
    matrix_path = tmp_path / "sample" / "HS.out"
    info_path = tmp_path / "sample" / "ZnCuSeS.out"
    cif_path = tmp_path / "sample" / "ZnCuSeS.cif"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text("matrix")
    info_path.write_text("info")
    cif_path.write_text("cif-v1")

    ds.cfg = SimpleNamespace(
        snapshot_cache_dir=str(cache_root),
        allow_openmx_positions_box_from_out=False,
    )
    ds.convention = "e3nn"
    ds.dtype = torch.float32

    cache_file_before = E3GNNDataset._snapshot_cache_file(ds, matrix_path, info_path)
    assert cache_file_before is not None

    cif_path.write_text("cif-v2 with different size")

    cache_file_after = E3GNNDataset._snapshot_cache_file(ds, matrix_path, info_path)
    assert cache_file_after is not None
    assert cache_file_before != cache_file_after


@pytest.mark.unit
def test_preprocessed_cache_file_changes_when_geometry_source_mode_changes(tmp_path):
    ds = object.__new__(E3GNNDataset)
    cache_root = tmp_path / "cache"
    matrix_path = tmp_path / "sample" / "HS.out"
    info_path = tmp_path / "sample" / "ZnCuSeS.out"
    cif_path = tmp_path / "sample" / "ZnCuSeS.cif"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text("matrix")
    info_path.write_text("info")
    cif_path.write_text("cif")

    ds.mapper = SimpleNamespace(
        orbital_cfg=SimpleNamespace(to_dict=lambda: {"Zn": "1s"})
    )
    ds.convention = "e3nn"
    ds.dtype = torch.float32
    ds.hamiltonian_envelope_mode = "off"
    ds.pair_distance_normalization = "off"
    ds.loss_weighting_mode = "off"
    ds.cfg = SimpleNamespace(
        snapshot_cache_dir=str(cache_root),
        l_max=5,
        n_radial=8,
        radial_embedding_scale="none",
        cutoff_radius=11.0,
        apply_cutoff_to_targets=True,
        matrix_targets=["hamiltonian", "density", "overlap"],
        train_target="matrix",
        symmetrize_hamiltonian_targets=True,
        require_exact_edge_match=True,
        precompute_edge_features=True,
        separate_shifted_self=True,
        allow_openmx_positions_box_from_out=False,
    )

    cache_file_cif = E3GNNDataset._preprocessed_sample_cache_file(
        ds, matrix_path, info_path
    )
    assert cache_file_cif is not None

    ds.cfg.allow_openmx_positions_box_from_out = True

    cache_file_out_fallback = E3GNNDataset._preprocessed_sample_cache_file(
        ds, matrix_path, info_path
    )
    assert cache_file_out_fallback is not None
    assert cache_file_cif != cache_file_out_fallback
