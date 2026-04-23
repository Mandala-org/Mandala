import pytest
import torch

from data.gnn_dataset import E3GNNDataset


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
