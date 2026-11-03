"""Structure-only stages are detected, cached with samples and reused safely."""

import sys
from pathlib import Path

import pytest
import torch

from data.graph_features import attach_static_features
from net.component_defaults import static_feature_pipeline

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "fixtures" / "legacy_mandala")
)
import capture  # noqa: E402


def _sample(model):
    """Structure inputs completed like a dataset sample."""
    from data.graph_features import complete_graph_metadata
    from data.graph_topology import prepare_graph_topology

    x = capture.inputs(model)
    prepare_graph_topology(x, None)
    complete_graph_metadata(
        x, model.mapper, separate_shifted_self=model.cfg.separate_shifted_self
    )
    attach_static_features(x, static_feature_pipeline(model.cfg, model.mapper))
    x["_geometry_reference"] = {
        "positions": x["positions"].clone(),
        "box": x["box"].clone(),
    }
    return x


@pytest.fixture(scope="module")
def model():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield capture.build("paper").eval()
    torch.set_default_dtype(previous)


def test_only_parameter_free_structure_stages_are_static(model):
    static = {record.id for record in model.pipeline.records if record.static}
    assert static == {"radial", "harmonics", "roles"}
    cache = static_feature_pipeline(model.cfg, model.mapper)
    assert [record.id for record in cache.records] == ["radial", "harmonics", "roles"]
    assert cache.static_fingerprint == model.pipeline.static_fingerprint


def test_cached_static_features_are_used_and_equal_recomputation(model, monkeypatch):
    x = _sample(model)
    assert set(x["_static_features"]["values"]) == {"radial", "harmonics", "roles"}
    with torch.no_grad():
        cached = model(x)
    calls = []
    module = model.pipeline.transforms["radial"]
    monkeypatch.setattr(
        module,
        "forward",
        lambda *a: calls.append(1) or type(module).forward(module, *a),
    )
    with torch.no_grad():
        model(x)
    assert not calls
    fresh = dict(x)
    del fresh["_static_features"]
    with torch.no_grad():
        recomputed = model(fresh)
    assert calls
    for key, value in recomputed["hamiltonian"].pair_vectors.items():
        torch.testing.assert_close(cached["hamiltonian"].pair_vectors[key], value)


def test_cache_is_ignored_for_moved_or_differentiated_geometry(model):
    x = _sample(model)
    assert model._cached_static_features(x)
    moved = dict(x, positions=x["positions"] + 0.1)
    assert not model._cached_static_features(moved)
    differentiated = dict(x, positions=x["positions"].clone().requires_grad_())
    assert not model._cached_static_features(differentiated)
    stale = dict(x, _static_features={**x["_static_features"], "fingerprint": "other"})
    assert not model._cached_static_features(stale)


def test_batched_graphs_keep_and_reuse_concatenated_static_features(model):
    from data.sample import union_graph_inputs

    first, second = _sample(model), _sample(model)
    second["positions"] = second["positions"] * 1.03
    from data.graph_features import attach_static_features as attach

    second.pop("_static_features")
    attach(second, static_feature_pipeline(model.cfg, model.mapper))
    second["_geometry_reference"] = {
        "positions": second["positions"].clone(),
        "box": second["box"].clone(),
    }
    union = union_graph_inputs(
        [(first, {}), (second, {})], layouts=model.pipeline.input_layouts
    )
    assert set(union["_static_features"]["values"]) == {"radial", "harmonics", "roles"}
    with torch.no_grad():
        cached = model(union)
    fresh = dict(union)
    del fresh["_static_features"]
    with torch.no_grad():
        recomputed = model(fresh)
    for key, value in recomputed["hamiltonian"].pair_vectors.items():
        torch.testing.assert_close(cached["hamiltonian"].pair_vectors[key], value)
