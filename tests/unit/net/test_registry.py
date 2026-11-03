"""The transform registry describes, completes and builds every stage."""

import json

import pytest
from torch import nn

from net.features import FeatureSpec
from net.registry import (
    CATEGORIES,
    Option,
    Port,
    build_transform,
    get_transform,
    register_transform,
    registered_transforms,
    transform_catalog,
)


def test_every_registered_transform_is_described_and_serializable():
    catalog = transform_catalog()
    assert len(catalog) == len(registered_transforms()) > 50
    json.dumps(catalog)
    for entry in catalog:
        assert entry["category"] in CATEGORIES and entry["category"] != "custom"
        assert entry["summary"], entry["name"]
        assert entry["inputs"], entry["name"]
        names = [option["name"] for option in entry["options"]]
        assert len(names) == len(set(names)), entry["name"]


def test_options_record_explicit_inferred_and_default_values():
    features = FeatureSpec("node", "irreps", "4x0e+2x1o")
    built = build_transform("irrep_linear", {}, [features], detailed=True)
    assert built.options == {
        "degrees_of": None,
        "irreps_out": "4x0e+2x1o",
        "biases": False,
    }
    assert built.inferred == {"irreps_out"} and built.explicit == frozenset()
    built = build_transform(
        "irrep_linear", {"irreps_out": "3x0e"}, [features], detailed=True
    )
    assert built.explicit == {"irreps_out"} and not built.inferred
    assert built.outputs[0].irreps == "3x0e"


def test_unknown_or_invalid_options_and_arity_are_rejected():
    features = FeatureSpec("edge", irreps="3x0e")
    with pytest.raises(ValueError, match="Unknown options"):
        build_transform("gaussian_envelope", {"width": 2.0}, [features])
    with pytest.raises(ValueError, match="must be one of"):
        build_transform("indexed_aggregation", {"reduction": "max"}, [features] * 3)
    with pytest.raises(ValueError, match="expects inputs"):
        build_transform("irrep_linear", {}, [features, features])
    with pytest.raises(ValueError, match="requires option"):
        build_transform("one_hot", {}, [FeatureSpec("node", "index", frame="none")])


def test_registration_requires_explicit_override_and_keeps_metadata():
    @register_transform(
        "test_identity_stage",
        category="linear",
        summary="Identity",
        inputs=[Port("features")],
        options=[Option("scale", "float", 1.0, key=True)],
    )
    def identity(options, inputs, mapper):
        return nn.Identity(), [inputs[0]]

    with pytest.raises(ValueError, match="already registered"):
        register_transform("test_identity_stage")(identity)
    register_transform(
        "test_identity_stage", override=True, category="linear", inputs=[Port("x")]
    )(identity)
    info = get_transform("test_identity_stage")
    assert info.category == "linear" and [p.name for p in info.inputs] == ["x"]


def test_variadic_ports_accept_any_number_of_operands():
    spec = FeatureSpec("node", "irreps", "2x0e")
    _, (concatenated,) = build_transform("concat", {}, [spec, spec, spec])
    assert concatenated.irreps == "2x0e+2x0e+2x0e"
    _, (merged,) = build_transform("concat", {"simplify": True}, [spec, spec])
    assert merged.irreps == "4x0e"


def test_route_counts_must_cover_every_category_of_the_route_index():
    features = FeatureSpec("edge", "irreps", "2x0e")
    roles = FeatureSpec(
        "edge",
        "index",
        frame="none",
        width=3,
        categories=("onsite", "shifted_self", "offsite"),
    )
    with pytest.raises(ValueError, match="3 categories"):
        build_transform("e3mlp", {"routes": 2}, [features, roles])
    module, _ = build_transform(
        "e3mlp", {"inactive_routes": ["onsite"]}, [features, roles]
    )
    assert len(module.routes) == 3 and not list(module.routes[0].parameters())
    with pytest.raises(ValueError, match="Unknown route category"):
        build_transform("e3mlp", {"inactive_routes": ["bulk"]}, [features, roles])


def test_described_transforms_reject_unknown_options_even_without_option_schema():
    features = FeatureSpec("node", "irreps", "2x0e")
    with pytest.raises(ValueError, match="Unknown options for add"):
        build_transform("add", {"scale": 2.0}, [features, features])
