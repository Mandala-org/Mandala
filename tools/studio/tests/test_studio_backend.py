"""Studio backend: API shape, recipe round trips and the expression language.

Run from the repository root:
    PYTHONPATH=src:. python -m pytest -q tools/studio/tests
"""

import pytest
import torch
import yaml
from e3nn import o3

from studio import expressions, recipes
from studio.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    plugins = tmp_path_factory.mktemp("plugins")
    return create_app(plugins=plugins).test_client()


def test_catalog_describes_transforms_terms_and_inputs(client):
    catalog = client.get("/api/catalog").get_json()
    names = {t["name"] for t in catalog["transforms"]}
    assert {
        "e3mlp",
        "equivariant_convolution",
        "spherical_harmonics",
        "pair_projection",
    } <= names
    for transform in catalog["transforms"]:
        assert {"name", "category", "inputs", "outputs", "options"} <= set(transform)
    assert any(term["name"] == "matrix_error" for term in catalog["objective_terms"])
    assert catalog["inputs"]["edge_vectors"]["representation"] == "geometry"


def test_analysis_reports_stages_slots_costs_and_objective(client):
    document = recipes.load_preset(
        "configs/hybrids/tensor_messages_projected_attention.yaml"
    )
    result = client.post(
        "/api/analyze", json={"document": document, "system": {"H": [0, 1]}}
    ).get_json()
    assert result["errors"] == []
    assert result["totals"]["parameters"] > 0
    for stage in result["stages"]:
        assert stage["error"] is None
        assert stage["parameters"] >= 0 and stage["seconds"] > 0
        for slot in stage["outputs"]:
            assert slot in result["slots"]
    assert result["objective"]["objective"]["terms"]


def test_stage_errors_are_reported_not_raised(client):
    document = recipes.load_preset(
        "configs/hybrids/tensor_messages_projected_attention.yaml"
    )
    document["pipeline"]["stages"][0]["inputs"] = ["no_such_slot"] + document[
        "pipeline"
    ]["stages"][0]["inputs"][1:]
    result = client.post(
        "/api/analyze", json={"document": document, "system": {"H": [0, 1]}}
    ).get_json()
    assert result["errors"]
    assert any("no_such_slot" in error["message"] for error in result["errors"])


SYSTEMS = [None, {"H": [0, 1]}, {"O": [0, 0, 1, 1, 2], "Si": [0, 0, 1, 1, 2]}]


def _clean_analysis(client, document):
    for system in SYSTEMS:
        result = client.post(
            "/api/analyze", json={"document": document, "system": system}
        ).get_json()
        if not result["errors"]:
            return system, result
    return None, None


@pytest.mark.parametrize(
    "preset",
    [p["path"] for p in recipes.presets() if p["group"] in {"Reference", "Hybrids"}],
)
def test_yaml_round_trip_preserves_the_recipe(client, preset):
    document = recipes.load_preset(preset)
    system, before = _clean_analysis(client, document)
    if before is None:
        pytest.skip(f"{preset} does not compile for the test systems")
    document["studio"] = {
        "orientation": "LR",
        "expanded": [],
        "layout": {"stage:x": {"x": 1, "y": 2}},
    }
    text = client.post("/api/export", json={"document": document}).get_json()["yaml"]
    reloaded = yaml.safe_load(text)
    assert reloaded == document
    after = client.post(
        "/api/analyze", json={"document": reloaded, "system": system}
    ).get_json()
    assert after["totals"]["parameters"] == before["totals"]["parameters"]
    assert [s["id"] for s in after["stages"]] == [s["id"] for s in before["stages"]]


def test_flat_configuration_is_lowered_with_blocks(client):
    document = recipes.load_preset("configs/reference/mandala_paper.yaml")
    result = client.post(
        "/api/analyze", json={"document": document, "system": {"Si": [0, 0, 1, 1, 2]}}
    ).get_json()
    assert result["flat"] and result["errors"] == []
    assert {"node_update", "edge_update", "readout"} <= set(
        result["pipeline"]["blocks"]
    )
    assert result["objective"]["lowered"]


X = [{"name": "x", "domain": "node", "irreps": "8x0e+4x1o"}]


def test_expression_types_every_variable():
    result = expressions.check(
        "mix", X, 'h = linear(x, "4x0e+2x1o")\ns = norm(h)\nout = h * sigmoid(s)'
    )
    assert result["ok"], result
    assert result["types"]["s"]["irreps"] == "6x0e"
    assert result["types"]["out"]["irreps"] == "4x0e+2x1o"
    assert result["types"]["out"]["domain"] == "node"


@pytest.mark.parametrize(
    "source, message",
    [
        ("out = tanh(x)", "invariant"),
        ('out = x + linear(x, "4x0e")', "Cannot add"),
        ("out = x + 1.0", "breaks equivariance"),
        ('out = tp(x, x, "1x3e")', "cannot produce"),
        ('out = x * linear(x, "5x0e")', "Cannot multiply"),
        ("out = y", "Unknown name"),
        ("out = x ** 2", "Only + - * /"),
    ],
)
def test_expression_rejects_non_equivariant_programs(source, message):
    result = expressions.check("bad", X, source)
    assert not result["ok"]
    assert message in result["error"]["message"]
    assert result["error"]["line"] == 1


def test_generated_plugin_registers_an_equivariant_transform(client):
    from net.features import FeatureSpec
    from net.registry import build_transform

    request = {
        "name": "studio_test_gate",
        "inputs": X + [{"name": "s", "domain": "node", "irreps": "3x0e"}],
        "source": 'h = linear(x, "8x0e+4x1o")\ng = sigmoid(mlp(concat(norm(h), s), [16], 12))\nout = gate(h, g) + param("alpha", 0.5) * x',
        "outputs": ["out"],
        "summary": "Test gate",
        "category": "custom",
    }
    response = client.post("/api/plugin", json=request).get_json()
    assert any(t["name"] == "studio_test_gate" for t in response["transforms"])
    specs = [
        FeatureSpec("node", "irreps", "8x0e+4x1o"),
        FeatureSpec("node", "irreps", "3x0e"),
    ]
    module, outputs = build_transform("studio_test_gate", {}, specs)
    assert outputs[0].irreps == "8x0e+4x1o"
    torch.manual_seed(0)
    x, s = torch.randn(5, 20), torch.randn(5, 3)
    rotation = o3.rand_matrix()
    D = o3.Irreps("8x0e+4x1o").D_from_matrix(rotation)
    torch.testing.assert_close(
        module(x @ D.T, s), module(x, s) @ D.T, atol=1e-5, rtol=1e-5
    )
    with pytest.raises(ValueError, match="must have irreps"):
        build_transform(
            "studio_test_gate", {}, [FeatureSpec("node", "irreps", "8x0e"), specs[1]]
        )


def test_plugin_endpoint_refuses_invalid_programs(client):
    response = client.post(
        "/api/plugin", json={"name": "broken", "inputs": X, "source": "out = exp(x)"}
    )
    assert response.status_code == 400


def test_definitions_never_reach_executable_code(tmp_path):
    marker = tmp_path / "marker"
    source = f'out = x\n# """;open({str(marker)!r},"w").write("x");"""'
    result = expressions.check(
        "inert",
        [{"name": "x", "domain": "node", "irreps": "4x0e"}],
        source,
        ["out"],
        'a"""b\\',
    )
    assert result["ok"]
    exec(compile(result["code"], "plugin", "exec"), {"__name__": "inert_plugin"})
    assert not marker.exists()


@pytest.mark.parametrize(
    "source",
    [
        'out = linear(x, "0x0e")',
        'out = linear(x, "3x1e")',
        'out = x * param("training", 1.0)',
        "out = mlp(x, [True], 4)",
    ],
)
def test_degenerate_or_shadowing_programs_are_rejected(source):
    result = expressions.check(
        "bad", [{"name": "x", "domain": "node", "irreps": "4x0e"}], source
    )
    assert not result["ok"]


def test_server_rejects_cross_origin_and_non_json_requests(client):
    assert (
        client.get(
            "/api/presets", headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        client.get("/api/presets", headers={"Host": "evil.example:8765"}).status_code
        == 403
    )
    response = client.post(
        "/api/analyze", data="{}", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 415
    assert (
        client.get(
            "/api/presets", headers={"Origin": "http://localhost:8765"}
        ).status_code
        == 200
    )


def test_plugins_are_confined_to_the_repository_and_never_shadow_builtins(client):
    analysis = client.post(
        "/api/analyze", json={"document": {"plugins": ["/etc/passwd.py"]}}
    ).get_json()
    assert "plugins/ or examples/" in analysis["errors"][0]["message"]
    response = client.post(
        "/api/plugin",
        json={
            "name": "concat",
            "inputs": [{"name": "x", "domain": "node", "irreps": "2x0e"}],
            "source": "out = x",
        },
    )
    assert (
        response.status_code == 400
        and "framework transformation" in response.get_json()["error"]
    )


def test_catalog_lists_library_blocks_with_compiler_port_contracts(client):
    catalog = client.get("/api/catalog").get_json()
    library = catalog["block_library"]
    assert {"e3_node_update", "crystal_graph_update", "density_interaction"} <= set(
        library
    )
    block = library["e3_edge_update"]
    assert block["library"] == "deeph_e3"
    assert [p["name"] for p in block["ports"]["inputs"]] == block["inputs"]
    nodes = block["ports"]["inputs"][0]
    assert (nodes["domain"], nodes["kind"]) == ("node", "irreps")
    term = next(t for t in catalog["objective_terms"] if t["name"] == "matrix_error")
    assert term["operators"] == ["operator"]


def test_import_inlines_block_libraries_and_analysis_reports_block_ports(client):
    result = client.post(
        "/api/import",
        json={"path": "configs/reference/deeph_e3.yaml", "system": {"Si": [0, 1]}},
    ).get_json()
    pipeline = result["document"]["pipeline"]
    assert "block_libraries" not in pipeline
    assert {"e3_node_update", "e3_edge_update"} <= set(pipeline["blocks"])
    analysis = result["analysis"]
    assert analysis["errors"] == []
    ports = analysis["blocks"]["e3_node_update"]["inputs"]
    assert [p["name"] for p in ports] == pipeline["blocks"]["e3_node_update"]["inputs"]
    exported = yaml.safe_load(
        client.post("/api/export", json={"document": result["document"]}).get_json()[
            "yaml"
        ]
    )
    assert exported["pipeline"]["blocks"]["e3_node_update"]["library"] == "deeph_e3"


def test_disabled_stage_is_bypassed_and_reported(client):
    result = client.post(
        "/api/import",
        json={"path": "configs/reference/deeph_e3.yaml", "system": {"Si": [0, 1]}},
    ).get_json()
    document = result["document"]
    stage = next(
        s for s in document["pipeline"]["stages"] if s["id"] == "node_update_1"
    )
    stage["enabled"] = False
    analysis = client.post(
        "/api/analyze", json={"document": document, "system": {"Si": [0, 1]}}
    ).get_json()
    assert analysis["errors"] == []
    record = next(s for s in analysis["stages"] if s["id"] == "node_update_1")
    assert record["enabled"] is False and record["parameters"] == 0
    assert analysis["totals"]["parameters"] < result["analysis"]["totals"]["parameters"]
