"""Composite blocks expand into namespaced stages with substituted options."""

import pytest
import torch

from net.blocks import expand_blocks
from net.pipeline import FeaturePipeline

BLOCK = {
    "inputs": ["features"],
    "outputs": ["out"],
    "options": {"irreps_out": None, "label": "x"},
    "stages": [
        {
            "id": "mix",
            "transform": "irrep_linear",
            "inputs": ["features"],
            "outputs": ["mixed"],
            "options": {"irreps_out": "$irreps_out"},
        },
        {
            "id": "act",
            "transform": "irrep_activation",
            "inputs": ["mixed"],
            "outputs": ["out"],
        },
    ],
}


def _pipeline(stages, blocks):
    return {
        "inputs": {
            "h": {"domain": "node", "representation": "irreps", "irreps": "3x0e+1x1o"}
        },
        "blocks": blocks,
        "stages": stages,
        "outputs": {"features": stages[-1]["outputs"][0]},
    }


def test_instances_are_namespaced_and_null_options_are_inferred():
    stages = [
        {"id": "a", "transform": "unit", "inputs": ["h"], "outputs": ["h1"]},
        {
            "id": "b",
            "transform": "unit",
            "inputs": ["h1"],
            "outputs": ["h2"],
            "options": {"irreps_out": "2x0e"},
        },
    ]
    pipeline = FeaturePipeline(_pipeline(stages, {"unit": BLOCK}), num_species=1)
    assert [r.id for r in pipeline.records] == ["a/mix", "a/act", "b/mix", "b/act"]
    first, second = pipeline.records[0], pipeline.records[2]
    assert first.inputs == ("h",) and first.outputs == ("a/mixed",)
    assert "irreps_out" in first.inferred and first.options["irreps_out"] == "3x0e+1x1o"
    assert second.options["irreps_out"] == "2x0e" and "irreps_out" in second.explicit
    assert pipeline.feature_specs["h2"].irreps == "2x0e"
    assert pipeline.groups["b"]["block"] == "unit" and pipeline.groups["b"][
        "explicit"
    ] == ["irreps_out"]
    output = pipeline({"h": torch.randn(4, 6)})["features"]
    assert output.shape == (4, 2)


def test_nested_blocks_and_textual_substitution():
    outer = {
        "inputs": ["x"],
        "outputs": ["y"],
        "options": {"width": 2},
        "stages": [
            {
                "id": "inner",
                "transform": "unit",
                "inputs": ["x"],
                "outputs": ["y"],
                "options": {"irreps_out": "$(width)x0e"},
            },
        ],
    }
    stages, groups = expand_blocks(
        [
            {
                "id": "o",
                "transform": "outer",
                "inputs": ["h"],
                "outputs": ["z"],
                "options": {"width": 5},
            }
        ],
        {"unit": BLOCK, "outer": outer},
    )
    assert [stage["id"] for stage in stages] == ["o/inner/mix", "o/inner/act"]
    assert stages[0]["options"]["irreps_out"] == "5x0e"
    assert stages[-1]["outputs"] == ["z"]
    assert set(groups) == {"o", "o/inner"}


@pytest.mark.parametrize(
    "block, instance, message",
    [
        (
            {**BLOCK, "stages": [dict(BLOCK["stages"][0], inputs=["edge_index"])]},
            {},
            "neither block inputs",
        ),
        (BLOCK, {"options": {"depth": 3}}, "unknown options"),
        (BLOCK, {"inputs": ["h", "h"]}, "expects inputs"),
        (
            {
                **BLOCK,
                "stages": [
                    dict(BLOCK["stages"][0], options={"irreps_out": "$missing"}),
                    BLOCK["stages"][1],
                ],
            },
            {},
            "unknown block option",
        ),
    ],
)
def test_invalid_blocks_and_instances_are_rejected(block, instance, message):
    stage = {
        "id": "a",
        "transform": "unit",
        "inputs": ["h"],
        "outputs": ["y"],
        **instance,
    }
    with pytest.raises(ValueError, match=message):
        expand_blocks([stage], {"unit": block})


def test_collect_mode_reports_every_failure_and_skips_dependents():
    stages = [
        {
            "id": "bad",
            "transform": "irrep_linear",
            "inputs": ["h"],
            "outputs": ["h1"],
            "options": {"irreps_out": "q"},
        },
        {
            "id": "after",
            "transform": "irrep_activation",
            "inputs": ["h1"],
            "outputs": ["h2"],
        },
        {
            "id": "unknown",
            "transform": "no_such_transform",
            "inputs": ["h"],
            "outputs": ["h3"],
        },
    ]
    pipeline = FeaturePipeline(_pipeline(stages, {}), num_species=1, errors="collect")
    failed = {stage for stage, _ in pipeline.errors}
    assert failed == {"bad", "after", "unknown"}
    assert "failing stage" in dict(pipeline.errors)["after"]
    with pytest.raises(ValueError):
        FeaturePipeline(_pipeline(stages, {}), num_species=1)


def test_collect_mode_isolates_malformed_stages_and_block_instances():
    stages = [
        {"id": "good", "transform": "irrep_linear", "inputs": ["h"], "outputs": ["h1"]},
        {"transform": "irrep_linear", "inputs": ["h"], "outputs": ["h2"]},
        {
            "id": "bad_block",
            "transform": "unit",
            "inputs": ["h"],
            "outputs": ["h3"],
            "options": {"depth": 1},
        },
        {
            "id": "options",
            "transform": "irrep_linear",
            "inputs": ["h"],
            "outputs": ["h4"],
            "options": [1],
        },
        {"id": "good", "transform": "irrep_linear", "inputs": ["h"], "outputs": ["h5"]},
        {
            "id": "after_block",
            "transform": "irrep_activation",
            "inputs": ["h3"],
            "outputs": ["h6"],
        },
    ]
    pipeline = FeaturePipeline(
        _pipeline(stages, {"unit": BLOCK}), num_species=1, errors="collect"
    )
    assert [record.id for record in pipeline.records if record.error is None] == [
        "good"
    ]
    messages = " ".join(message for _, message in pipeline.errors)
    for fragment in (
        "needs an id",
        "unknown options",
        "options must be a mapping",
        "unique",
        "failing stage",
    ):
        assert fragment in messages


def _linear(sid, source, target, **options):
    return {
        "id": sid,
        "transform": "irrep_linear",
        "inputs": [source],
        "outputs": [target],
        "options": options,
    }


def test_disabled_stages_and_instances_pass_their_bypass_input_through():
    stages = [
        _linear("a", "h", "a", irreps_out="3x0e+1x1o"),
        {
            "id": "res",
            "transform": "residual",
            "inputs": ["h", "a"],
            "outputs": ["r"],
            "enabled": False,
        },
        {"id": "blk", "transform": "gate", "inputs": ["r"], "outputs": ["out"]},
    ]
    blocks = {"gate": {**BLOCK, "options": {"irreps_out": None, "label": "x"}}}
    spec = _pipeline(stages, blocks)
    spec["stages"][2]["enabled"] = False
    model = FeaturePipeline(spec, num_species=1)
    h = torch.randn(4, 6)
    _, slots = model({"h": h}, return_trace=True)
    # A disabled residual passes the update; a disabled block its first input.
    torch.testing.assert_close(slots["r"], slots["a"])
    torch.testing.assert_close(slots["out"], slots["a"])
    record = {r.id: r for r in model.records}
    assert not record["res"].enabled and not record["blk"].enabled
    assert sum(p.numel() for p in model.transforms["res"].parameters()) == 0
    with pytest.raises(ValueError, match="enabled must be true or false"):
        FeaturePipeline(_pipeline([{**stages[0], "enabled": "no"}], {}), num_species=1)


def test_bypass_names_the_input_a_disabled_stage_or_instance_passes():
    two = {
        "inputs": ["first", "second"],
        "outputs": ["y"],
        "stages": [
            {
                "id": "s",
                "transform": "add",
                "inputs": ["first", "second"],
                "outputs": ["y"],
            }
        ],
    }
    stages = [
        _linear("a", "h", "a", irreps_out="3x0e+1x1o"),
        _linear("b", "h", "b", irreps_out="3x0e+1x1o"),
        {
            "id": "sum",
            "transform": "add",
            "inputs": ["a", "b"],
            "outputs": ["s"],
            "enabled": False,
            "bypass": 1,
        },
        {
            "id": "pair",
            "transform": "two",
            "inputs": ["a", "b"],
            "outputs": ["p"],
            "enabled": False,
            "bypass": "second",
        },
    ]
    model = FeaturePipeline(_pipeline(stages, {"two": two}), num_species=1)
    _, slots = model({"h": torch.randn(4, 6)}, return_trace=True)
    torch.testing.assert_close(slots["s"], slots["b"])
    torch.testing.assert_close(slots["p"], slots["b"])
    stages[3]["bypass"] = "third"
    with pytest.raises(ValueError, match="bypass 'third' is not an input of two"):
        FeaturePipeline(_pipeline(stages, {"two": two}), num_species=1)
    stages[3]["bypass"] = "second"
    stages[2]["bypass"] = "nope"
    with pytest.raises(ValueError, match="bypass 'nope' is not an input of add"):
        FeaturePipeline(_pipeline(stages, {"two": two}), num_species=1)


def _ordered_block(order):
    return {
        "inputs": ["x"],
        "outputs": ["y"],
        "initialization_order": order,
        "stages": [
            _linear("first", "x", "u", irreps_out="3x0e+1x1o"),
            _linear("second", "u", "y", irreps_out="3x0e+1x1o"),
        ],
    }


def test_initialization_order_draws_parameters_in_the_declared_order():
    instance = {"id": "b", "transform": "pair", "inputs": ["h"], "outputs": ["y"]}

    def weights(order):
        torch.manual_seed(5)
        model = FeaturePipeline(
            _pipeline([instance], {"pair": _ordered_block(order)}), num_species=1
        )
        return [model.transforms[f"b/{s}"].weight.detach() for s in ("first", "second")]

    natural = weights(["first", "second"])
    swapped = weights(["second", "first"])
    torch.testing.assert_close(swapped[0], natural[1])
    torch.testing.assert_close(swapped[1], natural[0])
    # A repeated id draws again; the last draw is kept.
    redrawn = weights(["first", "second", "first"])
    torch.manual_seed(5)
    torch.randn(natural[0].numel() + natural[1].numel())
    torch.testing.assert_close(redrawn[1], natural[1])
    assert not torch.equal(redrawn[0], natural[0])
    with pytest.raises(ValueError, match="omits the parameterized stages"):
        weights(["first"])
    with pytest.raises(ValueError, match="unknown stages"):
        weights(["first", "third"])


def test_block_libraries_are_imported_and_recipe_blocks_override_them():
    from net.block_library import library_blocks, merged_blocks

    library = library_blocks(["deeph_e3"])
    assert library["e3_node_update"]["library"] == "deeph_e3"
    own = {"e3_node_update": {"inputs": [], "outputs": [], "stages": []}}
    merged = merged_blocks({"block_libraries": ["deeph_e3"], "blocks": own})
    assert merged["e3_node_update"] is own["e3_node_update"]
    assert "e3_edge_update" in merged
    with pytest.raises(ValueError, match="Unknown block library"):
        library_blocks(["no_such_library"])


def test_block_port_contracts_follow_the_inner_consumers():
    from net.blocks import block_ports

    ports = block_ports({"gate": BLOCK})["gate"]
    assert [p["name"] for p in ports["inputs"]] == ["features"]
    assert ports["inputs"][0]["kind"] == "irreps"
    assert ports["outputs"][0]["name"] == "out"


def test_recipes_may_tighten_but_not_change_built_in_input_contracts():
    stage = {
        "id": "a",
        "transform": "concat",
        "inputs": ["sub_edge_ang", "sub_edge_ang"],
        "outputs": ["o"],
    }
    angles = {
        "domain": "bond_neighbor",
        "representation": "local_components",
        "frame": "bond",
    }
    model = FeaturePipeline(
        {
            "inputs": {"sub_edge_ang": {**angles, "width": 9}},
            "stages": [stage],
            "outputs": {"o": "o"},
        },
        num_species=1,
    )
    assert model.feature_specs["o"].dim == 18
    with pytest.raises(ValueError, match="Cannot redefine built-in input contract"):
        FeaturePipeline(
            {
                "inputs": {
                    "node_one_hot": {"domain": "node", "irreps": "1x1o", "width": 3}
                },
                "stages": [stage],
                "outputs": {"o": "o"},
            },
            num_species=3,
        )


def test_bypass_still_checks_options_and_rejects_null_switches():
    stage = {
        "id": "a",
        "transform": "irrep_linear",
        "inputs": ["h"],
        "outputs": ["o"],
        "options": {"bogus": 1},
        "enabled": False,
    }
    with pytest.raises(ValueError, match="Unknown options"):
        FeaturePipeline(_pipeline([stage], {}), num_species=1)
    block = {
        "inputs": ["x"],
        "outputs": ["y"],
        "options": {"on": None},
        "stages": [{**_linear("l", "x", "y"), "enabled": "$on"}],
    }
    instance = {"id": "b", "transform": "blk", "inputs": ["h"], "outputs": ["y"]}
    with pytest.raises(ValueError, match="enabled must be true or false"):
        FeaturePipeline(_pipeline([instance], {"blk": block}), num_species=1)


def test_static_only_compilation_leaves_the_random_state_untouched():
    spec = {
        "stages": [
            {
                "id": "e",
                "transform": "embedding",
                "inputs": ["node_type_idx"],
                "outputs": ["e"],
                "options": {"width": 4},
            }
        ],
        "outputs": {"e": "e"},
    }
    before = torch.random.get_rng_state()
    FeaturePipeline(spec, num_species=2, static_only=True)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_a_relation_port_receives_its_lattice_shifts():
    # edge_roles reads edge_index; the compiler passes edge_shift with it.
    stage = {
        "id": "roles",
        "transform": "edge_roles",
        "inputs": ["edge_index"],
        "outputs": ["roles"],
    }
    model = FeaturePipeline(_pipeline([stage], {}), num_species=1)
    record = model.records[0]
    assert record.inputs == ("edge_index", "edge_shift")
    edge_index = torch.tensor([[0, 0, 1], [0, 0, 0]])
    edge_shift = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    _, slots = model(
        {"edge_index": edge_index, "edge_shift": edge_shift}, return_trace=True
    )
    assert slots["roles"].tolist() == [0, 1, 2]
    # A recipe that lists the shift explicitly is taken as written.
    explicit = FeaturePipeline(
        _pipeline([{**stage, "inputs": ["edge_index", "edge_shift"]}], {}),
        num_species=1,
    )
    assert explicit.records[0].inputs == ("edge_index", "edge_shift")
    with pytest.raises(ValueError, match="has no lattice shifts"):
        FeaturePipeline(
            _pipeline([{**stage, "inputs": ["edge_batch"]}], {}), num_species=1
        )
