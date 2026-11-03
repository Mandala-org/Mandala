"""The modular MANDALA model reproduces the monolithic implementation.

``tests/fixtures/legacy_mandala`` holds weights and outputs of the monolithic
model (captured before the decomposition) for three configurations that
together exercise every branch of the old update blocks and readout. Mapped
onto the stage weights, the modular model must reproduce them to round-off.
"""

import sys
from pathlib import Path

import pytest
import torch

from net.legacy_weights import translate_monolithic_state_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "legacy_mandala"
sys.path.insert(0, str(FIXTURES))
import capture  # noqa: E402

pytestmark = pytest.mark.physics


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("variant", sorted(capture.VARIANTS))
def test_modular_model_reproduces_monolithic_outputs(variant):
    reference = torch.load(FIXTURES / f"{variant}.pt", weights_only=False)
    model = capture.build(variant).eval()
    mapped = translate_monolithic_state_dict(
        reference["state_dict"],
        separate_shifted_self=model.cfg.separate_shifted_self,
        onsite_from_nodes=model.cfg.head_use_node_embeddings_for_self_edges,
    )
    assert set(mapped) == set(model.state_dict())
    assert sum(
        v.numel() for k, v in mapped.items() if k in dict(model.named_parameters())
    ) == sum(p.numel() for p in model.parameters())
    model.load_state_dict(mapped, strict=True)
    with torch.no_grad():
        predictions = model(capture.inputs(model))
    for operator, pairs in reference["outputs"].items():
        assert set(predictions[operator].pair_vectors) == set(pairs)
        for pair, expected in pairs.items():
            torch.testing.assert_close(
                predictions[operator].pair_vectors[pair],
                expected,
                rtol=1e-10,
                atol=1e-11,
            )


def test_flat_configuration_lowers_to_blocks_of_registered_stages():
    from net.component_defaults import default_graph
    from net.registry import registered_transforms

    model = capture.build("variant_b")
    graph = default_graph(model.cfg, 2)
    assert set(graph["blocks"]) == {
        "edge_encoder",
        "node_update",
        "edge_update",
        "readout",
    }
    used = {
        stage["transform"]
        for block in graph["blocks"].values()
        for stage in block["stages"]
    }
    used |= {stage["transform"] for stage in graph["stages"]} - set(graph["blocks"])
    assert used <= set(registered_transforms())
    groups = model.pipeline.groups
    assert {"node_update_0", "edge_update_1", "readout_hamiltonian"} <= set(groups)
    assert all(record.group for record in model.pipeline.records if "/" in record.id)


def test_distance_encoder_with_layer_norm_and_pair_radius_lengths_compile():
    from net.component_defaults import default_graph
    from net.common import Config

    graph = default_graph(Config(edge_encoder_style="distance", e3layernorm=True), 2)
    encoder = [
        stage["transform"] for stage in graph["blocks"]["edge_encoder"]["stages"]
    ]
    assert encoder == ["scalar_linear", "layer_norm"]
    graph = default_graph(
        Config(
            pair_distance_normalization="pair_r0",
            hamiltonian_envelope_path="envelopes.json",
        ),
        2,
    )
    envelope, ratio, radial = graph["stages"][:3]
    assert envelope["transform"] == "pair_envelope"
    assert envelope["outputs"] == ["edge_envelope", "edge_r0"]
    assert ratio["transform"] == "length_ratio"
    assert ratio["inputs"] == ["edge_lengths", "edge_r0"]
    assert radial["inputs"] == ["relative_lengths"]
    assert radial["options"]["end"] == 1.0
    assert "inputs" not in graph
    assert graph["outputs"]["edge_envelope"] == "edge_envelope"
