"""The pair envelope as pipeline stages.

``pair_envelope`` evaluates the fitted table per edge and ``envelope_scale``
multiplies the predicted blocks; this must equal scaling the predictions of
the same network afterwards (the former, model-level implementation).
"""

import json
import sys
from pathlib import Path

import pytest
import torch

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_mandala"
sys.path.insert(0, str(FIXTURES))
import capture  # noqa: E402

THETA = {
    "H-H": [0.1, -0.7, 0.3, 2.5, -1.0],
    "H-O": [-0.2, -0.4, 0.1, 2.9, -0.8],
    "O-O": [0.3, -0.5, 0.2, 3.1, -0.9],
}


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "envelope.json"
    pairs = {key: {"theta": theta} for key, theta in THETA.items()}
    path.write_text(json.dumps({"family": "slater_soft_cutoff", "pairs": pairs}))
    return path


def _model(**options):
    from core.block_irrep_mapper import BlockIrrepMapper
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from net.common import Config
    from net.e3gnn import E3GNN

    cfg = Config(**{**capture.BASE, **capture.VARIANTS["variant_b"], **options})
    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict(capture.SHELLS), dtype=torch.float64
    )
    torch.manual_seed(3)
    return E3GNN(mapper, cfg).double().eval()


def _expected_envelope(model, x, path):
    from data.envelope import build_edge_envelope, load_pair_envelope_table

    order = model.mapper.orbital_cfg.elements()
    table = load_pair_envelope_table(
        path,
        pair_order=[f"{a}-{b}" for a in order for b in order],
        dtype=torch.float64,
    )
    return build_edge_envelope(x["edge_length"], x["edge_type_idx"], table)


def test_envelope_stage_scales_predictions_like_the_model_level_envelope(table):
    from data.envelope import scale_block_matrix_by_edge_values

    plain = _model()
    scaled = _model(
        hamiltonian_envelope_path=str(table),
        hamiltonian_envelope_mode="multiply_prediction",
    )
    assert scaled.state_dict().keys() == plain.state_dict().keys()
    scaled.load_state_dict(plain.state_dict())
    x = capture.inputs(plain)
    with torch.no_grad():
        raw = plain.predicted_irreps_to_block_matrices(plain(x), x)
        physical = scaled.predicted_irreps_to_block_matrices(scaled(x), x)
        normalized = scaled.predicted_irreps_to_block_matrices(
            scaled(x), x, physical=False
        )
    envelope = _expected_envelope(plain, x, table)
    torch.testing.assert_close(
        scaled._last_outputs["edge_envelope"], envelope, rtol=0, atol=0
    )
    for operator in ("hamiltonian", "overlap"):
        expected = scale_block_matrix_by_edge_values(
            raw[operator], envelope, x["edge_partitions"]
        )
        for key, blocks in expected.pair_blocks.items():
            torch.testing.assert_close(
                physical[operator].pair_blocks[key], blocks, rtol=1e-13, atol=1e-15
            )
            torch.testing.assert_close(
                normalized[operator].pair_blocks[key],
                raw[operator].pair_blocks[key],
                rtol=1e-12,
                atol=1e-14,
            )


def test_target_normalization_compares_predictions_relative_to_the_envelope(table):
    model = _model(
        hamiltonian_envelope_path=str(table),
        hamiltonian_envelope_mode="normalize_target",
    )
    x = capture.inputs(model)
    with torch.no_grad():
        prediction = model.predicted_irreps_to_block_matrices(model(x), x)[
            "hamiltonian"
        ]
    target = prediction
    loss_prediction, loss_target, physical = model._apply_matrix_envelope_mode(
        name="hamiltonian", pred_matrix=prediction, target_matrix=target, x=x
    )
    assert physical is prediction
    envelope = _expected_envelope(model, x, table)
    for key, blocks in prediction.pair_blocks.items():
        rows = envelope.index_select(0, x["edge_partitions"][key]["global_idx"])
        torch.testing.assert_close(
            loss_prediction.pair_blocks[key], blocks / rows[:, None, None]
        )
        torch.testing.assert_close(
            loss_target.pair_blocks[key], blocks / rows[:, None, None]
        )


def test_pair_radius_lengths_come_from_the_envelope_stage(table):
    model = _model(
        hamiltonian_envelope_path=str(table), pair_distance_normalization="pair_r0"
    )
    x = capture.inputs(model)
    seen = {}
    ratio = model.pipeline.transforms["relative_lengths"]
    ratio.register_forward_hook(
        lambda module, args, output: seen.update(
            lengths=args[0], r0=args[1], ratio=output
        )
    )
    with torch.no_grad():
        model(x)
    radius = torch.tensor(
        [THETA["H-H"][3], THETA["H-O"][3], THETA["H-O"][3], THETA["O-O"][3]]
    )
    assert list(model.mapper.orbital_cfg.elements()) == ["H", "O"]
    torch.testing.assert_close(seen["r0"], radius[x["edge_type_idx"]])
    torch.testing.assert_close(seen["ratio"], seen["lengths"] / seen["r0"])


def test_flat_envelope_settings_add_stages_to_an_explicit_recipe(table):
    from net.common import Config
    from net.component_defaults import with_envelope

    pipeline = {
        "stages": [{"id": "a", "transform": "x", "inputs": [], "outputs": ["m"]}],
        "outputs": {"hamiltonian": "m"},
    }
    cfg = Config(
        hamiltonian_envelope_path=str(table),
        hamiltonian_envelope_mode="multiply_prediction",
    )
    graph = with_envelope(pipeline, cfg)
    assert [s["transform"] for s in graph["stages"]] == [
        "pair_envelope",
        "x",
        "envelope_scale",
    ]
    assert graph["outputs"] == {
        "hamiltonian": "m_physical",
        "edge_envelope": "edge_envelope",
    }
    assert with_envelope(graph, cfg) is graph
    assert with_envelope(pipeline, Config()) is pipeline


def test_batched_graphs_read_their_own_envelope_slice(table):
    from data.graph_topology import prepare_graph_topology
    from data.sample import union_graph_inputs

    model = _model(
        hamiltonian_envelope_path=str(table),
        hamiltonian_envelope_mode="normalize_target",
    )
    first = capture.inputs(model)
    shifted = capture.POSITIONS[1:] + capture.POSITIONS[:1]
    second = capture.build_model_input_from_structure(
        atoms=capture.ATOMS[1:] + capture.ATOMS[:1],
        positions=torch.tensor(shifted, dtype=torch.float64) * 1.05,
        box=torch.tensor(capture.BOX, dtype=torch.float64),
        cfg=model.cfg,
        mapper=model.mapper,
    )
    single = []
    with torch.no_grad():
        for x in (first, second):
            model(x)
            single.append(model._envelope(x).clone())
        graphs = [(prepare_graph_topology(dict(x), None), {}) for x in (first, second)]
        union = union_graph_inputs(graphs, layouts=model.pipeline.input_layouts)
        predictions = model(union)
        parts = model._split_batch_predictions(predictions, graphs, union)
    assert len(single[0]) != len(single[1]) or not torch.equal(single[0], single[1])
    for ((x, _), _), expected in zip(parts, single):
        torch.testing.assert_close(model._envelope(x), expected)
