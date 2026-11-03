"""Checkpoints of the monolithic model restore into the modular model."""

import sys
from pathlib import Path

import pytest
import torch

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_mandala"
sys.path.insert(0, str(FIXTURES))
import capture  # noqa: E402

MONOLITHIC_PIPELINE = {
    "stages": [
        {
            "id": "node_encoding",
            "transform": "species_irrep_embedding",
            "inputs": ["node_type_idx"],
            "outputs": ["node_0"],
        },
    ],
    "outputs": {"hamiltonian": "hamiltonian_vectors"},
}


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _assert_reproduces(model, reference):
    with torch.no_grad():
        predictions = model(capture.inputs(model))
    for operator, pairs in reference["outputs"].items():
        for pair, expected in pairs.items():
            torch.testing.assert_close(
                predictions[operator].pair_vectors[pair], expected
            )


def test_saved_monolithic_lowering_and_weights_restore(tmp_path):
    from net.e3gnn import E3GNN

    reference = torch.load(FIXTURES / "paper.pt", weights_only=False)
    template = capture.build("paper")
    cfg = template.cfg
    cfg.pipeline = MONOLITHIC_PIPELINE
    restored = E3GNN(template.mapper, cfg).double().eval()
    assert "readout_hamiltonian/project" in restored.pipeline.transforms
    restored.load_state_dict(reference["state_dict"], strict=True)
    _assert_reproduces(restored, reference)


def test_compatibility_initialization_translates_monolithic_checkpoints(tmp_path):
    from net.checkpoint_compat import initialize_model_from_checkpoint

    reference = torch.load(FIXTURES / "paper.pt", weights_only=False)
    model = capture.build("paper").eval()
    path = tmp_path / "legacy.ckpt"
    torch.save(
        {"state_dict": reference["state_dict"], "hyper_parameters": {"cfg": model.cfg}},
        path,
    )
    report = initialize_model_from_checkpoint(model, path)
    assert not report.skipped_keys
    _assert_reproduces(model, reference)


def test_compatibility_maps_readout_routes_by_role_name():
    from net.checkpoint_compat import remap_role_routes

    head = "pipeline.transforms.readout_hamiltonian/"
    source = {
        head + "trunk.routes.1.w": "shifted_self",
        head + "trunk.routes.2.w": "offsite",
        head + "project.projections.2_O_H.w": "offsite_projection",
        head + "project.shared.0.w": "onsite_shared",
    }
    target = remap_role_routes(source, source_separate=True, target_separate=False)
    assert target == {
        head + "trunk.routes.1.w": "offsite",
        head + "project.projections.1_O_H.w": "offsite_projection",
        head + "project.shared.0.w": "onsite_shared",
    }
