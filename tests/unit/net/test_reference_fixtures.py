"""Offline source-generated regression oracles; no upstream checkout is needed.

The full default stage count is retained at small widths. Probes cover every
parameter's gradient/update using three coordinates and two aggregate moments;
direct-source tests additionally compare every parameter coordinate.
"""

import json
from pathlib import Path

import pytest
import torch

from net.pipeline import FeaturePipeline
from net.training_policies import masked_error


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/reference_operators"


def _tensor(value):
    if "real" in value:
        return (
            torch.complex(torch.tensor(value["real"]), torch.tensor(value["imag"]))
            .to(getattr(torch, value["dtype"]))
            .reshape(value["shape"])
        )
    return torch.tensor(value["values"], dtype=getattr(torch, value["dtype"])).reshape(
        value["shape"]
    )


def _assert_probe(parameter, expected):
    if parameter is None or expected is None:
        assert parameter is expected
        return
    assert list(parameter.shape) == expected["shape"]
    flat = parameter.detach().double().reshape(-1)
    if not flat.numel():
        assert not expected["values"]
        return
    # CPU backend/library updates can move final digits; the tested tolerance
    # is numerical fidelity, not a cross-version bitwise guarantee.
    torch.testing.assert_close(
        flat[expected["indices"]],
        torch.tensor(expected["values"]),
        rtol=2e-8,
        atol=2e-10,
    )
    torch.testing.assert_close(
        flat.sum(), torch.tensor(expected["sum"]), rtol=2e-8, atol=2e-9
    )
    torch.testing.assert_close(
        flat.square().sum(), torch.tensor(expected["squared_sum"]), rtol=2e-8, atol=2e-9
    )


def _source_name(kind, name):
    _, stage, tail = name.split(".", 2)
    if kind == "deeph_e3":
        prefix = {
            "embedding": "embedding",
            "node_readout": "lin_node",
            "edge_readout": "lin_edge",
        }.get(stage)
        if stage.startswith("node_update_"):
            prefix = f"node_update_blocks.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("edge_update_"):
            prefix = f"edge_update_blocks.{stage.rsplit('_', 1)[-1]}"
    else:
        prefix = {
            "node_embedding": "mace.node_embedding.linear",
            "density_radial_basis": "mace.radial_embedding",
            "edge_readout": "edge_update.lin_edge",
        }.get(stage)
        if stage.startswith("density_") and stage.rsplit("_", 1)[-1].isdigit():
            prefix = f"mace.interactions.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("product_"):
            prefix = f"mace.products.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("angular_expansion_"):
            prefix = f"expansion.node_order_expansion_blocks.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("edge_update_"):
            prefix = f"edge_update.edge_update_blocks.{stage.rsplit('_', 1)[-1]}"
    assert prefix is not None, f"Unmapped parameter {name}"
    return f"{prefix}.{tail}"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_offline_default_topology_outputs_gradients_and_three_fresh_adam_steps(path):
    record = json.loads(path.read_text())
    previous_dtype = torch.get_default_dtype()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(
            getattr(torch, record["runtime"]["initialization_dtype"])
        )
        torch.manual_seed(record["seed"])
        model = FeaturePipeline(
            record["configuration"]["pipeline"], num_species=1
        ).double()
        torch.set_default_dtype(torch.float64)
        parameters = list(model.named_parameters())
        if record["kind"] == "deeph":
            source_names = list(record["initial_parameters"])
            assert len(parameters) == len(source_names)
            coordinates = {
                source: p for source, (_, p) in zip(source_names, parameters)
            }
        else:
            coordinates = {
                _source_name(record["kind"], name): p for name, p in parameters
            }
        assert coordinates.keys() == record["initial_parameters"].keys()
        for name, parameter in coordinates.items():
            _assert_probe(parameter, record["initial_parameters"][name])
        if record["statistics"]:
            stats = record["statistics"]
            affine = model.transforms["output_affine"]
            with torch.no_grad():
                affine.mean.copy_(_tensor(stats["mean"])[None])
                affine.std.copy_(_tensor(stats["std"])[None])
                affine.fitted.fill_(True)
        inputs = {k: _tensor(v) for k, v in record["inputs"].items()}
        target, mask = _tensor(record["target"]), _tensor(record["mask"])
        optimizer = torch.optim.Adam(model.parameters(), **record["optimizer"])
        for step in record["steps"]:
            optimizer.zero_grad()
            prediction = model(inputs)["hamiltonian"]
            torch.testing.assert_close(
                prediction, _tensor(step["output"]), rtol=2e-8, atol=2e-10
            )
            loss = masked_error(prediction, target, mask, kind="mse")
            torch.testing.assert_close(
                loss, torch.tensor(step["loss"]), rtol=2e-9, atol=2e-10
            )
            loss.backward()
            for name, parameter in coordinates.items():
                _assert_probe(parameter.grad, step["gradients"][name])
            optimizer.step()
            for name, parameter in coordinates.items():
                _assert_probe(parameter, step["updated_parameters"][name])
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


def test_frozen_fixture_provenance_and_scope():
    paths = sorted(FIXTURES.glob("*.json"))
    assert {p.stem for p in paths} >= {"deeph", "deeph_e3", "mace_h"}
    for path in paths:
        assert path.stat().st_size < 1_000_000
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        assert len(data["source"]["commit"]) == 40
        assert data["source"]["sha256"]
        assert all(len(value) == 64 for value in data["source"]["sha256"].values())
        assert data["runtime"]["device"] == "cpu"
        assert len(data["steps"]) == 3
