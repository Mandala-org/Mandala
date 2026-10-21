"""Default five-layer local-frame network, compared to the pinned executable."""

from pathlib import Path
import importlib
import os
import sys

import pytest
import torch

from data.bond_geometry import bond_frames, bond_neighborhoods
from net.configuration import resolve_configuration
from net.pipeline import FeaturePipeline


def tiny_configuration():
    return resolve_configuration(
        Path(__file__).resolve().parents[3] / "configs/reference/deeph.yaml",
        species_shells={"H": [0]},
        overrides={
            "parameters": {
                "node_width": 4,
                "edge_width": 8,
                "post_edge_width": 4,
                "angular_dim": 4,
                "num_l": 2,
            }
        },
    )


def graph():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.1, 0.3]])
    index = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    vectors = positions[index[1]] - positions[index[0]]
    frames = bond_frames(index, vectors)
    return {
        "node_type_idx": torch.zeros(3, dtype=torch.long),
        "node_batch": torch.zeros(3, dtype=torch.long),
        "edge_index": index,
        "edge_lengths": vectors.norm(dim=-1),
        "edge_vectors": vectors,
        **bond_neighborhoods(index, vectors, frames, num_l=2),
    }, positions


def test_complete_local_coordinate_pipeline_forward_backward_and_reload():
    cfg, codec = tiny_configuration()
    model = FeaturePipeline(cfg["pipeline"], num_species=1)
    x, _ = graph()
    prediction = model(x)["hamiltonian"]
    assert prediction.shape == (9, 1)
    prediction.square().mean().backward()
    active = [(name, p) for name, p in model.named_parameters() if ".bn." not in name]
    assert all(p.grad is not None for _, p in active)
    restored = FeaturePipeline(cfg["pipeline"], num_species=1)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x)["hamiltonian"], prediction)


def test_complete_local_coordinate_default_topology_against_upstream():
    root = os.environ.get("MANDALA_DEEPH_REFERENCE")
    if not root:
        pytest.skip("Set MANDALA_DEEPH_REFERENCE to the pinned DeepH checkout")
    sys.path.insert(0, root)
    source = importlib.import_module("deeph.model")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        cfg, _ = tiny_configuration()
        torch.manual_seed(42)
        ours = FeaturePipeline(cfg["pipeline"], num_species=1)
        torch.manual_seed(42)
        original = source.HGNN(
            1,
            4,
            8,
            1,
            "GaussianBasis",
            6.0,
            True,
            False,
            True,
            True,
            "LayerNorm",
            "CGConv",
            False,
            False,
            False,
            num_l=2,
        )
        # The source Gaussian constructor explicitly uses FloatTensor. Match
        # native initialization in float32, then use double for comparisons.
        ours.double()
        original.double()
        torch.set_default_dtype(torch.float64)
        for layer in original.modules():
            if isinstance(layer, source.CGConv):
                # Old propagate_type omits distance; use unchanged generic PyG
                # dispatcher instead of its incompatible generated signature.
                layer.propagate = layer._orig_propagate
        assert len(list(ours.parameters())) == len(list(original.parameters()))
        for a, b in zip(ours.parameters(), original.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x, positions = graph()
        edge_attr = torch.cat(
            [
                x["edge_lengths"][:, None],
                positions[x["edge_index"][0]],
                positions[x["edge_index"][1]],
            ],
            -1,
        )
        a_opt = torch.optim.Adam(ours.parameters(), lr=0.001)
        b_opt = torch.optim.Adam(original.parameters(), lr=0.001)
        for step in range(3):
            a_opt.zero_grad()
            b_opt.zero_grad()
            a = ours(x)["hamiltonian"]
            b = original(
                x["node_type_idx"],
                x["edge_index"],
                edge_attr,
                x["node_batch"],
                x["sub_atom_idx"],
                x["sub_edge_idx"],
                x["sub_edge_ang"],
                x["sub_index"],
            )
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
            a.square().mean().backward()
            b.square().mean().backward()
            for pa, pb in zip(ours.parameters(), original.parameters()):
                assert (pa.grad is None) == (pb.grad is None)
                if pa.grad is not None:
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-8, atol=1e-10)
            a_opt.step()
            b_opt.step()
            for pa, pb in zip(ours.parameters(), original.parameters()):
                torch.testing.assert_close(pa, pb, rtol=1e-9, atol=1e-10)
    finally:
        torch.set_default_dtype(previous)
        sys.path.remove(root)
